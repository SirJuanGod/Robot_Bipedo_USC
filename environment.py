"""
Entorno de entrenamiento para el robot bípedo Bipedo.xml
con currículo de marcha walk → run.

Nomenclatura del MJCF (onshape-to-robot):
  Torso root   : cbh_d_2
  Pie derecho  : ft_d    | Pie izquierdo: ft_i
  Tobillo D    : ank_d   | Tobillo I    : ank_i
  Rodilla D    : kne_d   | Rodilla I    : kne_i
  Pierna D(seg): leg_i   | Pierna I(seg): leg_d  (nombres cruzados en MJCF)
  Cadera / tronco superior: cbh_d_2

Joints actuados (13):
  Torso/brazo : BD, HD, HI, BI, HEAD
  Pierna D    : CD, LD, KD, FD
  Pierna I    : CI, LI, KI, FI
"""

import torch
import genesis as gs

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    EntityManager,
    ObservationManager,
    ActuatorManager,
    PositionActionManager,
    VelocityCommandManager,
    ContactManager,
)
from genesis_forge.mdp import reset, rewards, terminations, observations
from gait_command_manager import BipedGaitCommandManager

# ── Configuración inicial ──────────────────────────────────────────────────────
# El robot mide ~0.2340 m de altura en posición de pie (cbh_d_2 como root).
HEIGHT_OFFSET                = 0.2340
INITIAL_BODY_POSITION        = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT                 = [1.0, 0.0, 0.0, 0.0]
CURRICULUM_CHECK_EVERY_STEPS = 100

# ── Escalas de ruido sim2real para las observaciones del actor ─────────────────
NOISE_SCALES = {
    "ang_vel": 0.05,   # rad/s  — giroscopio IMU
    "gravity":  0.05,  # u.n.   — gravedad proyectada
    "dof_pos":  0.01,  # rad    — encoder de posición
    "dof_vel":  0.10,  # rad/s  — encoder de velocidad
}


def _noisy(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Ruido gaussiano aditivo para simulación de sensores reales."""
    return tensor + torch.randn_like(tensor) * scale


# ──────────────────────────────────────────────────────────────────────────────
class BipedGaitTrainingEnv(ManagedEnvironment):
    """
    Entorno de entrenamiento para bípedo con currículo walk → run.
    """

    def __init__(
        self,
        num_envs:             int        = 1,
        dt:                   float      = 1 / 50,
        max_episode_length_s: int | None = 20,
        headless:             bool       = True,
        gamepad_control:      bool       = False,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.2,
        )
        self._gamepad_control = gamepad_control
        self._next_curriculum_check_step = CURRICULUM_CHECK_EVERY_STEPS

        # ── Escena ────────────────────────────────────────────────────────────
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
        )

        self.terrain = self.scene.add_entity(gs.morphs.Plane())

        # ── Robot bípedo (MJCF) ───────────────────────────────────────────────
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="model/Bipedo.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        self.camera = self.scene.add_camera(
            pos=(2.5, 0.0, 2.5),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
            GUI=self._gamepad_control,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def config(self):

        # ── Caché de idx_local para las lambdas del critic ────────────────────
        # Se resuelven aquí (en config) porque obs.build() se llama dentro de
        # super().build() — antes de que nuestro build() propio pueda ejecutarse.
        # El robot ya existe en self.robot desde __init__, así que get_link()
        # funciona correctamente en este punto.
        self._critic_foot_idx  = [self.robot.get_link("ft_d").idx_local,
                                   self.robot.get_link("ft_i").idx_local]
        self._critic_ankle_idx = [self.robot.get_link("ank_d").idx_local,
                                   self.robot.get_link("ank_i").idx_local]
        self._critic_knee_idx  = [self.robot.get_link("kne_d").idx_local,
                                   self.robot.get_link("kne_i").idx_local]

        # ── Reset del robot ───────────────────────────────────────────────────
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.position,
                    "params": {
                        "position":      INITIAL_BODY_POSITION,
                        "quat":          INITIAL_QUAT,
                        "zero_velocity": True,
                    },
                },
            },
        )

        # ── Actuadores ────────────────────────────────────────────────────────
        # Joints del MJCF agrupados por función:
        #   Torso/brazo: BD (hombro D), HD (cadera D), HI (cadera I), BI (hombro I), HEAD
        #   Pierna D   : CD (cadera D abductora), LD (muslo D), KD (rodilla D), FD (pie D)
        #   Pierna I   : CI (cadera I abductora), LI (muslo I), KI (rodilla I), FI (pie I)
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=["BD", "HD", "HI", "BI", "HEAD",
                         "CD", "LD", "KD", "FD",
                         "CI", "LI", "KI", "FI"],
            kp=50.0,     # igual al kp del MJCF (default class → position kp="50")
            kv=1.0,
            default_pos={
                # Torso y cabeza en posición neutra
                "BD":   0.0,
                "HD":   0.0,
                "HI":   0.0,
                "BI":   0.0,
                "HEAD": 0.0,
                # Postura de pie: cadera ligeramente flexionada,
                # rodilla extendida, pie neutro
                "CD":  0.0,
                "LD": -0.2,
                "KD":  0.4,
                "FD": -0.2,
                "CI":  0.0,
                "LI": -0.2,
                "KI":  0.4,
                "FI": -0.2,
            },
            max_force={
                "BD":   5.0,
                "HD":   5.0,
                "HI":   5.0,
                "BI":   5.0,
                "HEAD": 2.0,
                "CD":  15.0,
                "LD":  20.0,
                "KD":  20.0,
                "FD":  10.0,
                "CI":  15.0,
                "LI":  20.0,
                "KI":  20.0,
                "FI":  10.0,
            },
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.4,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        # ── Contactos ─────────────────────────────────────────────────────────
        # Torso raíz — terminación si toca el suelo
        self.torso_contact_manager = ContactManager(
            self,
            link_names=["cbh_d_2"],
        )
        # Pies — recompensa de tiempo de vuelo y fase de marcha
        self.feet_contact_manager = ContactManager(
            self,
            link_names=["ft_d", "ft_i"],
            track_air_time=True,
            air_time_contact_threshold=1.0,
        )
        # Rodillas y tobillos — penalización por contacto indeseado
        self.knee_contact_manager = ContactManager(
            self,
            link_names=["kne_d", "kne_i", "ank_d", "ank_i"],
        )
        # Segmentos de pierna (leg_d, leg_i) — contacto estructural
        self.leg_contact_manager = ContactManager(
            self,
            link_names=["leg_d", "leg_i"],
        )

        # ── Comando de velocidad ──────────────────────────────────────────────
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [0.0, 1.5],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_z": [-0.5, 0.5],
            },
            standing_probability=0.05,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={"envs_idx": [0], "arrow_offset": 0.12},
        )

        # ── Gestor de marcha (walk / run) ─────────────────────────────────────
        # ft_i = pie izquierdo (L), ft_d = pie derecho (R)
        self.gait_command_manager = BipedGaitCommandManager(
            self,
            foot_names={
                "L": "ft_i",   # pie izquierdo
                "R": "ft_d",   # pie derecho
            },
            resample_time_sec=5.0,
        )

        # ── Recompensas ───────────────────────────────────────────────────────
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={ # type: ignore
                "gait_phase_reward": {
                    "weight": 1.5,
                    "fn": self.gait_command_manager.gait_phase_reward,
                    "params": {"contact_manager": self.feet_contact_manager},
                },
                "foot_height_reward": {
                    "weight": 0.9,
                    "fn": self.gait_command_manager.foot_height_reward,
                },
                "tracking_lin_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager":  self.robot_manager,
                    },
                },
                "tracking_ang_vel": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager":  self.robot_manager,
                    },
                },
                "feet_air_time": {
                    "weight": 2.0,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "time_threshold":     0.2,
                        "time_threshold_max": 0.5,
                        "contact_manager":    self.feet_contact_manager,
                        "vel_cmd_manager":    self.velocity_command,
                    },
                },
                "lin_vel_z": {
                    "weight": -2.0,
                    "fn": rewards.lin_vel_z_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "ang_vel_xy_l2": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "body_acceleration": {
                    "weight": -0.1,
                    "fn": rewards.body_acceleration_exp,
                    "params": {"entity_manager": self.robot_manager},
                },
                "base_height_target": {
                    "weight": -20.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": HEIGHT_OFFSET - 0.05,
                        "entity_attr":   "robot",
                    },
                },
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate_l2,
                },
                "similar_to_default": {
                    "weight": -0.05,
                    "fn": rewards.dof_similar_to_default,
                    "params": {"action_manager": self.action_manager},
                },
                "bad_contact": {
                    "weight": -1.5,
                    "fn": rewards.contact_force,
                    "params": {"contact_manager": self.knee_contact_manager},
                },
            },
        )

        # ── Terminación ───────────────────────────────────────────────────────
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={ # type: ignore
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                "torso_contact": {
                    "fn": terminations.contact_force,
                    "params": {"contact_manager": self.torso_contact_manager},
                },
                "fall_over": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle":    25.0,
                        "entity_manager": self.robot_manager,
                    },
                },
            },
        )

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║  OBSERVACIONES DEL ACTOR (policy)                                  ║
        # ║  Solo lo que un robot real puede medir — sin cambios.              ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        ObservationManager(
            self,
            name="policy",
            history_len=5,
            cfg={ # type: ignore
                "velocity_cmd": {
                    "fn": self.velocity_command.observation,
                },
                "gait_command": {
                    "fn": self.gait_command_manager.observation,
                },
                "imu_ang_velocity": {
                    "fn": lambda env: _noisy(
                        self.robot_manager.get_angular_velocity(),
                        NOISE_SCALES["ang_vel"],
                    ),
                },
                "imu_projected_gravity": {
                    "fn": lambda env: _noisy(
                        self.robot_manager.get_projected_gravity(),
                        NOISE_SCALES["gravity"],
                    ),
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║  OBSERVACIONES DEL CRITIC                                          ║
        # ║  Todos los estados del simulador — completamente limpios.          ║
        # ║                                                                    ║
        # ║  Firmas verificadas con inspect.signature() sobre la versión       ║
        # ║  instalada de genesis_forge.                                       ║
        # ║                                                                    ║
        # ║  1. Velocidades del torso   (entity_linear/angular_velocity)       ║
        # ║  2. IMU limpio              (entity_projected_gravity)             ║
        # ║  3. DOF completo            (dofs_position / velocity / force)     ║
        # ║  4. Links clave             (get_links_pos / get_links_vel)        ║
        # ║  5. Fuerzas de contacto     (contact_force)                        ║
        # ║  6. Reloj de marcha         (clock_input, gait_phase)              ║
        # ║  7. Pose del torso          (base_pos, base_quat de EntityManager) ║
        # ║  8. Acciones exactas        (current_actions)                      ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        ObservationManager(
            self,
            name="critic",
            history_len=5,
            cfg={ # type: ignore

                # ── 1. Velocidades del torso ──────────────────────────────────
                # entity_linear_velocity(env, entity_manager, entity_attr)
                # entity_angular_velocity(env, entity_manager, entity_attr)
                "root_linear_velocity": {
                    "fn": observations.entity_linear_velocity,
                    "params": {"entity_manager": self.robot_manager},
                },
                "root_angular_velocity": {
                    "fn": observations.entity_angular_velocity,
                    "params": {"entity_manager": self.robot_manager},
                },

                # ── 2. IMU limpio ─────────────────────────────────────────────
                # entity_projected_gravity(env, entity_manager, entity_attr)
                "projected_gravity_clean": {
                    "fn": observations.entity_projected_gravity,
                    "params": {"entity_manager": self.robot_manager},
                },

                # ── 3. Estado articular completo (13 DOF) ────────────────────
                # entity_dofs_position(env, actuator_manager, entity_attr,
                #                      dofs_idx, action_manager)
                # entity_dofs_velocity(env, action_manager, entity_attr, dofs_idx)
                # entity_dofs_force(env, actuator_manager, entity_attr,
                #                   dofs_idx, clip_to_max_force, action_manager)
                "dof_pos": {
                    "fn": observations.entity_dofs_position,
                    "params": {"action_manager": self.action_manager},
                },
                "dof_vel": {
                    "fn": observations.entity_dofs_velocity,
                    "params": {"action_manager": self.action_manager},
                },
                "dof_force": {
                    "fn": observations.entity_dofs_force,
                    "params": {"action_manager": self.action_manager},
                    "scale": 0.1,
                },

                # ── 4. Posición y velocidad de links clave ───────────────────
                # get_links_pos / get_links_vel son métodos directos de RigidEntity
                # en genesis — no pasan por el módulo observations.
                # Se resuelven una sola vez en build() para evitar get_link()
                # en cada paso (ver _foot_link_ids en build).
                "feet_pos": {
                    # ft_d (pie D) y ft_i (pie I) — posición 3D → (N, 6)
                    "fn": lambda env: env.robot.get_links_pos(
                        links_idx_local=self._critic_foot_idx
                    ).reshape(env.num_envs, -1),
                },
                "feet_vel": {
                    # ft_d y ft_i — velocidad 3D → (N, 6)
                    "fn": lambda env: env.robot.get_links_vel(
                        links_idx_local=self._critic_foot_idx
                    ).reshape(env.num_envs, -1),
                },
                "ankle_pos": {
                    # ank_d y ank_i — posición 3D → (N, 6)
                    "fn": lambda env: env.robot.get_links_pos(
                        links_idx_local=self._critic_ankle_idx
                    ).reshape(env.num_envs, -1),
                },
                "knee_pos": {
                    # kne_d y kne_i — posición 3D → (N, 6)
                    "fn": lambda env: env.robot.get_links_pos(
                        links_idx_local=self._critic_knee_idx
                    ).reshape(env.num_envs, -1),
                },
                "knee_vel": {
                    # kne_d y kne_i — velocidad 3D → (N, 6)
                    "fn": lambda env: env.robot.get_links_vel(
                        links_idx_local=self._critic_knee_idx
                    ).reshape(env.num_envs, -1),
                },

                # ── 5. Fuerzas de contacto ────────────────────────────────────
                # contact_force(env, contact_manager) → tensor de fuerzas
                "foot_contact_force": {
                    "fn": observations.contact_force,
                    "params": {"contact_manager": self.feet_contact_manager},
                },
                "knee_contact_force": {
                    "fn": observations.contact_force,
                    "params": {"contact_manager": self.knee_contact_manager},
                },

                # ── 6. Reloj de marcha ────────────────────────────────────────
                # Buffers internos del BipedGaitCommandManager — ya son tensores
                # (N, 4) y (N, 1) respectivamente, no necesitan params.
                "gait_clock": {
                    # [sin_L, sin_R, cos_L, cos_R]
                    "fn": lambda env: self.gait_command_manager.clock_input,
                },
                "gait_phase_raw": {
                    # fase ∈ [0, 1)
                    "fn": lambda env: self.gait_command_manager.gait_phase,
                },

                # ── 7. Pose del torso ─────────────────────────────────────────
                # EntityManager expone base_pos y base_quat como propiedades
                # (NO como métodos — NO tienen paréntesis).
                "root_pos": {
                    # posición 3D del torso (cbh_d_2) → (N, 3)
                    "fn": lambda env: self.robot_manager.base_pos,
                },
                "root_quat": {
                    # cuaternión de orientación → (N, 4)
                    "fn": lambda env: self.robot_manager.base_quat,
                },

                # ── 8. Acciones actuales exactas ──────────────────────────────
                # current_actions(env, action_manager) — firma verificada.
                "current_actions": {
                    "fn": observations.current_actions,
                    "params": {"action_manager": self.action_manager},
                },
            },
        )

    # ── Hooks del ciclo de vida ────────────────────────────────────────────────
    def build(self):
        super().build()
        self.camera.follow_entity(self.robot)

    def step(self, actions: torch.Tensor):
        if self._gamepad_control:
            self.camera.render()
        return super().step(actions)

    def reset(self, envs_idx: list[int] | None = None):
        result = super().reset(envs_idx)
        if envs_idx is not None:
            self.update_curriculum()
        return result

    # ── Currículo de marcha ────────────────────────────────────────────────────
    def update_curriculum(self):
        """
        Aumenta la dificultad automáticamente:
          · gait_phase_reward > 0.75 → desbloquea run
          · foot_height_reward > 0.80 → mayor clearance
        """
        if self.step_count < self._next_curriculum_check_step:
            return
        self._next_curriculum_check_step = (
            self.step_count + CURRICULUM_CHECK_EVERY_STEPS
        )

        gait_r = self.reward_manager.last_episode_mean_reward(
            "gait_phase_reward", before_weight=True
        )
        if gait_r > 0.75:
            self.gait_command_manager.increment_num_gaits()
            self.gait_command_manager.increment_gait_period_range()

        foot_r = self.reward_manager.last_episode_mean_reward(
            "foot_height_reward", before_weight=True
        )
        if foot_r > 0.80:
            self.gait_command_manager.increment_foot_clearance_range()