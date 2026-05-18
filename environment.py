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
from genesis_forge.managers.actuator import NoisyValue
from gait_command_manager import BipedGaitCommandManager

# El robot mide ~0.2340 m de altura en posición de pie.
HEIGHT_OFFSET                = 0.2340
INITIAL_BODY_POSITION        = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT                 = [1.0, 0.0, 0.0, 0.0]
CURRICULUM_CHECK_EVERY_STEPS = 50

# Valores de ruido sim2real para las observaciones del actor policy
NOISE_SCALES = {
    "ang_vel": 0.05,   # rad/s  — giroscopio IMU
    "gravity":  0.05,  # u.n.   — gravedad proyectada
    "dof_pos": 0.01,   # rad   — encoder abs/rel típico
    "dof_vel": 0.50,   # rad/s — derivada discreta muy ruidosa en hardware real
}

# ── Parámetros físicos del robot (del MJCF) ──────────────────────────────────
ROBOT_MASS_KG     = 1.09     # masa total aproximada
ROBOT_HEIGHT_M    = 0.234    # altura de pie
LEG_LENGTH_M      = 0.084    # longitud del eslabón de pierna
GRAVITY           = 9.81


def _noisy(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor + torch.randn_like(tensor) * scale


# ═══════════════════════════════════════════════════════════════════════════════
#  Funciones de recompensa custom — normalizadas a [0, 1] o clipeadas
# ═══════════════════════════════════════════════════════════════════════════════

def reward_alive_bonus(env, **kwargs) -> torch.Tensor:
    """Bono constante por estar vivo. Incentiva supervivencia."""
    return torch.ones(env.num_envs, device=gs.device)


def reward_upright_stability(env, entity_manager, **kwargs) -> torch.Tensor:
    """
    Recompensa basada en la componente Z de la gravedad proyectada en el frame
    del cuerpo (disponible desde el IMU).
    
    Si el robot está perfectamente vertical, projected_gravity = [0, 0, -1].
    La componente Z será -1 → reward = 1.0
    Si está de lado, Z → 0 → reward ≈ 0
    Si está boca abajo, Z → +1 → reward = 0
    
    R = clamp(-g_z, 0, 1)^2   (cuadrático para penalizar más las inclinaciones)
    """
    proj_grav = entity_manager.get_projected_gravity()  # (N, 3)
    gz = proj_grav[:, 2]  # negativo cuando está de pie
    uprightness = torch.clamp(-gz, 0.0, 1.0)
    return uprightness ** 2  # [0, 1]


def reward_angular_velocity_penalty(env, entity_manager, max_val=5.0, **kwargs) -> torch.Tensor:
    """
    Penalización por velocidad angular en ejes X,Y (roll/pitch).
    Normalizada: ||ω_xy||² / max_val² , clipeada a [0, 1].
    
    max_val=5.0 rad/s es un límite razonable para un robot de 23cm.
    """
    ang_vel = entity_manager.get_angular_velocity()  # (N, 3)
    ang_vel_xy_sq = ang_vel[:, 0]**2 + ang_vel[:, 1]**2
    return torch.clamp(ang_vel_xy_sq / (max_val ** 2), 0.0, 1.0)


def reward_linear_vel_z_penalty(env, entity_manager, max_val=1.0, **kwargs) -> torch.Tensor:
    """
    Penalización por velocidad vertical del cuerpo.
    Normalizada: vz² / max_val², clipeada a [0, 1].
    """
    lin_vel = entity_manager.get_linear_velocity()  # (N, 3)
    vz_sq = lin_vel[:, 2] ** 2
    return torch.clamp(vz_sq / (max_val ** 2), 0.0, 1.0)


def reward_body_acceleration_penalty(env, entity_manager, max_val=20.0, **kwargs) -> torch.Tensor:
    """
    Penalización por aceleración del cuerpo (suavidad).
    Computa aceleración lineal como Δv/dt entre steps.
    Normalizada a [0, 1].
    """
    curr_vel = entity_manager.get_linear_velocity()  # (N, 3)
    if not hasattr(env, '_prev_lin_vel') or env._prev_lin_vel is None:
        env._prev_lin_vel = curr_vel.clone()
        return torch.zeros(env.num_envs, device=gs.device)
    
    acc = (curr_vel - env._prev_lin_vel) / env.dt  # (N, 3)
    env._prev_lin_vel = curr_vel.clone()
    acc_norm = torch.norm(acc, dim=-1)  # (N,)
    return torch.clamp(acc_norm / max_val, 0.0, 1.0)


def reward_action_rate_penalty(env, max_val=2.0, **kwargs) -> torch.Tensor:
    """
    Penalización por cambios bruscos en las acciones (suavidad).
    ||a_t - a_{t-1}||² normalizado por num_actions * max_val².
    Resultado en [0, 1].
    """
    curr = env.action_manager.get_actions()
    if not hasattr(env, '_prev_actions_reward') or env._prev_actions_reward is None:
        env._prev_actions_reward = curr.clone()
        return torch.zeros(env.num_envs, device=gs.device)
    
    prev = env._prev_actions_reward
    diff_sq = torch.sum((curr - prev) ** 2, dim=-1)  # (N,)
    num_act = curr.shape[-1]
    env._prev_actions_reward = curr.clone()
    return torch.clamp(diff_sq / (num_act * max_val ** 2), 0.0, 1.0)


def reward_energy_penalty(env, max_val=1.5, **kwargs) -> torch.Tensor:
    """
    Penalización por gasto energético (magnitud de acciones).
    Proxy de consumo de torque: ||a||² / (num_actions * max_val²).
    Resultado en [0, 1].
    """
    actions = env.action_manager.get_actions()  # (N, num_dof)
    act_sq = torch.sum(actions ** 2, dim=-1)
    num_act = actions.shape[-1]
    return torch.clamp(act_sq / (num_act * max_val ** 2), 0.0, 1.0)


def reward_contact_binary_penalty(env, contact_manager, **kwargs) -> torch.Tensor:
    """
    Penalización suave por contacto no deseado. Resultado en [0, 1).
    Usa exponencial negativa para gradiente continuo en lugar de escalón binario.
    """
    ids = contact_manager._link_ids
    if ids is None or len(ids) == 0:
        return torch.zeros(env.num_envs, device=gs.device)
    forces = contact_manager.get_contact_forces(ids.tolist())  # (N, num_links, 3)
    force_norm = torch.norm(forces, dim=-1)                    # (N, num_links)
    max_force = force_norm.max(dim=-1).values                  # (N,)
    return 1.0 - torch.exp(-max_force / 2.0)


def reward_lateral_drift_penalty(env, entity_manager, vel_cmd_manager, max_val=0.3, **kwargs) -> torch.Tensor:
    """
    Penalización por drift lateral (velocidad Y no comandada).
    Solo penaliza si el comando lateral es ~0.
    """
    lin_vel = entity_manager.get_linear_velocity()  # (N, 3)
    cmd = vel_cmd_manager.command  # (N, 3) → [vx, vy, wz]
    vy_error = (lin_vel[:, 1] - cmd[:, 1]) ** 2
    return torch.clamp(vy_error / (max_val ** 2), 0.0, 1.0)


def reward_imu_smoothness(env, entity_manager, max_val=10.0, **kwargs) -> torch.Tensor:
    """
    Premia la suavidad del movimiento medida por el IMU.
    Penaliza la norma total de la velocidad angular (incluido yaw).
    Un robot caminando bien tiene ω pequeño y controlado.
    """
    ang_vel = entity_manager.get_angular_velocity()  # (N, 3)
    omega_norm = torch.norm(ang_vel, dim=-1)
    return torch.clamp(omega_norm / max_val, 0.0, 1.0)


def reward_base_height(env, entity_manager, target_height: float = 0.234, max_val: float = 0.1, **kwargs) -> torch.Tensor:
    """
    Penalización por alejarse de la altura objetivo del cuerpo.
    Evita que el robot aprenda a agacharse o arrastrarse.
    target_height = HEIGHT_OFFSET = 0.234 m
    Resultado en [0, 1].
    """
    height = entity_manager.base_pos[:, 2]  # (N,)
    error  = (height - target_height) ** 2
    return torch.clamp(error / (max_val ** 2), 0.0, 1.0)


def reward_arm_symmetry(env, target: float = 0.943, max_val: float = 0.3, **kwargs) -> torch.Tensor:
    """
    Penaliza alejarse de la postura de brazos durante la caminata.
    Target: BD = +0.943 rad, BI = -0.943 rad (antisimetrico).
    BD no puede usarse en default_pos del simulador sin explotar,
    por eso se maneja aqui independientemente.
    Resultado en [0, 1].
    """
    dof_pos   = env.action_manager.get_dofs_position()  # (N, num_dof)
    dof_names = env.actuator_manager.dofs_names

    bd_idx = dof_names.index("BD")
    bi_idx = dof_names.index("BI")

    bd = dof_pos[:, bd_idx]   # (N,)
    bi = dof_pos[:, bi_idx]   # (N,)

    error_bd = (bd - target) ** 2   # quiere BD = +target
    error_bi = (bi + target) ** 2   # quiere BI = -target
    error = (error_bd + error_bi) / 2.0
    return torch.clamp(error / (max_val ** 2), 0.0, 1.0)


def reward_dof_pos_deviation(env, max_val: float = 0.5, **kwargs) -> torch.Tensor:
    """
    Penalización por alejarse de la posición articular por defecto.
    Excluye BD y BI — tienen su propia recompensa arm_symmetry con
    targets distintos al default_pos del simulador.
    Resultado en [0, 1].
    """
    dof_pos   = env.action_manager.get_dofs_position()  # (N, num_dof)
    default   = env.action_manager.default_dofs_pos     # (N, num_dof)
    dof_names = env.actuator_manager.dofs_names

    exclude = {"BD", "BI"}
    mask = torch.tensor(
        [name not in exclude for name in dof_names],
        dtype=torch.bool, device=dof_pos.device,
    )
    deviation = torch.sum(((dof_pos - default) ** 2)[:, mask], dim=-1)
    num_dof   = mask.sum().item()
    return torch.clamp(deviation / (num_dof * max_val ** 2), 0.0, 1.0)


class BipedGaitTrainingEnv(ManagedEnvironment):
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

        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.5, 0.0, 1.5),
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

        self.terrain = self.scene.add_entity(gs.morphs.Plane()) #type: ignore

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="model/Bipedo.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
            ) #type: ignore

        self.camera = self.scene.add_camera(
            pos=(2.5, 0.0, 1.5),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
            GUI=self._gamepad_control,
        )

    def config(self):
        self._critic_foot_idx  = [self.robot.get_link("ft_d").idx_local,
                                   self.robot.get_link("ft_i").idx_local]
        self._critic_ankle_idx = [self.robot.get_link("ank_d").idx_local,
                                   self.robot.get_link("ank_i").idx_local]
        self._critic_knee_idx  = [self.robot.get_link("kne_d").idx_local,
                                   self.robot.get_link("kne_i").idx_local]
        
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.position, #type: ignore
                    "params": {
                        "position":      INITIAL_BODY_POSITION,
                        "quat":          INITIAL_QUAT,
                        "zero_velocity": True,
                    },
                },
            },
        )

        self.actuator_manager = ActuatorManager(
            self,
            joint_names=["BD", "HD", "HI", "BI", "HEAD",
                         "CD", "LD", "KD", "FD",
                         "CI", "LI", "KI", "FI"],
            kp=NoisyValue(20.0, 0.15),
            kv=NoisyValue(0.4, 0.15),
            damping=NoisyValue(0.5, 0.25),
            frictionloss=NoisyValue(0.15, 0.1),
            default_pos={
                # Posición base (simétrica)
                "BD":   NoisyValue(0.0,  0.01),
                "BI":   NoisyValue(-0.943,  0.01),
                
                # Pierna derecha
                "CD":   NoisyValue(0.0,   0.01),
                "LD":   NoisyValue(-0.22, 0.01),
                "KD":   NoisyValue(-0.236, 0.01),
                "FD":   NoisyValue(0.0157, 0.01),
                
                # Pierna izquierda
                "CI":   NoisyValue(0.0,   0.01),
                "LI":   NoisyValue(0.22,  0.01),
                "KI":   NoisyValue(0.236, 0.01),
                "FI":   NoisyValue(-0.0157, 0.01),
                
                # Cabeza
                "HEAD": NoisyValue(0.0, 0.01),
            },
            max_force={
                "BD": 2.0, "HD": 2.0, "HI": 2.0, "BI": 2.0,
                "HEAD": 1.0,
                "CD": 3.0, "LD": 3.0, "KD": 3.0, "FD": 2.5,
                "CI": 3.0, "LI": 3.0, "KI": 3.0, "FI": 2.5,
            },
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.4,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        self.torso_contact_manager = ContactManager(
            self,
            link_names=["cbh_d_2"],
        )

        self.feet_contact_manager = ContactManager(
            self,
            link_names=["ft_d", "ft_i"],
            track_air_time=True,
            air_time_contact_threshold=1.0,
        )
        
        self.knee_contact_manager = ContactManager(
            self,
            link_names=["kne_d", "kne_i", "ank_d", "ank_i"],
        )
        
        self.leg_contact_manager = ContactManager(
            self,
            link_names=["leg_d", "leg_i"],
        )

        self.velocity_command = VelocityCommandManager(
            self,
            range={ # type: ignore
                "lin_vel_x": [-0.7, 0.7],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_z": [-0.5, 0.5],
            },
            standing_probability=0.15,
            resample_time_sec=3.0,
            debug_visualizer=True,
            debug_visualizer_cfg={"envs_idx": [0], "arrow_offset": 0.12}, #type: ignore
        )

        self.gait_command_manager = BipedGaitCommandManager(
            self,
            foot_names={
                "L": "ft_i",   
                "R": "ft_d",   
            },
            resample_time_sec=4.0,
        )

        # ══════════════════════════════════════════════════════════════════════
        #  REWARD MANAGER — Matemáticamente balanceado
        #
        #  PRESUPUESTO:
        #    Positivos  → hasta +6.3  (alive 1.0 + upright 1.5 + tracking 1.5
        #                              + gait 1.0 + foot_h 0.5 + air_time 0.8)
        #    Negativos  → hasta -3.5  (ang_vel 0.7 + vz 0.3 + acc 0.3
        #                              + action_rate 0.3 + energy 0.5
        #                              + bad_contact 0.8 + drift 0.3
        #                              + smoothness 0.3)
        #
        #  Ratio positivo/negativo ≈ 1.8:1 (robot de pie)
        #  Todos los términos están en [0, 1] antes de ponderar.
        # ══════════════════════════════════════════════════════════════════════
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={ # type: ignore
                # ── POSITIVOS ─────────────────────────────────────────────
                "alive_bonus": {
                    "weight": 1.0,
                    "fn": reward_alive_bonus,
                },
                "upright_stability": {
                    "weight": 1.5,
                    "fn": reward_upright_stability,
                    "params": {"entity_manager": self.robot_manager},
                },
                "tracking_lin_vel": {
                    "weight": 1.5,
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
                "gait_phase_reward": {
                    "weight": 1.0,
                    "fn": self.gait_command_manager.gait_phase_reward,
                    "params": {"contact_manager": self.feet_contact_manager},
                },
                "foot_height_reward": {
                    "weight": 0.5,
                    "fn": self.gait_command_manager.foot_height_reward,
                },
                "feet_air_time": {
                    "weight": 0.8,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "time_threshold":     0.2,
                        "time_threshold_max": 0.5,
                        "contact_manager":    self.feet_contact_manager,
                        "vel_cmd_manager":    self.velocity_command,
                    },
                },
                # ── NEGATIVOS (todos normalizados a [0,1]) ───────────────
                "ang_vel_xy": {
                    "weight": -0.7,
                    "fn": reward_angular_velocity_penalty,
                    "params": {"entity_manager": self.robot_manager,
                               "max_val": 5.0},
                },
                "lin_vel_z": {
                    "weight": -0.3,
                    "fn": reward_linear_vel_z_penalty,
                    "params": {"entity_manager": self.robot_manager,
                               "max_val": 1.0},
                },
                "body_acceleration": {
                    "weight": -0.3,
                    "fn": reward_body_acceleration_penalty,
                    "params": {"entity_manager": self.robot_manager,
                               "max_val": 20.0},
                },
                "action_rate": {
                    "weight": -0.3,
                    "fn": reward_action_rate_penalty,
                    "params": {"max_val": 2.0},
                },
                "energy": {
                    "weight": -0.5,
                    "fn": reward_energy_penalty,
                    "params": {"max_val": 1.5},
                },
                "bad_contact": {
                    "weight": -0.8,
                    "fn": reward_contact_binary_penalty,
                    "params": {"contact_manager": self.knee_contact_manager},
                },
                "base_height": {
                    "weight": -1.2,
                    "fn": reward_base_height,
                    "params": {
                        "entity_manager": self.robot_manager,
                        "target_height":  HEIGHT_OFFSET,
                        "max_val":        0.05,
                    },
                },
                "arm_symmetry": {
                    "weight": -1.0,
                    "fn": reward_arm_symmetry,
                    "params": {"max_val": 0.3},
                },
                "dof_pos_deviation": {
                    "weight": -1.5,
                    "fn": reward_dof_pos_deviation,
                    "params": {"max_val": 0.5},
                },
                "lateral_drift": {
                    "weight": -0.3,
                    "fn": reward_lateral_drift_penalty,
                    "params": {"entity_manager": self.robot_manager,
                               "vel_cmd_manager": self.velocity_command,
                               "max_val": 0.3},
                },
                "imu_smoothness": {
                    "weight": -0.3,
                    "fn": reward_imu_smoothness,
                    "params": {"entity_manager": self.robot_manager,
                               "max_val": 10.0},
                },
            },
        )

        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={ # type: ignore
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                "body_contact": {
                    "fn": terminations.contact_force,
                    "params": {
                        "contact_manager": self.torso_contact_manager,
                        "threshold": 1.0,
                        },
                },
                "fall_over": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle":    45.0,
                        "entity_manager": self.robot_manager,
                    },
                },
            },
        )

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
        
        ObservationManager(
            self,
            name="critic",
            history_len=5,
            cfg={ # type: ignore
                "linear_velocity": {
                    "fn": observations.entity_linear_velocity,
                    "params": {"entity_manager": self.robot_manager},
                },
                "angular_velocity": {
                    "fn": observations.entity_angular_velocity,
                    "params": {"entity_manager": self.robot_manager},
                },
                "projected_gravity": {
                    "fn": observations.entity_projected_gravity,
                    "params": {"entity_manager": self.robot_manager},
                },

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
                "feet_pos": {
                    "fn": lambda env: env.robot.get_links_pos(
                        links_idx_local=self._critic_foot_idx
                    ).reshape(env.num_envs, -1),
                },
                "feet_vel": {
                    "fn": lambda env: env.robot.get_links_vel(
                        links_idx_local=self._critic_foot_idx
                    ).reshape(env.num_envs, -1), #type: ignore
                },
                "ankle_pos": {
                    "fn": lambda env: env.robot.get_links_pos(
                        links_idx_local=self._critic_ankle_idx
                    ).reshape(env.num_envs, -1),
                },
                "knee_pos": {
                    "fn": lambda env: env.robot.get_links_pos(
                        links_idx_local=self._critic_knee_idx
                    ).reshape(env.num_envs, -1),
                },
                "knee_vel": {
                    "fn": lambda env: env.robot.get_links_vel(
                        links_idx_local=self._critic_knee_idx
                    ).reshape(env.num_envs, -1), #type: ignore
                },
                "foot_contact_force": {
                    "fn": observations.contact_force,
                    "params": {"contact_manager": self.feet_contact_manager},
                },
                "knee_contact_force": {
                    "fn": observations.contact_force,
                    "params": {"contact_manager": self.knee_contact_manager},
                },
                "gait_clock": {
                    "fn": lambda env: self.gait_command_manager.clock_input,
                },
                "gait_phase_raw": {
                    "fn": lambda env: self.gait_command_manager.gait_phase,
                },

                "robot_pos": {
                    "fn": lambda env: self.robot_manager.base_pos,
                },
                "robot_quat": {
                    "fn": lambda env: self.robot_manager.base_quat,
                },
                "current_actions": {
                    "fn": observations.current_actions,
                    "params": {"action_manager": self.action_manager},
                },
            },
        )

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

    def update_curriculum(self):

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