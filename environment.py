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

HEIGHT_OFFSET                = 0.2340
INITIAL_BODY_POSITION        = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT                 = [1.0, 0.0, 0.0, 0.0]
CURRICULUM_CHECK_EVERY_STEPS = 50

<<<<<<< HEAD
# Valores de ruido sim2real para las observaciones del actor policy
=======
# --- NUEVA CONSTANTE: Dificultad del terreno ---
TERRAIN_VERTICAL_SCALE       = 0.015  # 1.5 cm de irregularidad (súbelo gradualmente)

>>>>>>> Transfer-Learning-branch
NOISE_SCALES = {
    "ang_vel": 0.05,
    "gravity":  0.05,
    "dof_pos": 0.01,
    "dof_vel": 0.50,
}

ROBOT_MASS_KG  = 1.09
ROBOT_HEIGHT_M = 0.234
LEG_LENGTH_M   = 0.084
GRAVITY        = 9.81


def _noisy(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor + torch.randn_like(tensor) * scale


def reward_tracking_lin_vel(env, entity_manager, vel_cmd_manager, sigma: float = 0.25, **kwargs) -> torch.Tensor:
    """Premia seguir la velocidad lineal deseada en X (frente) e Y (lateral)."""
    lin_vel = entity_manager.get_linear_velocity()[:, :2]
    cmd     = vel_cmd_manager.command[:, :2]
    error   = torch.sum((lin_vel - cmd) ** 2, dim=-1)
    return torch.exp(-error / sigma)

def reward_tracking_ang_vel(env, entity_manager, vel_cmd_manager, sigma: float = 0.25, **kwargs) -> torch.Tensor:
    """Premia seguir la velocidad de giro (Yaw)."""
    ang_vel_z = entity_manager.get_angular_velocity()[:, 2]
    cmd_wz    = vel_cmd_manager.command[:, 2]
    error     = (ang_vel_z - cmd_wz) ** 2
    return torch.exp(-error / sigma)

def reward_alive_bonus(env, **kwargs) -> torch.Tensor:
    """Incentiva a no caerse sumando puntos por cada step vivo."""
    return torch.ones(env.num_envs, device=gs.device)

def reward_upright_stability(env, entity_manager, **kwargs) -> torch.Tensor:
    """Mantiene el torso alineado con la gravedad (evita que se incline demasiado)."""
    proj_grav = entity_manager.get_projected_gravity()
    gz = proj_grav[:, 2]
    return torch.clamp(-gz, 0.0, 1.0)

def reward_base_height(env, entity_manager, target_height: float = 0.234, max_val: float = 0.05, **kwargs) -> torch.Tensor:
    """Crucial para bípedos: evita que el robot camine en cuclillas para no caerse."""
    # MathCodeOptimizer: se precomputa el escalar inv_max2 fuera de la división
    # para evitar recalcularlo por entorno; la operación queda como un fma vectorizado.
    inv_max2 = 1.0 / (max_val ** 2)
    height   = entity_manager.base_pos[:, 2]
    error    = (height - target_height) ** 2
    return torch.clamp(error * inv_max2, 0.0, 1.0)

def reward_foot_alternation(env, contact_manager, vel_cmd_manager, force_scale: float = 5.0, **kwargs) -> torch.Tensor:
    """
    (Muy Importante) Fuerza al robot a alternar los pies (izquierdo/derecho) al moverse.
    Evita que el bípedo aprenda a saltar con los dos pies a la vez (bunny hopping).
    """
    ids = contact_manager._link_ids
    if ids is None or len(ids) < 2:
        return torch.zeros(env.num_envs, device=gs.device)

    # MathCodeOptimizer: .tolist() es NECESARIO — get_contact_forces(link_idx: int | list[int])
    # hace isinstance(link_idx, list); un tensor no pasa esa rama y devuelve idx=[]
    inv_scale  = 1.0 / force_scale
    forces     = contact_manager.get_contact_forces(ids.tolist())  # (N, 2, 3)
    force_norm = torch.linalg.norm(forces, dim=-1)                 # (N, 2) — más explícito que torch.norm

    p = torch.sigmoid(force_norm * inv_scale - 1.0)               # (N, 2) ∈ (0,1)
    p_L, p_R = p[:, 0], p[:, 1]

    alternation = p_L * (1.0 - p_R) + p_R * (1.0 - p_L)          # (N,) ∈ [0,1]

    cmd_norm = torch.linalg.norm(vel_cmd_manager.command[:, :2], dim=-1)
    moving   = (cmd_norm > 0.05).float()

    return alternation * moving

def reward_action_rate_penalty(env, max_val: float = 2.0, **kwargs) -> torch.Tensor:
    """Evita vibraciones y protege los motores (limita el cambio brusco de comandos)."""
    curr = env.action_manager.get_actions()
    if not hasattr(env, '_prev_actions_reward') or env._prev_actions_reward is None:
        env._prev_actions_reward = curr.detach()  # MathCodeOptimizer: detach en lugar de clone; evita retener el grafo
        return torch.zeros(env.num_envs, device=gs.device)
    prev    = env._prev_actions_reward
    # MathCodeOptimizer: se precomputa el escalar normalizador para que la clamp use una mul en vez de dos divs
    inv_norm = 1.0 / (curr.shape[-1] * max_val ** 2)
    diff_sq  = torch.sum((curr - prev) ** 2, dim=-1)
    env._prev_actions_reward = curr.detach()      # MathCodeOptimizer: detach suficiente; clone es innecesario aquí
    return torch.clamp(diff_sq * inv_norm, 0.0, 1.0)

def reward_dof_pos_deviation(env, max_val: float = 0.5, **kwargs) -> torch.Tensor:
    """Mantiene los motores cerca de su posición inicial, resultando en una marcha simétrica."""
    dof_pos   = env.action_manager.get_dofs_position()
    default   = env.action_manager.default_dofs_pos
    deviation = torch.sum((dof_pos - default) ** 2, dim=-1)
    num_dof   = dof_pos.shape[-1]
    return torch.clamp(deviation / (num_dof * max_val ** 2), 0.0, 1.0)

def reward_base_motion_penalty(env, entity_manager, max_ang_vel: float = 5.0, max_lin_z: float = 1.0, **kwargs) -> torch.Tensor:
    """
    (Combinada) Penaliza que el torso rebote hacia arriba/abajo (lin_z) 
    y que se balancee bruscamente (ang_vel X e Y).
    Sustituye a linear_vel_z_penalty y angular_velocity_penalty.
    """
    ang_vel = entity_manager.get_angular_velocity()
    lin_vel = entity_manager.get_linear_velocity()

    # MathCodeOptimizer: torch.sum sobre la slice [:, :2] reemplaza la suma manual de escalares;
    # evita dos operaciones ** separadas y mantiene un solo kernel de reducción vectorizado.
    inv_ang2 = 1.0 / (max_ang_vel ** 2)
    inv_z2   = 1.0 / (max_lin_z ** 2)
    ang_vel_xy_sq = torch.sum(ang_vel[:, :2] ** 2, dim=-1)        # (N,) vectorizado
    vz_sq         = lin_vel[:, 2] ** 2                            # (N,)

    ang_penalty = torch.clamp(ang_vel_xy_sq * inv_ang2, 0.0, 1.0)
    z_penalty   = torch.clamp(vz_sq         * inv_z2,  0.0, 1.0)

    # Se suman y promedian para retornar un solo tensor escalar por entorno
    return (ang_penalty + z_penalty) * 0.5  # MathCodeOptimizer: * 0.5 en vez de / 2.0 (una mul vs una div)

def reward_contact_binary_penalty(env, contact_manager, **kwargs) -> torch.Tensor:
    """Penaliza fuertemente cualquier contacto de los links especificados (ej. rodillas)"""
    ids = contact_manager._link_ids
    if ids is None or len(ids) == 0:
        return torch.zeros(env.num_envs, device=gs.device)
    # MathCodeOptimizer: .tolist() es NECESARIO — get_contact_forces hace isinstance(link_idx, list);
    # pasar el tensor directamente devolvería idx=[] en silencio.
    # torch.linalg.norm y amax sí son optimizaciones válidas conservadas.
    forces     = contact_manager.get_contact_forces(ids.tolist())
    force_norm = torch.linalg.norm(forces, dim=-1)                 # (N, K)
    max_force  = force_norm.amax(dim=-1)                           # (N,) — amax evita la tupla de .max()
    return 1.0 - torch.exp(-max_force * 0.5)                      # MathCodeOptimizer: * 0.5 en vez de / 2.0


def reward_angular_velocity_penalty(env, entity_manager, max_val: float = 5.0, **kwargs) -> torch.Tensor:
    """Evita que el torso se balancee bruscamente hacia los lados o adelante/atrás"""
    # MathCodeOptimizer: torch.sum sobre slice reemplaza suma manual de dos escalares al cuadrado
    ang_vel       = entity_manager.get_angular_velocity()
    ang_vel_xy_sq = torch.sum(ang_vel[:, :2] ** 2, dim=-1)        # (N,) vectorizado
    return torch.clamp(ang_vel_xy_sq * (1.0 / (max_val ** 2)), 0.0, 1.0)


def reward_linear_vel_z_penalty(env, entity_manager, max_val: float = 1.0, **kwargs) -> torch.Tensor:
    """Evita que el robot salte o rebote verticalmente mientras camina"""
    # MathCodeOptimizer: escalar precomputado evita la división por entorno
    lin_vel = entity_manager.get_linear_velocity()
    vz_sq   = lin_vel[:, 2] ** 2
    return torch.clamp(vz_sq * (1.0 / (max_val ** 2)), 0.0, 1.0)

def reward_energy_penalty(env, max_val: float = 1.5, **kwargs) -> torch.Tensor:
    """Penaliza el uso excesivo de los motores (acciones grandes) para ahorrar energía y proteger el hardware."""
    actions  = env.action_manager.get_actions()
    # MathCodeOptimizer: inv_norm es un escalar Python puro; evita que PyTorch cree un tensor temporal
    # solo para la división. La operación resultante es un fma (fused multiply-add) vectorizado.
    inv_norm = 1.0 / (actions.shape[-1] * max_val ** 2)
    act_sq   = torch.sum(actions ** 2, dim=-1)
    return torch.clamp(act_sq * inv_norm, 0.0, 1.0)


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

        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                n_subterrains=(1, 1),
                subterrain_size=(8.0, 8.0),
                vertical_scale=0.01, # Escala base (la controlamos más abajo)
                subterrain_types=[["random_uniform_terrain"]],
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": -0.02,  # -2 cm de profundidad
                        "max_height": 0.02,   # +2 cm de altura máxima
                        "step": 0.01,         # Incrementos suaves
                        "downsampled_scale": 0.2, # Suavidad entre picos
                    },
                },
            )
        ) #type: ignore

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
        self._critic_foot_idx  = [self.robot.get_link("leg_p1_d").idx_local,
                                   self.robot.get_link("leg_p1_i").idx_local]
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
                "BD":   NoisyValue(0.943, 0.01),
                "HD":   NoisyValue(0.0, 0.01),
                "HI":   NoisyValue(0.0, 0.01),
                "BI":   NoisyValue(-0.943, 0.01),
                "HEAD": NoisyValue(0.0, 0.01),
                "CD":   NoisyValue(0.0, 0.01),
                "LD":   NoisyValue(-0.22, 0.01),
                "KD":   NoisyValue(-0.236, 0.01),
                "FD":   NoisyValue(-0.0157, 0.01),
                "CI":   NoisyValue(0.22, 0.01),
                "LI":   NoisyValue(0.157, 0.01),
                "KI":   NoisyValue(0.0, 0.01),
                "FI":   NoisyValue(0.0157, 0.01),
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
            link_names=["cadera"],
        )

        self.feet_contact_manager = ContactManager(
            self,
            link_names=["leg_p1_d", "leg_p1_i"],
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
            range={ #type: ignore
                "lin_vel_x": [0.0, 0.7],
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
                "L": "leg_p1_i",
                "R": "leg_p1_d",
            },
            resample_time_sec=4.0,
        )

        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={ #type: ignore
                
                "alive_bonus": {
                    "weight": 1.0,
                    "fn": reward_alive_bonus,
                },
<<<<<<< HEAD
                "upright_stability": {
                    "weight": 1.0, # MODIFICADO: Relajado de 1.5 a 1.0
                    "fn": reward_upright_stability,
                    "params": {"entity_manager": self.robot_manager},
                },
=======
>>>>>>> Transfer-Learning-branch
                "tracking_lin_vel": {
                    "weight": 1.5,
                    "fn": reward_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager":  self.robot_manager,
                        "sigma":           0.25,
                    },
                },
                "tracking_ang_vel": {
                    "weight": 0.5,
                    "fn": reward_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager":  self.robot_manager,
                        "sigma":           0.25,
                    },
                },

                "gait_phase_reward": {
                    "weight": 1.0,
                    "fn": self.gait_command_manager.gait_phase_reward,
                    "params": {"contact_manager": self.feet_contact_manager},
                },
                "foot_height_reward": {
<<<<<<< HEAD
                    "weight": 1.2, # MODIFICADO: Aumentado de 0.5 a 1.2
=======
                    "weight": 1.2, 
>>>>>>> Transfer-Learning-branch
                    "fn": self.gait_command_manager.foot_height_reward,
                },

                "upright_stability": {
                    "weight": 1.0, 
                    "fn": reward_upright_stability,
                    "params": {"entity_manager": self.robot_manager},
                },
                "base_height": {
                    "weight": -0.2, 
                    "fn": reward_base_height,
                    "params": {
                        "entity_manager": self.robot_manager,
                        "target_height":  HEIGHT_OFFSET-0.02,
                        "max_val":        0.05,
                    },
                },
                "bad_contact_knee": {
                    "weight": -1.5, # Penalización severa excelente
                    "fn": reward_contact_binary_penalty,
                    "params": {"contact_manager": self.knee_contact_manager},
                },

                "ang_vel_xy": {
                    "weight": -0.7,
                    "fn": reward_angular_velocity_penalty,
                    "params": {"entity_manager": self.robot_manager, "max_val": 5.0},
                },
                "lin_vel_z": {
                    "weight": -0.3,
                    "fn": reward_linear_vel_z_penalty,
                    "params": {"entity_manager": self.robot_manager, "max_val": 1.0},
                },
                "action_rate": {
                    "weight": -0.5,  # RewardArchitect: subido de -0.3; controla torque oscilante
                    "fn": reward_action_rate_penalty,
                    "params": {"max_val": 2.0},
                },
<<<<<<< HEAD
                "energy": {
                    "weight": -0.4,
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
                        "target_height":  HEIGHT_OFFSET-0.02,
                        "max_val":        0.05,
                    },
                },
=======
>>>>>>> Transfer-Learning-branch
                "dof_pos_deviation": {
                    "weight": -0.4,  # RewardArchitect: bajado de -0.5; reduce duplicidad con energy
                    "fn": reward_dof_pos_deviation,
                    "params": {"max_val": 0.5},
                },

                "energy": {
                    "weight": -0.8,  # RewardArchitect: duplicado de -0.4; ratio torque/vel_tracking ≈ 0.53 (rango saludable 0.3–0.6)
                    "fn": reward_energy_penalty,
                    "params": {"max_val": 1.5},
                },

            },
        )

        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={ #type: ignore
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
            cfg={ #type: ignore
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
            cfg={ #type: ignore
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
        self._next_curriculum_check_step = self.step_count + CURRICULUM_CHECK_EVERY_STEPS

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