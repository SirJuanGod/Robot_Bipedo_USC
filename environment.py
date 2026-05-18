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


def _noisy(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor + torch.randn_like(tensor) * scale


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
                "BD":   NoisyValue(0.943, 0.01),
                "HD":   NoisyValue(0.0, 0.01),
                "HI":   NoisyValue(0.0, 0.01),
                "BI":   NoisyValue(-0.943, 0.01),
                "HEAD": NoisyValue(0.0, 0.01),
                "CD":  NoisyValue(0.0, 0.01),
                "LD": NoisyValue(-0.22, 0.01),
                "KD": NoisyValue(-0.236, 0.01),
                "FD":  NoisyValue(-0.0157, 0.01),
                "CI":  NoisyValue(0.22, 0.01),
                "LI":  NoisyValue(0.157, 0.01),
                "KI":  NoisyValue(0.0, 0.01),
                "FI":  NoisyValue(0.0157, 0.01),
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
                    "weight": 1.5,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "time_threshold":     0.2,
                        "time_threshold_max": 0.5,
                        "contact_manager":    self.feet_contact_manager,
                        "vel_cmd_manager":    self.velocity_command,
                    },
                },
                "lin_vel_z": {
                    "weight": -0.3,
                    "fn": rewards.lin_vel_z_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "ang_vel_xy_l2": {
                    "weight": -0.5,
                    "fn": rewards.ang_vel_xy_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "body_acceleration": {
                    "weight": -0.5,
                    "fn": rewards.body_acceleration_exp,
                    "params": {"entity_manager": self.robot_manager},
                },
                "base_height_target": {
                    "weight": -2.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": HEIGHT_OFFSET - 0.05,
                        "entity_attr":   "robot",
                    },
                },
                "action_rate": {
                    "weight": -0.025,
                    "fn": rewards.action_rate_l2,
                },
                "similar_to_default": {
                    "weight": -0.01,
                    "fn": rewards.dof_similar_to_default,
                    "params": {"action_manager": self.action_manager},
                },
                "bad_contact": {
                    "weight": -1.0,
                    "fn": rewards.contact_force,
                    "params": {"contact_manager": self.knee_contact_manager},
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