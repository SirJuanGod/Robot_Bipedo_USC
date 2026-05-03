import genesis as gs
import torch
from collections import deque

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
from genesis_forge.mdp import reset, rewards, terminations
from genesis_forge.genesis_env import GenesisEnv


INITIAL_BODY_POSITION = [0.0, 0.0, 0.228501]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

MAX_LIN_VEL = 0.5


def similar_to_default_scaled(env: GenesisEnv) -> torch.Tensor:
    vx = env.velocity_command.get_command("lin_vel_x") # type: ignore
    vy = env.velocity_command.get_command("lin_vel_y") # type: ignore
    lin_vel_magnitude = torch.sqrt(vx**2 + vy**2)
    scale = torch.clamp(1.0 - lin_vel_magnitude / MAX_LIN_VEL, min=0.1, max=1.0)
    base_penalty = rewards.dof_similar_to_default(env, action_manager=env.action_manager) # type: ignore
    return base_penalty * scale


class BipedEnv(ManagedEnvironment):

    CURRICULUM = {
        1: {
            "tracking_lin_vel":   0.5,
            "tracking_ang_vel":   0.3,
            "lin_vel_z":         -2.0,
            "ang_vel_xy_l2":     -0.3,
            "action_rate":       -0.02,
            "similar_to_default": 0.0,
            "body_acceleration":  0.0,
            "stand_still":        0.0,
            "feet_air_time":      0.0,
        },
        2: {
            "tracking_lin_vel":   1.5,
            "tracking_ang_vel":   0.7,
            "lin_vel_z":         -3.0,
            "ang_vel_xy_l2":     -0.5,
            "action_rate":       -0.05,
            "similar_to_default":-0.2,
            "body_acceleration":  0.0,
            "stand_still":        0.0,
            "feet_air_time":      0.8,
        },
        3: {
            "tracking_lin_vel":   2.0,
            "tracking_ang_vel":   1.0,
            "lin_vel_z":         -5.0,
            "ang_vel_xy_l2":     -0.8,
            "action_rate":       -0.08,
            "similar_to_default":-0.2,
            "body_acceleration": -0.3,
            "stand_still":       -0.2,
            "feet_air_time":      1.0,
        },
        4: {
            "tracking_lin_vel":   2.0,
            "tracking_ang_vel":   1.0,
            "lin_vel_z":         -5.0,
            "ang_vel_xy_l2":     -0.8,
            "action_rate":       -0.1,
            "similar_to_default":-0.2,
            "body_acceleration": -0.5,
            "stand_still":       -0.3,
            "feet_air_time":      1.0,
        },
    }

    PHASE_THRESHOLDS = {
        1: {"duration_s": 12.0, "tracking": 0.3},
        2: {"duration_s": 15.0, "tracking": 0.5},
        3: {"duration_s": 18.0, "tracking": 0.7},
    }

    EVAL_WINDOW = 100

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 20,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )

        self.curriculum_phase = 1
        self._ep_durations: deque[float] = deque(maxlen=self.EVAL_WINDOW)
        self._ep_tracking: deque[float]  = deque(maxlen=self.EVAL_WINDOW)

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

        self.terrain = self.scene.add_entity(gs.morphs.Plane()) # type: ignore

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="./model/Bipedo.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ), # type: ignore
        )

        self.camera = self.scene.add_camera(
            pos=(0.8, 0.0, 0.5),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot)

    def config(self):

        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.position,  # type: ignore   
                    "params": {
                        "position": INITIAL_BODY_POSITION,
                        "quat": INITIAL_QUAT,
                        "zero_velocity": True,
                    },
                },
            },
        )

        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[".*"],
            kp={
                "(CD|CI)": 30.0,   # cadera
                "(LD|LI)": 28.0,   # muslo
                "(KD|KI)": 22.0,   # rodilla
                "(FD|FI)": 20.0,   # pie
                "(HD|HI)": 8.0,    # hombro
                "(BD|BI)": 8.0,    # codo
                "HEAD":    5.0,    # cabeza
            },
            kv={
                "(CD|CI)": 0.8,
                "(LD|LI)": 0.8,
                "(KD|KI)": 0.6,
                "(FD|FI)": 0.6,
                "(HD|HI)": 3.0,
                "(BD|BI)": 3.0,
                "HEAD":    3.0,
            },
            default_pos={
                "(BD)":  0.7854,   # codo derecho ligeramente doblado
                "(BI)": -0.7854,   # codo izquierdo ligeramente doblado
                ".*":    0.0,
            },
            max_force={".*": 1.0},
        )

        self.action_manager = PositionActionManager(
            self,
            scale=0.5,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (0.0, 0.5),
                "lin_vel_y": (0.0, 0.0),  # Lateral movement disabled intentionally
                "ang_vel_z": (-0.3, 0.3),
            },
            standing_probability=0.02,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={  # type: ignore
                "envs_idx": [0],
                "arrow_offset": 0.12,
            },
        )

        # Torso: link raíz del robot
        self.torso_contact_manager = ContactManager(
            self,
            link_names=["cbh_d"],
        )

        # Pies: links terminales de cada pierna
        self.feet_contact_manager = ContactManager(
            self,
            link_names=["ft_d", "ft_i"],
            track_air_time=True,
        )

        p = self.CURRICULUM[1]
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            
            cfg={ # type: ignore
                "tracking_lin_vel": {
                    "weight": p["tracking_lin_vel"],
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager": self.robot_manager,
                    },
                },
                "tracking_ang_vel": {
                    "weight": p["tracking_ang_vel"],
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager": self.robot_manager,
                    },
                },
                "lin_vel_z": {
                    "weight": p["lin_vel_z"],
                    "fn": rewards.lin_vel_z_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "ang_vel_xy_l2": {
                    "weight": p["ang_vel_xy_l2"],
                    "fn": rewards.ang_vel_xy_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "action_rate": {
                    "weight": p["action_rate"],
                    "fn": rewards.action_rate_l2,
                },
                "similar_to_default": {
                    "weight": p["similar_to_default"],
                    "fn": similar_to_default_scaled,
                },
                "body_acceleration": {
                    "weight": p["body_acceleration"],
                    "fn": rewards.body_acceleration_exp,
                    "params": {
                        "entity_manager": self.robot_manager,
                        "sensitivity": 0.1,
                    },
                },
                "stand_still": {
                    "weight": p["stand_still"],
                    "fn": rewards.stand_still_joint_deviation_l1,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "action_manager": self.action_manager,
                        "command_threshold": 0.06,
                    },
                },
                "feet_air_time": {
                    "weight": p["feet_air_time"],
                    "fn": rewards.feet_air_time,
                    "params": {
                        "contact_manager": self.feet_contact_manager,
                        "time_threshold": 0.2,
                        "time_threshold_max": 0.5,
                        "vel_cmd_manager": self.velocity_command,
                    },
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
                "torso_contact": {
                    "fn": terminations.contact_force,
                    "params": {
                        "contact_manager": self.torso_contact_manager,
                        "threshold": 1.0,
                    },
                },
                "torso_height": {
                    "fn": terminations.base_height_below_minimum,
                    "params": {
                        "entity_manager": self.robot_manager,
                        "minimum_height": 0.19,
                    },
                },
                "bad_orientation": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "entity_manager": self.robot_manager,
                        "limit_angle": 30.0,
                        "grace_steps": 5,
                    },
                },
            },
        )

        ObservationManager(
            self,
            cfg={
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity() # type: ignore
                                    + 0.05 * torch.randn_like(
                                        self.robot_manager.get_angular_velocity()
                                    ),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity() 
                                    + 0.05 * torch.randn_like(
                                        self.robot_manager.get_linear_velocity()
                                    ),
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity()
                                    + 0.02 * torch.randn_like(
                                        self.robot_manager.get_projected_gravity()
                                    ),
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )

    def _apply_phase_weights(self, phase: int):
        p = self.CURRICULUM[phase]
        for name, weight in p.items():
            self.reward_manager.cfg[name].weight = weight

    def _check_curriculum_advance(self):
        if self.curriculum_phase >= 4:
            return
        if len(self._ep_durations) < self.EVAL_WINDOW:
            return

        threshold = self.PHASE_THRESHOLDS[self.curriculum_phase]
        mean_duration = sum(self._ep_durations) / len(self._ep_durations)
        mean_tracking = sum(self._ep_tracking) / len(self._ep_tracking)

        if mean_duration >= threshold["duration_s"] and mean_tracking >= threshold["tracking"]:
            self.curriculum_phase += 1
            self._apply_phase_weights(self.curriculum_phase)
            print(
                f"\n{'='*60}\n"
                f"  CURRICULUM: Avanzando a Fase {self.curriculum_phase}\n"
                f"  Duración media:  {mean_duration:.1f}s  (umbral: {threshold['duration_s']}s)\n"
                f"  Tracking medio:  {mean_tracking:.3f}   (umbral: {threshold['tracking']})\n"
                f"{'='*60}\n"
            )

    def on_episode_end(self, episode_duration_s: float):
        self._ep_durations.append(episode_duration_s)
        tracking = self.reward_manager.last_episode_mean_reward(
            "tracking_lin_vel", before_weight=True
        )
        self._ep_tracking.append(tracking)
        self._check_curriculum_advance()

    def reset(self, envs_idx=None):
        if envs_idx is not None and len(envs_idx) > 0:
            durations = self.episode_length[envs_idx].float() * self.dt
            self.on_episode_end(durations.mean().item())
        return super().reset(envs_idx)