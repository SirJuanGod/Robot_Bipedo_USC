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
from genesis_forge.mdp import reset, rewards, terminations
import torch


INITIAL_BODY_POSITION = [0.0, 0.0, 0.228501]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]


class BipedEnv(ManagedEnvironment):
    """
    Example training environment for the Berkeley Humanoid robot.
    """

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

        # Construct the scene
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
        
        self.obs_history_len = 10  # 10 pasos de historia
        self.obs_history = None
        

        # Create terrain
        self.terrain = self.scene.add_entity(gs.morphs.Plane())

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="./model/Bipedo.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Camera, for headless video recording
        self.camera = self.scene.add_camera(
            pos=(2.5, 0.0, 2.5),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot)

    def config(self):

        """
        Configure the environment managers
        """
        ##
        # Robot manager
        # i.e. what to do with the robot when it is reset
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.position,
                    "params": {
                        "position": INITIAL_BODY_POSITION,
                        "quat": INITIAL_QUAT,
                        "zero_velocity": True,
                    },
                },
            },
        )

        ##
        # Joint Actions & actuator configuration
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[".*"],
            kp=15.0,
            kv=1.0,
            default_pos={
                ".*": 0.0,
            },
            max_force={
                ".*": 10.0,
            },
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.5,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        ##
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [0.0, 1.0],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_z": [-0.5, 0.5],
            },
            standing_probability=0.02,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
                "arrow_offset": 0.12,
            },
        )

        ##
        # Contact managers
        self.torso_contact_manager = ContactManager(
            self,
            link_names=["cadera"],
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # === Tracking de comandos (lo más importante) ===
                "tracking_lin_vel": {
                    "weight": 2.0,          # Aumentado — prioridad máxima
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager": self.robot_manager,
                    },
                },
                "tracking_ang_vel": {
                    "weight": 1.0,          # Aumentado
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager": self.robot_manager,
                    },
                },

                # === Estabilidad del torso ===
                "lin_vel_z": {
                    "weight": -2.0,         # Penaliza bote vertical
                    "fn": rewards.lin_vel_z_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "ang_vel_xy_l2": {
                    "weight": -0.1,         # Aumentado — penaliza más el bamboleo
                    "fn": rewards.ang_vel_xy_l2,
                    "params": {"entity_manager": self.robot_manager},
                },

                # === Suavidad de movimiento ===
                "action_rate": {
                    "weight": -0.01,        # Penaliza cambios bruscos de acción
                    "fn": rewards.action_rate_l2,
                },
                "similar_to_default": {
                    "weight": -0.1,         # Aumentado — mantener postura base
                    "fn": rewards.dof_similar_to_default,
                    "params": {"action_manager": self.action_manager},
                },
            },
        )

        ##
        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                # The episode ended
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                # Terminate if the robot's pitch and yaw angles are too large
                "torso_contact": {
                    "fn": terminations.contact_force,
                    "params": {
                        "contact_manager": self.torso_contact_manager,
                    },
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            cfg={
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity()
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
        
        
    def _update_obs_history(self, obs):
        """Mantiene un buffer circular de observaciones pasadas."""
        import torch
        if self.obs_history is None:
            # Inicializa con ceros
            self.obs_history = torch.zeros(
                self.num_envs,
                self.obs_history_len * obs.shape[-1],
                device=obs.device
            )
        # Desplaza el historial y añade la nueva observación
        obs_dim = obs.shape[-1]
        self.obs_history = torch.roll(self.obs_history, shifts=-obs_dim, dims=-1)
        self.obs_history[:, -obs_dim:] = obs
        return self.obs_history