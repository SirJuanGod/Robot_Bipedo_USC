import torch
import genesis as gs
from typing import TypedDict, Literal
from genesis_forge.managers.command.command_manager import CommandManager
from genesis_forge.managers import ContactManager
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

# ──────────────────────────────────────────────
# Rangos globales de parámetros de marcha
# ──────────────────────────────────────────────
GAIT_PERIOD_RANGE    = [0.3, 0.6]
FOOT_CLEARANCE_RANGE = [0.04, 0.24] 

GaitName = Literal["walk", "run"]
FootName = Literal["L", "R"]

GAIT_OFFSETS: dict[GaitName, dict[FootName, float]] = {
    "walk": {"L": 0.0, "R": 0.5},
    "run":  {"L": 0.0, "R": 0.5},
}
GAIT_PERIOD_HINTS: dict[GaitName, list[float]] = {
    "walk": [0.6, 0.8],
    "run":  [0.4, 0.6],
}
GAIT_CLEARANCE_HINTS: dict[GaitName, list[float]] = {
    "walk": [0.05, 0.12],
    "run":  [0.10, 0.20],
}


class FootNames(TypedDict):
    L: str
    R: str 


class BipedGaitCommandManager(CommandManager):
    def __init__(
        self,
        env: GenesisEnv,
        foot_names: FootNames,
        resample_time_sec: float = 5.0,
        robot_entity_attr: str = "robot",
    ):
        super().__init__(env, range={}, resample_time_sec=resample_time_sec)

        self._robot_entity_attr = robot_entity_attr
        self._foot_names        = foot_names
        self.foot_links         = []          
        self._gamepad: Gamepad | None = None
        self._gamepad_btn_pressed = False
        self._gamepad_gait_idx    = 0

        self._num_gaits         = 1
        self._gait_period_range = [(GAIT_PERIOD_RANGE[0] + GAIT_PERIOD_RANGE[1]) / 2] * 2
        self._foot_clearance_range = [FOOT_CLEARANCE_RANGE[0]] * 2
        self._all_gaits_learned = False

        N = env.num_envs
        self.foot_offset    = torch.zeros((N, 2), device=gs.device)
        self.gait_period    = torch.zeros((N, 1), device=gs.device)
        self.foot_height    = torch.zeros((N, 1), device=gs.device)
        self.gait_time      = torch.zeros((N, 1), dtype=torch.float, device=gs.device)
        self.gait_phase     = torch.zeros((N, 1), dtype=torch.float, device=gs.device)
        self.clock_input    = torch.zeros((N, 4), dtype=torch.float, device=gs.device)
        self._gait_selected = torch.zeros(N, dtype=torch.long, device=gs.device)

    @property
    def command(self) -> torch.Tensor:
        if self._gamepad is not None:
            self._process_gamepad_input()
        return torch.cat([self.foot_offset, self.foot_height, self.gait_period], dim=-1)

    def increment_num_gaits(self):
        if self._all_gaits_learned:
            return
        if self._num_gaits >= len(GAIT_OFFSETS):
            self._all_gaits_learned = True
            print("🎯 Todas las marchas aprendidas — muestreo uniforme activado.")
        else:
            self._num_gaits = min(self._num_gaits + 1, len(GAIT_OFFSETS))
            new_gait = list(GAIT_OFFSETS.keys())[self._num_gaits - 1]
            print(f"📈 Currículo: desbloqueada marcha '{new_gait}' ({self._num_gaits}/{len(GAIT_OFFSETS)})")

    def increment_gait_period_range(self):
        
        self._gait_period_range[0] = max(
            self._gait_period_range[0] - 0.05, GAIT_PERIOD_RANGE[0]
        )
        self._gait_period_range[1] = min(
            self._gait_period_range[1] + 0.05, GAIT_PERIOD_RANGE[1]
        )

    def increment_foot_clearance_range(self):
        
        self._foot_clearance_range[0] = max(
            self._foot_clearance_range[0] - 0.01, FOOT_CLEARANCE_RANGE[0]
        )
        self._foot_clearance_range[1] = min(
            self._foot_clearance_range[1] + 0.01, FOOT_CLEARANCE_RANGE[1]
        )


    def build(self):
        super().build()
        robot: RigidEntity = getattr(self.env, self._robot_entity_attr) #type: ignore
        for i, key in enumerate(("L", "R")):
            foot_link_name = self._foot_names[key]
            self.foot_links.insert(i, robot.get_link(foot_link_name))

    def resample_command(self, env_ids: list[int]):
        if self._gamepad is not None:
            return
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=gs.device, dtype=torch.long) #type: ignore

        gait_names = list(GAIT_OFFSETS.keys())[: self._num_gaits]

        if self._num_gaits == 1:
            self._set_gait(gait_names[0], env_ids) #type: ignore
            self._gait_selected[env_ids] = 0
        else:
            gait_indices = self._generate_random_gait_indices(len(env_ids))
            for gait_idx in range(self._num_gaits):
                mask = gait_indices == gait_idx
                if mask.any():
                    selected = env_ids[mask]
                    self._set_gait(gait_names[gait_idx], selected) #type: ignore
                    self._gait_selected[selected] = gait_idx

    def step(self):
        super().step()
        self._log_metrics()

        self.gait_time  = (self.gait_time + self.env.dt) % self.gait_period
        self.gait_phase = self.gait_time / self.gait_period

        for i in range(2): 
            foot_phase = (self.gait_phase + self.foot_offset[:, i].unsqueeze(1)) % 1.0
            self.clock_input[:, i]     = torch.sin(2 * torch.pi * foot_phase).squeeze(-1)
            self.clock_input[:, i + 2] = torch.cos(2 * torch.pi * foot_phase).squeeze(-1)

    def reset(self, env_ids: list[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device) #type: ignore
        super().reset(env_ids)
        self.clock_input[env_ids] = 0.0
        self.gait_time[env_ids]   = 0.0
        self.gait_phase[env_ids]  = 0.0

    def observation(self, env: GenesisEnv) -> torch.Tensor:
        return torch.cat([self.command, self.clock_input], dim=-1)

    def use_gamepad(self, gamepad: Gamepad):
        self._gamepad = gamepad
        self._num_gaits = len(GAIT_OFFSETS)
        self._gamepad_gait_idx = 0
        self._gamepad_select_gait(list(GAIT_OFFSETS.keys())[0])

## Recompensas

    def foot_height_reward(self, env: GenesisEnv, sensitivity: float = 0.1) -> torch.Tensor:

        link_idx    = [f.idx_local for f in self.foot_links]
        foot_vel    = env.robot.get_links_vel(links_idx_local=link_idx)   # (N, 2, 3)   
        foot_pos    = env.robot.get_links_pos(links_idx_local=link_idx)   # (N, 2, 3)

        foot_vel_xy_norm = torch.norm(foot_vel[:, :, :2], dim=-1)              # (N, 2) #type: ignore
        height_err  = torch.square(foot_pos[:, :, 2] - self.foot_height)  # (N, 2)

        clearance_error = torch.sum(foot_vel_xy_norm * height_err, dim=-1)     # (N,)
        return torch.exp(-clearance_error / sensitivity)

    def gait_phase_reward(self, env: GenesisEnv, contact_manager: ContactManager) -> torch.Tensor:
        
        r_L = self._foot_phase_reward(0, contact_manager)   # ft_i
        r_R = self._foot_phase_reward(1, contact_manager)   # ft_d
        return torch.exp(r_L.flatten() + r_R.flatten())

    def _foot_phase_reward(self, foot_idx: int, contact_manager: ContactManager) -> torch.Tensor:
        
        link = self.foot_links[foot_idx]
        N    = self.env.num_envs

        force_weight = torch.zeros(N, 1, dtype=torch.float, device=gs.device)
        vel_weight   = torch.zeros(N, 1, dtype=torch.float, device=gs.device)

        force    = torch.norm(contact_manager.get_contact_forces(link.idx), dim=-1).view(-1, 1)
        velocity = torch.norm(link.get_vel(), dim=-1).view(-1, 1)

        phase  = (self.gait_phase + self.foot_offset[:, foot_idx].unsqueeze(1)) % 1.0
        swing  = (phase < 0.5).squeeze(-1)
        stance = (phase >= 0.5).squeeze(-1)

        force_weight[swing]  = -1.0
        vel_weight[swing]    =  0.0
        force_weight[stance] =  0.0
        vel_weight[stance]   = -1.0

        return vel_weight * velocity + force_weight * force

    def _set_gait(self, gait_name: GaitName, env_ids: torch.Tensor | None = None):
        
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
            
        offsets = GAIT_OFFSETS[gait_name]
            
        self.foot_offset[env_ids, 0] = offsets["L"]   # ft_i
        self.foot_offset[env_ids, 1] = offsets["R"]   # ft_d

        n = len(env_ids)
        self.gait_period[env_ids, 0] = torch.empty(n, device=gs.device).uniform_(*GAIT_PERIOD_HINTS[gait_name])
        self.foot_height[env_ids, 0] = torch.empty(n, device=gs.device).uniform_(*GAIT_CLEARANCE_HINTS[gait_name])

    def _generate_random_gait_indices(self, num: int) -> torch.Tensor:
        if not self._all_gaits_learned:
            weights = torch.arange(self._num_gaits, device=gs.device, dtype=torch.float).exp()
        else:
            weights = torch.ones(self._num_gaits, device=gs.device)
        weights /= weights.sum()
        return torch.multinomial(weights.expand(num, -1), 1).squeeze(-1)

    def _process_gamepad_input(self):
        buttons = self._gamepad.buttons() #type: ignore
        if "a" in buttons:
            self._gamepad_btn_pressed = True
        elif self._gamepad_btn_pressed:
            self._gamepad_btn_pressed = False
            self._gamepad_gait_idx = (self._gamepad_gait_idx + 1) % self._num_gaits
            self._gamepad_select_gait(list(GAIT_OFFSETS.keys())[self._gamepad_gait_idx])

    def _gamepad_select_gait(self, gait_name: GaitName):
        print(f"🚶 Marcha seleccionada: {gait_name}")
        offsets = GAIT_OFFSETS[gait_name]
        self.foot_offset[0, 0] = offsets["L"]
        self.foot_offset[0, 1] = offsets["R"]
        self.gait_period[0, 0] = sum(GAIT_PERIOD_HINTS[gait_name]) / 2
        self.foot_height[0, 0] = sum(GAIT_CLEARANCE_HINTS[gait_name]) / 2

    def _log_metrics(self):
        log = self.env.extras[self.env.extras_logging_key]
        log["Metrics / num_gaits"] = self._num_gaits
        for i, name in enumerate(GAIT_OFFSETS.keys()):
            log[f"Metrics / gait_{name}_envs"] = (self._gait_selected == i).sum()