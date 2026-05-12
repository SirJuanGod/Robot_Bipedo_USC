# evaluacion.py — solo el diagnóstico de altura, sin foot_flat en el env todavía

import torch
import genesis as gs
from environment import BipedoEnv

gs.init(backend=gs.cpu)
env = BipedoEnv(num_envs=1, headless=True, max_episode_length_s=5)
env.build()
env.reset()

# Sin la recompensa foot_flat aún — solo medir altura
for _ in range(20):
    obs, rew, done, truncated, info = env.step(torch.zeros(1, env.num_actions))

ft_d = env.robot.get_link("ft_d")
ft_i = env.robot.get_link("ft_i")
print(f"ft_d altura en reposo: {ft_d.get_pos()[0, 2].item():.4f} m")
print(f"ft_i altura en reposo: {ft_i.get_pos()[0, 2].item():.4f} m")