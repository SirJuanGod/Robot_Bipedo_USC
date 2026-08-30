import os
import pickle
import shutil


def setup_log_dir(exp_name: str, resume: bool, base_dir: str = "./logs") -> str:
    log_path = os.path.join(base_dir, exp_name)
    if os.path.exists(log_path) and not resume:
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"📂 Logging en: {log_path}")
    return log_path


def save_cfg(cfg: dict, log_path: str) -> None:
    out_path = os.path.join(log_path, "cfgs.pkl")
    with open(out_path, "wb") as f:
        pickle.dump([cfg], f)


def print_run_summary(cfg: dict, num_envs: int, max_iterations: int) -> None:
    print("💪 Iniciando entrenamiento PPO con critic privilegiado...")
    print(f"   num_envs         : {num_envs}")
    print(f"   num_steps_per_env: {cfg['num_steps_per_env']}")
    print(f"   max_iterations   : {max_iterations}")
    print(f"   obs actor        : {cfg['obs_groups']['actor']}")
    print(f"   obs critic       : {cfg['obs_groups']['critic']}")
