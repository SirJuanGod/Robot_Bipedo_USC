import os
import copy
import torch
import shutil
import pickle
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import (
    VideoWrapper,
    RslRlWrapper,
)
from environment import BipedGaitTrainingEnv

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib").startswith("1."):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Por favor instala 'rsl-rl-lib>=2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

EXPERIMENT_NAME = "bipedo-usc-v1"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs",        type=int,   default=100)
parser.add_argument("--max_iterations",        type=int,   default=4000)
parser.add_argument("-d", "--device",          type=str,   default="gpu")
parser.add_argument("-e", "--exp_name",        type=str,   default=EXPERIMENT_NAME)
parser.add_argument("--resume",                action="store_true")
parser.add_argument("--resume_path",           type=str,   default=None)
parser.add_argument("--record_interval",       type=int,   default=100)
parser.add_argument("--no_video",              action="store_true", default=False,
                    help="Desactiva grabacion de video")
args = parser.parse_args()

def training_cfg(exp_name: str, max_iterations: int, num_envs: int) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param":                    0.2,
            "desired_kl":                    0.01,
            "entropy_coef":                  0.01,
            "gamma":                         0.99,
            "lam":                           0.95,
            "learning_rate":                 4e-4,
            "max_grad_norm":                 1.0,
            "num_learning_epochs":           5,
            "num_mini_batches":              4,
            "schedule":                      "adaptive",
            "use_clipped_value_loss":        True,
            "value_loss_coef":               1.0,
            "normalize_advantage_per_mini_batch": True,
            "optimizer":                     "adam",
            "rnd_cfg":                       None,
            "symmetry_cfg":                  None,
        },

        "actor": {
            "class_name":      "MLPModel",
            "hidden_dims":     [512,256, 128],
            "activation":      "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std":   0.1,
            },
        },

        "critic": {
            "class_name":      "MLPModel",
            "hidden_dims":     [1024, 512, 256, 128],
            "activation":      "elu",
            "obs_normalization": True,
        },

        "obs_groups": {
            "actor":  ["policy"],            # solo sensores reales
            "critic": ["policy", "critic"],  # policy + privilegiados del sim
        },

        "runner": {
            "checkpoint":      -1,
            "experiment_name": exp_name,
            "load_run":        -1,
            "log_interval":    1,
            "max_iterations":  max_iterations,
            "record_interval": args.record_interval,
            "resume":          args.resume,
            "resume_path":     args.resume_path,
            "run_name":        "",
        },
        "runner_class_name": "OnPolicyRunner",

        "seed": 1,
        "num_steps_per_env":     max(24, round( 98_304 / num_envs)),
        "save_interval":         100,
        "empirical_normalization": None,
        "torch_compile_mode":    None,
        "multi_gpu":             None,
    }


# ──────────────────────────────────────────────────────────────────────────────
def main():

    cuda_available = torch.cuda.is_available()
    if args.device == "gpu" and not cuda_available:
        print("⚠️  CUDA no disponible — forzando CPU.")
        args.device = "cpu"

    if args.device == "cpu":
        backend = gs.cpu  # type: ignore
        torch.set_default_device("cpu")
        num_threads = min(6, torch.get_num_threads())
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(2)
        print(f"CPU mode: {num_threads} threads (torch) + 2 interop")
    else:
        backend = gs.gpu  # type: ignore

    gs.init(logging_level="warning", backend=backend, performance_mode=True)

    log_path = os.path.join("./logs", args.exp_name)
    if os.path.exists(log_path) and not args.resume:
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"📂 Logging en: {log_path}")

    cfg = training_cfg(args.exp_name, args.max_iterations, args.num_envs)
    pickle.dump(
        [cfg],
        open(os.path.join(log_path, "cfgs.pkl"), "wb"),
    )

    env = BipedGaitTrainingEnv(
        num_envs=args.num_envs,
        headless=True,
    )

    if not args.no_video:
        env = VideoWrapper(
            env,
            video_length_sec=12,
            out_dir=os.path.join(log_path, "videos"),
            episode_trigger=lambda episode_id: episode_id % 4 == 0,
        )
    else:
        print("Video desactivado. Solo se guardarán checkpoints.")

    env = RslRlWrapper(env)  # type: ignore
    env.build()
    env.reset()
    env.cfg = {}  # type: ignore

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    print("💪 Iniciando entrenamiento PPO con critic privilegiado...")
    print(f"   num_envs        : {args.num_envs}")
    print(f"   num_steps_per_env: {cfg['num_steps_per_env']}")
    print(f"   max_iterations  : {args.max_iterations}")
    print(f"   obs actor       : {cfg['obs_groups']['actor']}")
    print(f"   obs critic      : {cfg['obs_groups']['critic']}")

    runner = OnPolicyRunner(
        env, #type: ignore
        copy.deepcopy(cfg),
        log_path,
        device=gs.device,  # type: ignore
    )
    runner.git_status_repos = ["."]  # type: ignore
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=False,
    )

    env.close()


if __name__ == "__main__":
    main()