import os
import copy
import torch
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import (
    VideoWrapper,
    RslRlWrapper,
)
from environment import BipedGaitTrainingEnv
from ppo_cfg    import training_cfg   # ScaffoldRL: config de red y algoritmo
from run_logger import (               # ScaffoldRL: observabilidad del experimento
    setup_log_dir,
    save_cfg,
    print_run_summary,
)

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
parser.add_argument("-n", "--num_envs",        type=int,   default=4096)
parser.add_argument("--max_iterations",        type=int,   default=2000)
parser.add_argument("-d", "--device",          type=str,   default="gpu")
parser.add_argument("-e", "--exp_name",        type=str,   default=EXPERIMENT_NAME)
parser.add_argument("--resume",                action="store_true")
parser.add_argument("--resume_path",           type=str,   default=None)
parser.add_argument("--record_interval",       type=int,   default=100)
parser.add_argument("--no_video",              action="store_true", default=False,
                    help="Desactiva grabacion de video")
args = parser.parse_args()

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

    # ── Logging (ScaffoldRL: delegado a run_logger) ────────────────────────
    log_path = setup_log_dir(args.exp_name, resume=args.resume)

    cfg = training_cfg(
        exp_name        = args.exp_name,
        max_iterations  = args.max_iterations,
        num_envs        = args.num_envs,
        record_interval = args.record_interval,
        resume          = args.resume,
        resume_path     = args.resume_path,
    )
    save_cfg(cfg, log_path)

    # ── Entorno ────────────────────────────────────────────────────────────
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

    # ── Entrenamiento ──────────────────────────────────────────────────────
    print_run_summary(cfg, args.num_envs, args.max_iterations)

    runner = OnPolicyRunner(
        env,  # type: ignore
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