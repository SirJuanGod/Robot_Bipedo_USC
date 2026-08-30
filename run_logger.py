"""
run_logger.py — Setup de carpetas, serialización de configuración y prints de inicio.

ScaffoldRL: este módulo es responsable únicamente de la observabilidad del
experimento (rutas de log, persistencia de config, resumen inicial). No
contiene lógica de política ni de entorno.
"""

import os
import pickle
import shutil


def setup_log_dir(exp_name: str, resume: bool, base_dir: str = "./logs") -> str:
    """
    Crea (o limpia) la carpeta de logs para el experimento.

    Si `resume` es False y la carpeta ya existe, la borra para empezar
    desde cero. Si `resume` es True, la conserva para continuar desde
    el checkpoint.

    Args:
        exp_name:  Nombre del experimento — se usa como subcarpeta.
        resume:    Si True, no borra los logs existentes.
        base_dir:  Raíz donde se crean las subcarpetas de experimentos.

    Returns:
        Ruta absoluta a la carpeta del experimento.
    """
    log_path = os.path.join(base_dir, exp_name)
    if os.path.exists(log_path) and not resume:
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"📂 Logging en: {log_path}")
    return log_path


def save_cfg(cfg: dict, log_path: str) -> None:
    """
    Serializa el dict de configuración a disco con pickle.

    Permite reproducir exactamente los hiperparámetros de cualquier run
    al hacer `pickle.load(open('cfgs.pkl', 'rb'))`.

    Args:
        cfg:      Dict de configuración completo (salida de ppo_cfg.training_cfg).
        log_path: Carpeta destino donde se escribe 'cfgs.pkl'.
    """
    out_path = os.path.join(log_path, "cfgs.pkl")
    with open(out_path, "wb") as f:
        pickle.dump([cfg], f)


def print_run_summary(cfg: dict, num_envs: int, max_iterations: int) -> None:
    """
    Imprime un resumen legible de los parámetros clave del experimento
    antes de iniciar el loop de entrenamiento.

    Args:
        cfg:            Dict de configuración completo.
        num_envs:       Número de entornos paralelos.
        max_iterations: Número total de iteraciones PPO.
    """
    print("💪 Iniciando entrenamiento PPO con critic privilegiado...")
    print(f"   num_envs         : {num_envs}")
    print(f"   num_steps_per_env: {cfg['num_steps_per_env']}")
    print(f"   max_iterations   : {max_iterations}")
    print(f"   obs actor        : {cfg['obs_groups']['actor']}")
    print(f"   obs critic       : {cfg['obs_groups']['critic']}")
