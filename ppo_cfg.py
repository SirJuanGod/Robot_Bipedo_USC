def training_cfg(exp_name: str, max_iterations: int, num_envs: int, record_interval: int = 100,
                 resume: bool = False, resume_path: str | None = None) -> dict:
    return {
        # ── Algoritmo PPO ───────────────────────────────────────────────────
        "algorithm": {
            "class_name":                      "PPO",
            "clip_param":                       0.2,
            "desired_kl":                       0.010,
            "entropy_coef":                     0.003,
            "gamma":                            0.99,
            "lam":                              0.95,
            "learning_rate":                    1e-4,
            "max_grad_norm":                    1.0,
            "num_learning_epochs":              5,
            "num_mini_batches":                 4,
            "schedule":                         "adaptive",
            "use_clipped_value_loss":           True,
            "value_loss_coef":                  1.0,
            "normalize_advantage_per_mini_batch": True,
            "optimizer":                        "adam",
            "rnd_cfg":                          None,
            "symmetry_cfg":                     None,
        },

        # ── Red actor (solo observaciones de sensores reales) ───────────────
        "actor": {
            "class_name":       "MLPModel",
            "hidden_dims":      [512, 256, 128],
            "activation":       "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std":   1.0,
            },
        },

        # ── Red crítico (política + observaciones privilegiadas del sim) ────
        "critic": {
            "class_name":       "MLPModel",
            "hidden_dims":      [1024, 512, 256, 128],
            "activation":       "elu",
            "obs_normalization": True,
        },

        # ── Grupos de observación por rol ───────────────────────────────────
        "obs_groups": {
            "actor":  ["policy"],            # solo sensores reales
            "critic": ["policy", "critic"],  # policy + privilegiados del sim
        },

        # ── Runner ──────────────────────────────────────────────────────────
        "runner": {
            "checkpoint":       -1,
            "experiment_name":  exp_name,
            "load_run":         -1,
            "log_interval":     1,
            "max_iterations":   max_iterations,
            "record_interval":  record_interval,
            "resume":           resume,
            "resume_path":      resume_path,
            "run_name":         "",
        },
        "runner_class_name": "OnPolicyRunner",

        # ── Parámetros globales del loop ────────────────────────────────────
        "seed":               1,
        "num_steps_per_env":  round(98_304 / num_envs),  # steps totales / envs
        "save_interval":      100,
        "empirical_normalization": None,
        "torch_compile_mode": None,
        "multi_gpu":          None,
    }
