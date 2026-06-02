# Robot Bípedo USC — Entrenamiento RL con Sim2Real

Proyecto de investigación de la Universidad Santiago de Cali para el desarrollo y entrenamiento de un robot bípedo humanoide mediante **Aprendizaje por Refuerzo (RL)** con enfoque **Sim2Real**. Se utiliza el algoritmo **PPO (Proximal Policy Optimization)** con **critic privilegiado**, el simulador físico **Genesis** y la librería **rsl-rl-lib** para aprender políticas de locomoción bípeda transferibles a hardware real.

---

## Tabla de Contenidos

- [Características Principales](#características-principales)
- [Arquitectura Sim2Real](#arquitectura-sim2real)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Entrenar el Modelo](#entrenar-el-modelo)
  - [Evaluar el Modelo](#evaluar-el-modelo)
  - [Visualizar Resultados](#visualizar-resultados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración del Entorno](#configuración-del-entorno)
  - [Observaciones (Actor vs Critic)](#observaciones-actor-vs-critic)
  - [Sistema de Recompensas](#sistema-de-recompensas)
  - [Terminaciones](#terminaciones)
  - [Actuadores](#actuadores)
  - [Sistema de Marcha (Gait)](#sistema-de-marcha-gait)
  - [Currículo de Aprendizaje](#currículo-de-aprendizaje)
- [Hiperparámetros PPO](#hiperparámetros-ppo)
- [Solución de Problemas](#solución-de-problemas)
- [Referencias](#referencias)

---

## Características Principales

- **Sim2Real con Critic Privilegiado** — El actor solo usa sensores disponibles en hardware (IMU); el critic usa información completa del simulador durante entrenamiento.
- **Algoritmo PPO** — Implementación estable con schedule adaptativo de learning rate.
- **Entrenamiento Masivamente Paralelo** — 6000 entornos paralelos por defecto en GPU.
- **Sistema de Marcha Bípeda** — Generador de comandos de gait con fases swing/stance, clock signals y foot clearance.
- **Aprendizaje Curricular** — Progresión automática de dificultad basada en métricas de recompensa.
- **Domain Randomization** — Ruido en observaciones, actuadores y propiedades físicas para robustez sim2real.
- **Simulación Física** — Usa Genesis con solver Newton y colisiones habilitadas.
- **GPU/CPU Flexible** — Detección automática de CUDA con fallback a CPU.
- **Monitoreo con TensorBoard** — Seguimiento en tiempo real de todas las métricas.

---

## Arquitectura Sim2Real

El proyecto implementa la arquitectura de **critic privilegiado** para transferencia sim2real:

```
┌─────────────────────────────────────────────────────────┐
│                    ENTRENAMIENTO (Simulación)            │
│                                                         │
│  ┌──────────────┐        ┌───────────────────────────┐  │
│  │  Actor (MLP)  │        │    Critic (MLP)            │  │
│  │  256→128→64   │        │    512→256→128→64          │  │
│  │               │        │                           │  │
│  │  Obs: IMU     │        │  Obs: IMU + DOF pos/vel   │  │
│  │  + comandos   │        │  + fuerzas de contacto    │  │
│  │  + acciones   │        │  + posiciones de pies     │  │
│  │               │        │  + velocidades de links   │  │
│  └──────┬───────┘        └───────────────────────────┘  │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │  Acciones     │ ──→ Simulador Genesis                │
│  │  (13 DOFs)    │                                      │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    DESPLIEGUE (Robot Real)               │
│                                                         │
│  ┌──────────────┐                                       │
│  │  Actor (MLP)  │  ← Solo este se transfiere           │
│  │  256→128→64   │                                      │
│  │               │                                      │
│  │  Obs: IMU     │  ← Giroscopio + Acelerómetro         │
│  │  + comandos   │  ← Enviados desde controlador        │
│  │  + acciones   │  ← Propias del agente                │
│  └──────┬───────┘                                       │
│         │                                               │
│         ▼                                               │
│  ┌──────────────┐                                       │
│  │  13 Servos    │ ──→ Robot físico                      │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> El robot real **no tiene encoders en los actuadores**. La política del actor fue diseñada para funcionar únicamente con datos de una IMU (giroscopio + acelerómetro). El critic privilegiado solo se usa durante el entrenamiento en simulación.

---

## Requisitos del Sistema

### Hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| CPU | Intel i7 / AMD Ryzen 5 | Intel i9 / AMD Ryzen 7 |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA con CUDA (4+ GB VRAM) | NVIDIA RTX (8+ GB VRAM) |

> [!NOTE]
> Con 6000 entornos paralelos se necesitan al menos 6-8 GB de VRAM. Si tienes menos, reduce con `-n 2000` o `-n 1000`.

### Software

- **Python**: 3.10+ (recomendado 3.11 o 3.12)
- **CUDA**: 11.8+ (necesario para GPU)
- **rsl-rl-lib**: >= 2.2.4

---

## Instalación

### 1. Crear ambiente virtual

```bash
conda create -n robot_bpd python=3.11
conda activate robot_bpd
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Verificar instalación

```bash
python -c "import genesis; import torch; import rsl_rl; print('OK')"
```

---

## Uso

### Entrenar el Modelo

```bash
# Entrenamiento por defecto (GPU, 6000 envs, 2000 iteraciones)
python train.py

# Personalizar
python train.py -n 4000 --max_iterations 3000 -e "experimento-v2"

# Reanudar entrenamiento previo
python train.py --resume -e "bipedo-usc-v1"

# Forzar CPU (si no hay GPU)
python train.py -d cpu -n 100
```

**Parámetros disponibles:**

| Parámetro | Abreviatura | Default | Descripción |
|-----------|-------------|---------|-------------|
| `--num_envs` | `-n` | `6000` | Entornos paralelos |
| `--max_iterations` | — | `2000` | Iteraciones máximas |
| `--device` | `-d` | `"gpu"` | Dispositivo: `"cpu"` o `"gpu"` |
| `--exp_name` | `-e` | `"bipedo-usc-v1"` | Nombre del experimento |
| `--resume` | — | `false` | Reanudar entrenamiento previo |
| `--no_video` | — | `true` | Desactivar grabación de video |

> [!TIP]
> Tiempos estimados con GPU (RTX 3060+):
> - **2000 iteraciones, 6000 envs**: ~1-2 horas
> - **CPU con 100 envs**: ~8-12 horas

### Evaluar el Modelo

```bash
python eval.py -e "bipedo-usc-v1"
python eval.py -d gpu    # Evaluación con GPU
```

> [!NOTE]
> La evaluación carga automáticamente el último checkpoint disponible en `logs/<exp_name>/`.

### Visualizar Resultados

```bash
tensorboard --logdir=./logs
```

Métricas disponibles:
- Recompensa episódica total y por componente
- Pérdidas de actor y critic
- Entropía de la política
- Métricas de curriculum (gaits desbloqueadas, clearance)
- Tasas de terminación (timeout, caída, contacto)

---

## Estructura del Proyecto

```
Robot_Bipedo_USC/
├── train.py                    # Script de entrenamiento PPO
├── eval.py                     # Script de evaluación
├── environment.py              # Entorno RL (recompensas, observaciones, terminaciones)
├── gait_command_manager.py     # Generador de comandos de marcha bípeda
├── requirements.txt            # Dependencias Python
├── confi_cad.ipynb             # Notebook de configuración CAD
├── README.md
│
├── model/
│   ├── Bipedo.xml              # Robot MJCF (13 DOFs, ~1 kg, 23.4 cm)
│   ├── config.json             # Config de exportación OnShape
│   ├── scene.xml               # Escena de simulación
│   └── assets/                 # Meshes STL del robot
│
└── logs/                       # Resultados de entrenamiento
    └── bipedo-usc-v1/
        ├── model_*.pt          # Checkpoints (cada 100 iter)
        ├── events.out.tfevents # Datos TensorBoard
        └── cfgs.pkl            # Configuración del experimento
```

---

## Configuración del Entorno

### Robot

| Propiedad | Valor |
|-----------|-------|
| Altura | 0.234 m (23.4 cm) |
| Masa total | ~1 kg |
| DOFs | 13 (brazos: BD, HD, HI, BI · cabeza: HEAD · piernas: CD, LD, KD, FD, CI, LI, KI, FI) |
| Frecuencia de control | 50 Hz (`dt = 0.02s`) |
| Substeps de simulación | 2 |
| Episodio máximo | 20 segundos |

### Observaciones (Actor vs Critic)

#### Actor — Grupo `"policy"` (sensores reales)

Solo contiene lo que está disponible en el robot físico:

| Observación | Dimensión | Sensor Real |
|-------------|-----------|-------------|
| `velocity_cmd` | 3 | Comando de velocidad (controlador) |
| `gait_command` | 8 | Comando de marcha + clock signals |
| `imu_ang_velocity` | 3 | Giroscopio IMU (ruido: ±0.05 rad/s) |
| `imu_projected_gravity` | 3 | Acelerómetro IMU (ruido: ±0.05) |
| `actions` | 13 | Últimas acciones enviadas |

> [!IMPORTANT]
> El actor tiene `history_len=5`, por lo que recibe los últimos 5 timesteps concatenados. Esto compensa la falta de encoders dándole memoria temporal implícita.

#### Critic — Grupo `"policy"` + `"critic"` (privilegiado)

Además de todo lo del actor, el critic recibe:

| Observación | Fuente |
|-------------|--------|
| `linear_velocity`, `angular_velocity` | Estado exacto del simulador |
| `projected_gravity` | Sin ruido |
| `dof_pos`, `dof_vel`, `dof_force` | Posiciones, velocidades y fuerzas de articulaciones |
| `feet_pos`, `feet_vel` | Posición y velocidad de los pies |
| `ankle_pos`, `knee_pos`, `knee_vel` | Estado de tobillos y rodillas |
| `foot_contact_force`, `knee_contact_force` | Fuerzas de contacto |
| `gait_clock`, `gait_phase_raw` | Fase de marcha exacta |
| `robot_pos`, `robot_quat` | Pose global del robot |
| `current_actions` | Acciones actuales |

### Sistema de Recompensas

| Recompensa | Peso | Tipo | Descripción |
|------------|------|------|-------------|
| `gait_phase_reward` | +1.5 | Positiva | Seguimiento de fases swing/stance |
| `foot_height_reward` | +0.9 | Positiva | Clearance del pie durante swing |
| `tracking_lin_vel` | +1.0 | Positiva | Seguimiento de velocidad lineal comandada |
| `tracking_ang_vel` | +0.5 | Positiva | Seguimiento de velocidad angular comandada |
| `feet_air_time` | +2.0 | Positiva | Tiempo de vuelo de pies (0.2–0.5s) |
| `lin_vel_z` | −0.5 | Penalización | Oscilación vertical del torso |
| `ang_vel_xy` | −0.05 | Penalización | Balanceo lateral |
| `body_acceleration` | −0.5 | Penalización | Aceleraciones bruscas |
| `base_height_target` | −5.0 | Penalización | Desviación de altura objetivo (0.184 m) |
| `action_rate` | −0.015 | Penalización | Cambios bruscos entre acciones |
| `similar_to_default` | −0.05 | Penalización | Desviación de la pose por defecto |
| `bad_contact` | −1.5 | Penalización | Contacto de rodillas/tobillos con el suelo |

### Terminaciones

| Condición | Parámetro |
|-----------|-----------|
| Timeout | Fin del episodio (20s) |
| Contacto del torso | Fuerza > 1.0 N en `cbh_d_2` |
| Caída | Inclinación > 45° |

### Actuadores

| Parámetro | Valor |
|-----------|-------|
| Ganancia proporcional (`kp`) | 20.0 (±15% ruido) |
| Ganancia derivativa (`kv`) | 0.4 (±15% ruido) |
| Damping | 0.5 (±25% ruido) |
| Friction loss | 0.15 (±10% ruido) |
| Torque máx. brazos | 2.0 Nm |
| Torque máx. piernas | 3.0 Nm |
| Torque máx. pies | 2.5 Nm |
| Torque máx. cabeza | 1.0 Nm |
| Action scale | 0.4 rad (~23°) |

### Sistema de Marcha (Gait)

Definido en `gait_command_manager.py`:

| Marcha | Offset L/R | Período | Clearance |
|--------|------------|---------|-----------|
| `walk` | 0.0 / 0.5 | 0.55–0.75s | 0.04–0.10 m |
| `run` | 0.0 / 0.5 | 0.25–0.40s | 0.12–0.22 m |

**Rangos globales de curriculum:**
- Período de gait: 0.3–0.6 s
- Foot clearance: 0.02–0.08 m

**Comandos de velocidad:**
- Velocidad lineal X: [-0.7, 0.7] m/s
- Velocidad lineal Y: 0.0 m/s
- Velocidad angular Z: [-0.5, 0.5] rad/s
- Probabilidad de comando "quieto": 15%

### Currículo de Aprendizaje

El currículo se evalúa automáticamente cada 50 pasos:

| Métrica | Umbral | Acción |
|---------|--------|--------|
| `gait_phase_reward` > 0.75 | Se desbloquea la siguiente marcha y se amplía el rango de período |
| `foot_height_reward` > 0.80 | Se amplía el rango de foot clearance |

---

## Hiperparámetros PPO

Definidos en `train.py`:

```python
"clip_param":       0.2       # Clip ratio de PPO
"learning_rate":    3e-4      # LR inicial (schedule adaptativo)
"gamma":            0.99      # Factor de descuento
"lam":              0.95      # GAE lambda
"entropy_coef":     0.05      # Exploración
"desired_kl":       0.01      # KL target para schedule adaptativo
"num_learning_epochs": 5      # Épocas por iteración
"num_mini_batches":    4      # Mini-batches por época
"num_steps_per_env":  24      # Pasos de rollout por entorno (mínimo)
```

**Arquitectura de redes:**

| Red | Capas | Activación |
|-----|-------|------------|
| Actor | 256 → 128 → 64 | ELU |
| Critic | 512 → 256 → 128 → 64 | ELU |

Ambas redes usan normalización de observaciones (`obs_normalization: True`).

---

## Solución de Problemas

### El entrenamiento es muy lento

```bash
# Verificar GPU
python -c "import torch; print(torch.cuda.is_available())"

# Usar GPU explícitamente
python train.py -d gpu
```

### CUDA out of memory

Reduce el número de entornos:
```bash
python train.py -n 2000 -d gpu
```

### El modelo no converge

1. Verifica en TensorBoard que las recompensas positivas crecen
2. Revisa que la entropía no colapse a 0 (debe mantenerse > 0.01)
3. Si el KL divergence sube mucho, reduce `learning_rate`

### Módulos no encontrados

```bash
pip install genesis-world
pip install --upgrade rsl-rl-lib
pip install -r requirements.txt --force-reinstall
```

---

## Referencias

- **Genesis Simulator**: [Documentación oficial](https://genesis-world.readthedocs.io)
- **RSL-RL Library**: [Repositorio en GitHub](https://github.com/leggedrobotics/rsl_rl)
- **PPO Algorithm**: Schulman et al., 2017 — *Proximal Policy Optimization Algorithms*
- **OnShape CAD**: [Modelo del robot](https://cad.onshape.com/documents/c4b361cf4ba327d1f57e882d/w/2ee79dd7520cc42214465bb5/e/7518343c4784b4c1e8e306be)

---

## Autor y Contacto

**Proyecto**: Robot Bípedo USC
**Institución**: Universidad Santiago de Cali
**Período**: 2024–2026

---

## Historial de Versiones

| Versión | Cambios |
|---------|---------|
| v1.0 | Versión inicial del proyecto |
| v1.1 | Soporte para GPU |
| v1.2 | Implementación del currículo de aprendizaje |
| v2.0 | Arquitectura sim2real con critic privilegiado, sistema de gait bípedo, domain randomization, rebalanceo de recompensas, actuadores calibrados para hardware real |

---

> [!NOTE]
> **Última actualización**: Mayo 2026 | **Estado**: En desarrollo activo