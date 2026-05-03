# Robot Bípedo USC — Entrenamiento con Aprendizaje por Refuerzo

Proyecto de investigación para el desarrollo y entrenamiento de un robot bípedo humanoide mediante **Aprendizaje por Refuerzo (RL)** usando el algoritmo **PPO (Proximal Policy Optimization)**. El proyecto utiliza el simulador **Genesis** y la librería **rsl-rl-lib** para entrenar políticas de control locomotor.

---

## Tabla de Contenidos

- [Características Principales](#características-principales)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Configuración del Ambiente](#configuración-del-ambiente)
- [Uso](#uso)
  - [Entrenar el Modelo](#entrenar-el-modelo)
  - [Evaluar el Modelo](#evaluar-el-modelo)
  - [Visualizar Resultados](#visualizar-resultados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración de Parámetros](#configuración-de-parámetros)
- [Puntos Clave a Considerar](#puntos-clave-a-considerar)
- [Solución de Problemas](#solución-de-problemas)
- [Referencias](#referencias)

---

## Características Principales

- **Algoritmo PPO** — Implementación de Proximal Policy Optimization para entrenamiento estable.
- **Entrenamiento Distribuido** — Soporte para múltiples entornos paralelos (200 por defecto).
- **Aprendizaje Curricular** — 4 fases de dificultad progresiva para facilitar el aprendizaje.
- **Simulación Física** — Utiliza Genesis, un simulador físico de alto rendimiento.
- **GPU/CPU Flexible** — Soporta entrenamiento tanto en GPU como en CPU.
- **Registro de Videos** — Captura automática de videos durante el entrenamiento.
- **Monitoreo con TensorBoard** — Seguimiento en tiempo real de métricas de entrenamiento.

---

## Requisitos del Sistema

### Hardware Mínimo

| Componente | Requisito Mínimo | Recomendado |
|------------|-----------------|-------------|
| CPU | Intel i7 / AMD Ryzen 5 | Intel i9 / AMD Ryzen 7 |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA con CUDA | NVIDIA RTX (4+ GB VRAM) |

> [!NOTE]
> Si no dispones de GPU NVIDIA, el entrenamiento puede ejecutarse en CPU, aunque será considerablemente más lento.

### Software Requerido

- **Python**: 3.10 o superior (se recomienda 3.11 o 3.12)
- **Conda** o **Pip**: Gestor de paquetes Python
- **CUDA**: 11.8 o superior (opcional, pero recomendado para GPU)

---

## Instalación

### Paso 1: Acceder al Directorio del Proyecto

```bash
cd d:\USC\Proyectos\BPD_RBT_USC\Robot_Bipedo_USC
```

### Paso 2: Crear el Ambiente Virtual

#### Opción A: Conda (Recomendado)

```bash
# Crear el ambiente
conda create -n robot_bpd python=3.11

# Activar el ambiente
conda activate robot_bpd
```

#### Opción B: Virtualenv

```bash
# Crear el ambiente
python -m venv robot_bpd

# Activar en Windows
robot_bpd\Scripts\activate

# Activar en Linux/Mac
source robot_bpd/bin/activate
```

### Paso 3: Instalar Dependencias

> [!WARNING]
> Revisa el archivo `requirements.txt` antes de instalar. Algunos paquetes pueden tener incompatibilidades de versión.

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar la Instalación

```bash
python -c "import genesis; import torch; import rsl_rl; print('Todas las dependencias instaladas correctamente')"
```

> [!NOTE]
> Si la instalación fue exitosa, verás el mensaje de confirmación. En caso de errores, consulta la sección [Solución de Problemas](#solución-de-problemas).

---

## Configuración del Ambiente

### Archivo `environment.py`

Define el entorno de simulación del robot bípedo con los siguientes parámetros:

| Parámetro | Valor |
|-----------|-------|
| Posición inicial | `[0.0, 0.0, 0.228501]` metros |
| Velocidad máxima | `0.5` m/s |
| Dimensiones de observación | `25` (ángulos articulares, velocidades, comandos) |
| Fases del currículo | 4 fases progresivas |

### Archivo `config.json`

Contiene la configuración del modelo CAD exportado desde OnShape:

```json
{
  "url": "https://cad.onshape.com/...",
  "output_format": "mujoco",
  "joint_properties": {
    "rango_fuerza": 10.0,
    "limites_angulos": [-1.5708, 1.5708],
    "ganancia_proporcional": 30,
    "ganancia_velocidad": 3.0
  }
}
```

> [!CAUTION]
> No modifiques estos valores sin comprender sus efectos en la dinámica del robot. Cambios inadecuados pueden impedir que la política converja durante el entrenamiento.

---

## Uso

### Entrenar el Modelo

#### Entrenamiento por Defecto

```bash
python train.py
```

Esto iniciará el entrenamiento con la configuración predeterminada:

- 200 entornos paralelos
- 1500 iteraciones máximas
- Dispositivo: CPU (o GPU si está disponible)
- Nombre del experimento: `bipedo-usc-ppo-v1`

#### Entrenamiento Personalizado

```bash
# Especificar número de entornos
python train.py -n 100

# Especificar máximo de iteraciones
python train.py --max_iterations 2000

# Usar GPU
python train.py -d gpu

# Especificar nombre del experimento
python train.py -e "bipedo-usc-ppo-v2"

# Combinar parámetros
python train.py -n 300 --max_iterations 3000 -d gpu -e "entrenamiento-v3"
```

**Parámetros disponibles:**

| Parámetro | Abreviatura | Tipo | Por Defecto | Descripción |
|-----------|-------------|------|-------------|-------------|
| `--num_envs` | `-n` | int | 200 | Número de entornos paralelos |
| `--max_iterations` | — | int | 1500 | Máximo de iteraciones de entrenamiento |
| `--device` | `-d` | str | `"cpu"` | Dispositivo: `"cpu"` o `"gpu"` |
| `--exp_name` | `-e` | str | `"bipedo-usc-ppo-v1"` | Nombre del experimento |

> [!IMPORTANT]
> El tiempo de entrenamiento varía considerablemente según el hardware:
> - **GPU**: aproximadamente 1–2 horas para 1500 iteraciones.
> - **CPU**: aproximadamente 6–8 horas para 1500 iteraciones.

### Evaluar el Modelo

```bash
# Evaluación con configuración por defecto
python eval.py

# Especificar dispositivo
python eval.py -d gpu

# Especificar nombre del experimento
python eval.py -e "bipedo-usc-ppo-v1"
```

> [!NOTE]
> Si no existen modelos entrenados previamente, la evaluación fallará. Ejecuta primero `python train.py`.

### Visualizar Resultados

```bash
# Activar el ambiente
conda activate robot_bpd

# Iniciar TensorBoard
tensorboard --logdir=./logs
```

> [!TIP]
> Métricas disponibles en TensorBoard:
> - Recompensa episódica
> - Pérdida del actor y del crítico
> - Tasa de entropía
> - Velocidad del robot
> - Pasos de entrenamiento completados

---

## Estructura del Proyecto
```
Robot_Bipedo_USC/
├── train.py                      # Script principal de entrenamiento
├── eval.py                       # Script de evaluación
├── environment.py                # Definición del entorno
├── requirements.txt              # Dependencias del proyecto
├── confi_cad.ipynb               # Notebook de configuración CAD
├── README.md                     # Este archivo
│
├── model/                       #Configuración del robot
│   ├── Bipedo.xml               # Archivo de definición del robot
│   ├── config.json              # Configuración de parámetros
│   ├── scene.xml                # Escena de simulación
│   └── assets/                  # Modelos 3D del robot
│       ├── ank_d.part           # Tobillo derecho
│       ├── ank_i.part           # Tobillo izquierdo
│       ├── kne_d.part           # Rodilla derecha
│       ├── kne_i.part           # Rodilla izquierda
│       └── ... (más partes)
│
└── logs/                        # Resultados del entrenamiento
    └── bipedo-usc-ppo-v1/
        ├── model_0.pt           # Checkpoints del modelo
        ├── events.out.tfevents  # Datos de TensorBoard
        ├── videos/              # Videos de entrenamiento
        └── cfgs.pkl             # Configuración guardada
```

---

## Configuración de Parámetros

### Hiperparámetros de PPO (`train.py`)

```python
"clip_param":      0.2       # Límite del ratio de política (clip)
"learning_rate":   0.0003    # Tasa de aprendizaje
"gamma":           0.99      # Factor de descuento
"entropy_coef":    0.008     # Coeficiente de entropía
"max_iterations":  1500      # Máximo de iteraciones
"num_envs":        200       # Número de entornos paralelos
```

### Fases del Currículo de Aprendizaje

| Fase | Objetivo |
|------|----------|
| **Fase 1** | Mantener equilibrio básico |
| **Fase 2** | Aumentar la velocidad de caminata |
| **Fase 3** | Mejorar estabilidad y eficiencia de movimiento |
| **Fase 4** | Optimizar el movimiento natural del robot |

---

## Puntos Clave a Considerar

> [!CAUTION]
> **Versión de rsl-rl-lib**: Debe ser `>= 2.2.4`. Verifica con:
> ```bash
> pip show rsl-rl-lib
> ```

> [!CAUTION]
> **Dimensión de observación**: El observador debe tener exactamente **25 dimensiones**. No modifiques este valor sin ajustar también la arquitectura de la red neuronal.

> [!CAUTION]
> **Memoria GPU**: Se requieren al menos **4 GB de VRAM** para ejecutar 200 entornos. Si hay problemas de memoria, reduce el número de entornos con `--num_envs`.

> [!NOTE]
> - **Convergencia**: El modelo comienza a converger alrededor de las iteraciones 500–800.
> - **Checkpoints**: Los modelos se guardan automáticamente cada 100 iteraciones en `logs/`.
> - **Videos**: Se generan automáticamente cada 2 episodios para seguimiento visual.

---

## Solución de Problemas

### El entrenamiento es muy lento

**Causa**: Se está usando CPU en lugar de GPU.

```bash
# Verificar disponibilidad de GPU
python -c "import torch; print(torch.cuda.is_available())"

# Si devuelve True, entrena con GPU
python train.py -d gpu
```

### Error de memoria insuficiente (CUDA out of memory)

**Causa**: Demasiados entornos paralelos para la VRAM disponible.

```bash
# Reducir número de entornos
python train.py -n 64 -d gpu

# O usar CPU como alternativa
python train.py -n 100 -d cpu
```

### El modelo no converge

**Causas posibles**:
- Tasa de aprendizaje demasiado alta.
- Distribución incorrecta en las observaciones.
- Arquitectura de red neuronal inadecuada.

**Solución**: Monitorea las métricas en TensorBoard y ajusta los hiperparámetros en `train.py`.

### Errores de módulos no encontrados

> [!WARNING]
> **`No module named 'genesis'`**
> ```bash
> pip install genesis-world
> ```

> [!WARNING]
> **`Please install 'rsl-rl-lib>=2.2.4'`**
> ```bash
> pip install --upgrade rsl-rl-lib
> ```

> [!WARNING]
> **`ModuleNotFoundError: No module named 'rsl_rl'`**
> ```bash
> pip install -r requirements.txt --force-reinstall
> ```

### No hay modelos para evaluar

**Error**: `Warning: No model files found`

```bash
# Primero entrena el modelo
python train.py

# Luego evalúa
python eval.py
```

---

## Variables de Entorno

```bash
# Windows
set GENESIS_BACKEND=gpu
set TORCH_DEVICE=cuda

# Linux/Mac
export GENESIS_BACKEND=gpu
export TORCH_DEVICE=cuda
```

---

## Dependencias Principales

| Paquete | Versión Requerida | Propósito |
|---------|------------------|-----------|
| `genesis-world` | Última estable | Simulador físico |
| `torch` | >= 2.0 | Framework de aprendizaje profundo |
| `rsl-rl-lib` | >= 2.2.4 | Implementación del algoritmo PPO |
| `numpy` | — | Operaciones numéricas |
| `tensorboard` | — | Visualización de métricas de entrenamiento |

> [!WARNING]
> Algunos paquetes pueden generar conflictos de versión entre sí. Si ocurren problemas de compatibilidad, reinstala usando el archivo original:
> ```bash
> pip install -r requirements.txt --force-reinstall
> ```

---

## Tips de Optimización

### Para entrenamientos más rápidos

1. Aumentar el número de entornos (si hay VRAM disponible):
   ```bash
   python train.py -n 400 -d gpu
   ```
2. Usar GPU en lugar de CPU:
   ```bash
   python train.py -d gpu
   ```
3. Reducir iteraciones para prototipado rápido:
   ```bash
   python train.py --max_iterations 500
   ```

### Para mejores resultados

1. Aumentar las iteraciones de entrenamiento:
   ```bash
   python train.py --max_iterations 3000
   ```
2. Reducir la tasa de aprendizaje en `train.py` para mayor estabilidad.
3. Monitorear activamente las métricas en TensorBoard durante el entrenamiento.

---

## Documentación de Archivos

| Archivo | Descripción |
|---------|-------------|
| `train.py` | Script principal de entrenamiento. Define la arquitectura neuronal, los parámetros de PPO y genera los checkpoints, videos y logs. |
| `eval.py` | Carga un modelo entrenado desde `logs/` y ejecuta el comportamiento del robot en simulación. |
| `environment.py` | Define la tarea, el espacio de estados/acciones y el currículo de 4 fases con recompensas personalizadas. |
| `confi_cad.ipynb` | Notebook para visualizar y configurar el modelo CAD antes del entrenamiento. |

---

## Referencias

- **Genesis Simulator**: [Documentación oficial](https://genesis-world.readthedocs.io)
- **RSL-RL Library**: [Repositorio en GitHub](https://github.com/leggedrobotics/rsl_rl)
- **PPO Algorithm**: Schulman et al., 2017 — *Proximal Policy Optimization Algorithms*
- **OnShape CAD**: [https://cad.onshape.com/](https://cad.onshape.com/)

---

## Autor y Contacto

**Proyecto**: Robot Bípedo USC  
**Institución**: Universidad Santiago de Cali  
**Período**: 2024–2026

Para soporte, revisa en este orden:
1. Este README
2. Los logs en `./logs/bipedo-usc-ppo-v1/`
3. La salida en terminal durante la ejecución

---

## Licencia

> [!NOTE]
> Actualmente el proyecto no cuenta con una licencia oficial.

---

## Historial de Versiones

| Versión | Cambios |
|---------|---------|
| v1.0 | Versión inicial del proyecto |
| v1.1 | Adición de soporte para GPU |
| v1.2 | Implementación del currículo de aprendizaje |

---

> [!NOTE]
> **Última actualización**: Mayo 2026 | **Estado**: En desarrollo activo