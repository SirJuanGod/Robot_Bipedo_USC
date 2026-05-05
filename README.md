# Robot Bípedo USC — Entrenamiento con Aprendizaje por Refuerzo

Proyecto de investigación de la Universidad Santiago de Cali para el desarrollo y entrenamiento de un robot bípedo humanoide mediante **Aprendizaje por Refuerzo (RL)**. Se utiliza el algoritmo **PPO (Proximal Policy Optimization)**, el simulador físico **Genesis** y la librería **rsl-rl-lib** para aprender políticas de locomoción bípeda.

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

- **Algoritmo PPO** — Implementación estable de Proximal Policy Optimization.
- **Entrenamiento Distribuido** — Soporte para múltiples entornos paralelos (200 por defecto).
- **Aprendizaje Curricular** — 4 fases de dificultad progresiva para facilitar la convergencia.
- **Simulación Física** — Usa Genesis, un simulador de alto rendimiento.
- **GPU/CPU Flexible** — El entorno soporta ambos dispositivos. El dispositivo se configura directamente en `train.py` o por argumento al ejecutar el script.
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
> El entorno de simulación soporta tanto CPU como GPU. Si no dispones de una GPU NVIDIA compatible, puedes ejecutar el entrenamiento en CPU; sin embargo, será considerablemente más lento. Para cambiar el dispositivo, consulta la sección [Entrenar el Modelo](#entrenar-el-modelo).

### Software Requerido

- **Python**: 3.10 o superior (se recomienda 3.11 o 3.12)
- **Conda** o **Pip**: Gestor de paquetes Python
- **CUDA**: 11.8 o superior (opcional, pero necesario para entrenar en GPU)

---

## Instalación

### Paso 1: Acceder al Directorio del Proyecto

```bash
cd d:\USC\Proyectos\BPD_RBT_USC\Robot_Bipedo_USC
```

### Paso 2: Crear el Ambiente Virtual

#### Opción A: Conda (Recomendado)

```bash
conda create -n robot_bpd python=3.11
conda activate robot_bpd
```

#### Opción B: Virtualenv

```bash
python -m venv robot_bpd

# Windows
robot_bpd\Scripts\activate

# Linux/Mac
source robot_bpd/bin/activate
```

### Paso 3: Instalar Dependencias

> [!WARNING]
> Revisa el archivo `requirements.txt` antes de instalar. Algunos paquetes pueden tener incompatibilidades de versión entre sí.

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar la Instalación

```bash
python -c "import genesis; import torch; import rsl_rl; print('Todas las dependencias instaladas correctamente')"
```

> [!NOTE]
> Si la instalación fue exitosa, verás el mensaje de confirmación en la terminal. En caso de errores, consulta la sección [Solución de Problemas](#solución-de-problemas).

---

## Configuración del Ambiente

### Archivo `environment.py`

Define el entorno de simulación del robot bípedo con los siguientes parámetros:

| Parámetro | Valor |
|-----------|-------|
| Posición inicial | `[0.0, 0.0, 0.2340]` metros |
| Velocidad máxima | `0.3` m/s |
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
> No modifiques los valores de `config.json` sin comprender su efecto en la dinámica del robot. Cambios inadecuados en las ganancias o límites articulares pueden impedir que la política converja durante el entrenamiento.

---

## Uso

### Entrenar el Modelo

#### Sobre el dispositivo de cómputo

> [!IMPORTANT]
> El script `train.py` usa **CPU por defecto**. El entorno de simulación está preparado para correr en GPU, pero debes indicarlo explícitamente al lanzar el entrenamiento. Puedes hacerlo de dos formas:
>
> **Opción 1 — Argumento al ejecutar:**
> ```bash
> python train.py -d gpu
> ```
>
> **Opción 2 — Cambiando el valor por defecto en `train.py`:**
> Busca la línea donde se define el parámetro `device` y cambia `"cpu"` por `"gpu"`:
> ```python
> # Antes
> parser.add_argument("-d", "--device", type=str, default="cpu")
>
> # Después
> parser.add_argument("-d", "--device", type=str, default="gpu")
> ```
> Con este cambio, todos los entrenamientos usarán GPU sin necesidad de pasar el argumento cada vez.

#### Entrenamiento por Defecto (CPU)

```bash
python train.py
```

Inicia el entrenamiento con la configuración predeterminada:
- 200 entornos paralelos
- 1500 iteraciones máximas
- Dispositivo: **CPU**
- Nombre del experimento: `bipedo-usc-ppo-v1`

#### Entrenamiento Personalizado

```bash
# Usar GPU (recomendado si está disponible)
python train.py -d gpu

# Especificar número de entornos paralelos
python train.py -n 100

# Especificar máximo de iteraciones
python train.py --max_iterations 2000

# Asignar un nombre al experimento
python train.py -e "bipedo-usc-ppo-v2"

# Combinar parámetros (ejemplo completo con GPU)
python train.py -n 300 --max_iterations 3000 -d gpu -e "entrenamiento-v3"
```

**Parámetros disponibles:**

| Parámetro | Abreviatura | Tipo | Por Defecto | Descripción |
|-----------|-------------|------|-------------|-------------|
| `--num_envs` | `-n` | int | `200` | Número de entornos paralelos |
| `--max_iterations` | — | int | `1500` | Máximo de iteraciones de entrenamiento |
| `--device` | `-d` | str | `"cpu"` | Dispositivo de cómputo: `"cpu"` o `"gpu"` |
| `--exp_name` | `-e` | str | `"bipedo-usc-ppo-v1"` | Nombre del experimento (afecta la carpeta de logs) |

> [!TIP]
> Si tienes GPU disponible, úsala siempre. La diferencia en tiempo es significativa:
> - **GPU**: aproximadamente 1–2 horas para 1500 iteraciones.
> - **CPU**: aproximadamente 6–8 horas para las mismas iteraciones.

### Evaluar el Modelo

```bash
# Evaluación con configuración por defecto
python eval.py

# Evaluación usando GPU
python eval.py -d gpu

# Especificar el experimento a evaluar
python eval.py -e "bipedo-usc-ppo-v1"
```

> [!NOTE]
> Si no existen modelos entrenados en `logs/`, la evaluación fallará con un aviso. Asegúrate de haber completado al menos un entrenamiento antes de ejecutar `eval.py`.

### Visualizar Resultados

```bash
conda activate robot_bpd
tensorboard --logdir=./logs
```

> [!TIP]
> Métricas disponibles en TensorBoard durante y después del entrenamiento:
> - Recompensa episódica acumulada
> - Pérdida del actor y del crítico
> - Tasa de entropía de la política
> - Velocidad de desplazamiento del robot
> - Número de pasos de entrenamiento completados

---

## Estructura del Proyecto
Robot_Bipedo_USC/
├── train.py                      # Script principal de entrenamiento
├── eval.py                       # Script de evaluación
├── environment.py                # Definición del entorno y currículo
├── requirements.txt              # Dependencias del proyecto
├── confi_cad.ipynb               # Notebook de configuración CAD
├── README.md                     # Este archivo
│
├── model/                        # Configuración del robot
│   ├── Bipedo.xml                # Definición del robot en formato MuJoCo
│   ├── config.json               # Parámetros de articulaciones y ganancias
│   ├── scene.xml                 # Escena de simulación
│   └── assets/                   # Modelos 3D del robot
│       ├── ank_d.part            # Tobillo derecho
│       ├── ank_i.part            # Tobillo izquierdo
│       ├── kne_d.part            # Rodilla derecha
│       ├── kne_i.part            # Rodilla izquierda
│       └── ... (más partes)
│
└── logs/                         # Resultados del entrenamiento
└── bipedo-usc-ppo-v1/
├── model_0.pt            # Checkpoints del modelo
├── events.out.tfevents   # Datos para TensorBoard
├── videos/               # Videos capturados durante entrenamiento
└── cfgs.pkl              # Configuración guardada del experimento

---

## Configuración de Parámetros

### Hiperparámetros de PPO (`train.py`)

Estos valores se encuentran directamente en `train.py` y pueden ajustarse antes de entrenar:

```python
"clip_param":      0.2       # Límite del ratio de política (clip de PPO)
"learning_rate":   0.0003    # Tasa de aprendizaje del optimizador
"gamma":           0.99      # Factor de descuento para recompensas futuras
"entropy_coef":    0.008     # Coeficiente de entropía (fomenta exploración)
"max_iterations":  1500      # Máximo de iteraciones de entrenamiento
"num_envs":        200       # Número de entornos paralelos
```

### Fases del Currículo de Aprendizaje

El entrenamiento avanza progresivamente a través de 4 fases definidas en `environment.py`:

| Fase | Objetivo |
|------|----------|
| **Fase 1** | Mantener el equilibrio básico en posición estática |
| **Fase 2** | Comenzar a caminar aumentando la velocidad objetivo |
| **Fase 3** | Mejorar la estabilidad y la eficiencia energética del movimiento |
| **Fase 4** | Optimizar la naturalidad y fluidez del paso del robot |

---

## Puntos Clave a Considerar

> [!CAUTION]
> **Versión de rsl-rl-lib**: debe ser `>= 2.2.4`. Para verificar la versión instalada:
> ```bash
> pip show rsl-rl-lib
> ```

> [!CAUTION]
> **Dimensión de observación**: el vector de observación debe tener exactamente **25 dimensiones**, definidas en `environment.py`. Si modificas este número, debes ajustar también la arquitectura de la red neuronal en `train.py` para que coincidan.

> [!CAUTION]
> **Memoria GPU**: se requieren al menos **4 GB de VRAM** para ejecutar 200 entornos en paralelo. Si el entrenamiento falla por falta de memoria, reduce el número de entornos:
> ```bash
> python train.py -d gpu -n 64
> ```

> [!NOTE]
> - **Convergencia**: el modelo típicamente comienza a mostrar comportamientos coherentes entre las iteraciones 500 y 800.
> - **Checkpoints**: el modelo se guarda automáticamente cada 100 iteraciones en `logs/`.
> - **Videos**: se generan automáticamente cada 2 episodios para seguimiento visual del progreso.

---

## Solución de Problemas

### El entrenamiento es muy lento

**Causa probable**: el entrenamiento está corriendo en CPU en lugar de GPU.

```bash
# Verifica si PyTorch detecta tu GPU
python -c "import torch; print(torch.cuda.is_available())"
```

Si devuelve `True`, tu GPU está disponible. Actívala con:

```bash
python train.py -d gpu
```

Si devuelve `False`, revisa que CUDA esté instalado correctamente y que el driver de tu GPU sea compatible.

---

### Error de memoria insuficiente (CUDA out of memory)

**Causa probable**: demasiados entornos paralelos para la VRAM disponible.

Reduce el número de entornos hasta que el error desaparezca:

```bash
# Prueba con menos entornos en GPU
python train.py -n 64 -d gpu

# O entrena en CPU si el problema persiste
python train.py -n 100 -d cpu
```

---

### El modelo no converge

**Causas posibles**:
- Tasa de aprendizaje demasiado alta o demasiado baja.
- Distribución incorrecta en las observaciones de `environment.py`.
- Arquitectura de red neuronal inadecuada para el espacio de estados.

Monitorea las métricas en TensorBoard para identificar si el problema es de exploración (entropía muy baja), de gradientes (pérdida inestable) o de recompensa (currículo mal calibrado). Ajusta los hiperparámetros en `train.py` según lo que observes.

---

### Errores de módulos no encontrados

> [!WARNING]
> **`No module named 'genesis'`** — El simulador no está instalado:
> ```bash
> pip install genesis-world
> ```

> [!WARNING]
> **`Please install 'rsl-rl-lib>=2.2.4'`** — La versión instalada es antigua:
> ```bash
> pip install --upgrade rsl-rl-lib
> ```

> [!WARNING]
> **`ModuleNotFoundError: No module named 'rsl_rl'`** — Reinstala todas las dependencias desde cero:
> ```bash
> pip install -r requirements.txt --force-reinstall
> ```

---

### No hay modelos disponibles para evaluar

**Error**: `Warning: No model files found`

El directorio `logs/` no contiene ningún checkpoint. Primero ejecuta un entrenamiento completo:

```bash
python train.py
python eval.py
```

---

## Variables de Entorno

Si prefieres configurar el dispositivo a través de variables de entorno del sistema operativo en lugar de argumentos:

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
> Algunos paquetes pueden generar conflictos de versión entre sí. Si encuentras incompatibilidades después de una instalación parcial, reinstala todo desde el archivo original:
> ```bash
> pip install -r requirements.txt --force-reinstall
> ```

---

## Tips de Optimización

### Para entrenamientos más rápidos

1. **Usar GPU** — es el cambio más impactante:
```bash
   python train.py -d gpu
```
2. **Aumentar entornos paralelos** si tienes VRAM suficiente (más de 6 GB):
```bash
   python train.py -n 400 -d gpu
```
3. **Reducir iteraciones** para prototipado rápido (sin esperar convergencia completa):
```bash
   python train.py --max_iterations 500
```

### Para mejores resultados finales

1. **Aumentar las iteraciones** para que el currículo complete todas sus fases:
```bash
   python train.py --max_iterations 3000
```
2. **Reducir la tasa de aprendizaje** en `train.py` si el entrenamiento es inestable (prueba con `0.0001`).
3. **Monitorear TensorBoard activamente** durante el entrenamiento para detectar problemas temprano.

---

## Documentación de Archivos

| Archivo | Descripción |
|---------|-------------|
| `train.py` | Script principal de entrenamiento. Define la arquitectura neuronal, los hiperparámetros de PPO y gestiona la generación de checkpoints, videos y logs. Aquí se configura el dispositivo (`cpu`/`gpu`) y el número de entornos. |
| `eval.py` | Carga un modelo entrenado desde `logs/` y lo ejecuta en simulación para evaluar su comportamiento. |
| `environment.py` | Define la tarea, el espacio de estados y acciones, el currículo de 4 fases y las funciones de recompensa. |
| `confi_cad.ipynb` | Notebook para inspeccionar y configurar el modelo CAD antes de iniciar el entrenamiento. |

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
> El proyecto no cuenta con una licencia oficial por el momento.

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