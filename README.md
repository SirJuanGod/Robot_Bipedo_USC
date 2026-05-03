# Robot Bípedo USC - Entrenamiento con Aprendizaje por Refuerzo

Proyecto de investigación para el desarrollo y entrenamiento de un robot bípedo humanoide utilizando aprendizaje por refuerzo (RL) con algoritmo PPO (Proximal Policy Optimization). Este proyecto utiliza el simulador Genesis y la librería rsl-rl-lib para entrenar políticas de control locomotor.

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

- **Algoritmo PPO**: Implementación de Proximal Policy Optimization para entrenamiento estable
- **Entrenamiento Distribuido**: Soporte para múltiples entornos paralelos (200 por defecto)
- **Aprendizaje Curricular**: 4 fases de dificultad progresiva para facilitar el aprendizaje
- **Simulación Física**: Utiliza Genesis, un simulador física de alto rendimiento
- **GPU/CPU Flexible**: Soporta entrenamiento tanto en GPU como en CPU
- **Registración de Videos**: Captura automática de videos del entrenamiento
- **Monitoreo con TensorBoard**: Seguimiento en tiempo real de métricas de entrenamiento

---

## Requisitos del Sistema

### Hardware Mínimo
- **CPU**: Intel i7 / AMD Ryzen 5 o superior
- **RAM**: 16 GB (recomendado 32 GB)
- **GPU**: NVIDIA GPU con CUDA (recomendado para entrenamientos rápidos)
  - Si no tienes GPU, puede entrenar en CPU (más lento)

### Software Requerido
- **Python**: 3.8 o superior (se recomienda 3.10 o 3.11)
- **Conda** o **Pip**: Gestor de paquetes Python
- **CUDA**: 11.8+ (opcional pero recomendado para GPU)

---

## Instalación

### Paso 1: Clonar o Descargar el Proyecto

```bash
cd d:\USC\Proyectos\BPD_RBT_USC\Robot_Bipedo_USC
```

### Paso 2: Crear Ambiente Virtual

#### Opción A: Usar Conda (Recomendado)

```bash
# Crear ambiente
conda create -n robot_bpd python=3.11

# Activar ambiente
conda activate robot_bpd
```

#### Opción B: Usar Virtualenv

```bash
# Crear ambiente
python -m venv robot_bpd

# Activar ambiente (Windows)
robot_bpd\Scripts\activate

# Activar ambiente (Linux/Mac)
source robot_bpd/bin/activate
```

### Paso 3: Instalar Dependencias

>[!WARNING]
> Lee el archivo `requirements.txt` antes de instalar. Algunos paquetes pueden tener incompatibilidades de versión.

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar Instalación

```bash
python -c "import genesis; import torch; import rsl_rl; print('Todas las dependencias instaladas')"
```

>[!NOTE]
> Si la instalación fue exitosa, deberías ver el mensaje de confirmación. Si hay errores, revisa la sección de [Solución de Problemas](#solución-de-problemas).

---

## Configuración del Ambiente

### Archivo `environment.py`

Este archivo define el entorno de simulación del robot bípedo:

- **Posición inicial**: [0.0, 0.0, 0.228501] metros
- **Velocidad máxima**: 0.5 m/s
- **Currículo de aprendizaje**: 4 fases progresivas
- **Observaciones**: 25 dimensiones (ángulos articulares, velocidades, comandos)
- **Acciones**: Control de posición de articulaciones

### Archivo `config.json`

Contiene la configuración del modelo CAD:

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

>[!ALERT]
> No modifiques estos valores sin entender sus efectos en la dinámica del robot. Cambios inadecuados pueden impedir que el robot converja durante el entrenamiento.

---

## Uso

### Entrenar el Modelo

#### Opción 1: Entrenamiento por Defecto

```bash
python train.py
```

Esto iniciará el entrenamiento con:
- 200 entornos paralelos
- 1500 iteraciones máximas
- Dispositivo: CPU (o GPU si está disponible)
- Nombre del experimento: "bipedo-usc-ppo-v1"

#### Opción 2: Personalizar Parámetros

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

| Parámetro | Corto | Tipo | Por defecto | Descripción |
|-----------|-------|------|-------------|-------------|
| `--num_envs` | `-n` | int | 200 | Número de entornos paralelos |
| `--max_iterations` | - | int | 1500 | Máximo de iteraciones de entrenamiento |
| `--device` | `-d` | str | "cpu" | Dispositivo ("cpu" o "gpu") |
| `--exp_name` | `-e` | str | "bipedo-usc-ppo-v1" | Nombre del experimento |

>[!IMPORTANT]
> El entrenamiento puede durar varios minutos a horas dependiendo del hardware. En GPU: ~1-2 horas. En CPU: ~6-8 horas.

### Evaluar el Modelo

Para evaluar un modelo entrenado:

```bash
# Usar modelo de la última iteración
python eval.py

# Especificar dispositivo
python eval.py -d gpu

# Especificar nombre del experimento
python eval.py -e "bipedo-usc-ppo-v1"

# Usar CPU
python eval.py -d cpu
```

>[!NOTE]
> Si no existen modelos entrenados, la evaluación fallará. Entrena primero con `python train.py`.

### Visualizar Resultados

#### Con TensorBoard

```bash
# Activar el ambiente
conda activate robot_bpd

# Iniciar TensorBoard
tensorboard --logdir=./logs
```
>[!TIP]
> **Métricas disponibles para monitorear:**
> - Recompensa episódica
> - Pérdida del actor y crítico
> - Tasa de entropía
> - Velocidad del robot
> - Pasos de entrenamiento

---

## Estructura del Proyecto

```
Robot_Bipedo_USC/
├── train.py                      # Script principal de entrenamiento
├── eval.py                       # Script de evaluación
├── environment.py                # Definición del entorno
├── requirements.txt              # Dependencias del proyecto
├── confi_cad.ipynb              # Notebook de configuración CAD
├── README.md                     # Este archivo
│
├── model/                        # Configuración del robot
│   ├── Bipedo.xml               # Archivo de definición del robot
│   ├── config.json              # Configuración de parámetros
│   ├── scene.xml                # Escena de simulación
│   └── assets/                  # Modelos 3D del robot
│       ├── ank_d.part           # Tobillo derecho
│       ├── ank_i.part           # Tobillo izquierdo
│       ├── kne_d.part           # Rodilla derecha
│       ├── kne_i.part           # Rodilla izquierda
│       └── ... (más partes)
│
└── logs/                         # Resultados del entrenamiento
    └── bipedo-usc-ppo-v1/
        ├── model_0.pt           # Checkpoints del modelo
        ├── events.out.tfevents  # Datos de TensorBoard
        ├── videos/              # Videos de entrenamiento
        └── cfgs.pkl             # Configuración guardada
```

---

## Configuración de Parámetros

### Parámetros de PPO en `train.py`

```python
"clip_param": 0.2                    # Clip del ratio de política
"learning_rate": 0.0003              # Tasa de aprendizaje
"gamma": 0.99                        # Factor de descuento
"entropy_coef": 0.008                # Coeficiente de entropía
"max_iterations": 1500               # Máximo de iteraciones
"num_envs": 200                      # Número de entornos paralelos
```

### Recompensas del Currículo

El aprendizaje es progresivo con 4 fases:

**Fase 1**: Mantener equilibrio básico
**Fase 2**: Aumentar velocidad de caminata
**Fase 3**: Mejorar estabilidad y eficiencia
**Fase 4**: Optimizar movimiento natural

---
Requisitos Críticos

>[!ALERT]
> **1. Versión de rsl-rl-lib**: Debe ser >= 2.2.4
> ```bash
> pip show rsl-rl-lib
> ```

>[!ALERT]
> **2. Dimensión de observación**: Exactamente 25 dimensiones en el observador. No modificar sin ajustar la red neuronal.

>[!ALERT]
> **3. Memoria GPU**: Requiere al menos 4GB para 200 entornos. Reducir `--num_envs` si hay problemas de memoria.

### Información Importante

>[!NOTE]
> - **Convergencia**: El modelo comienza a converger alrededor de la iteración 500-800
> - **Guardar Modelos**: Se guardan cada 100 iteraciones en `logs/`
> - **Videos**: Se generan cada 2 episodios para seguimiento visual
> - **Tiempo de Entrenamiento**: ~1-2 horas en GPU, ~6-8 horas en CPU para 1500 iteraciones

### Errores Comunes

>[!WARNING]
> **Error: "Please install 'rsl-rl-lib>=2.2.4'"**
> ```bash
> pip install --upgrade rsl-rl-lib
> ```

>[!WARNING]
> **Error: "No module named 'genesis'"**
> ```bash
> pip install genesis-world
> ```

>[!WARNING]
> **Error: CUDA out of memory**
> ```bash
> python train.py -n 100 -d gpu
> ```

>[!WARNING]
> **Error: "ModuleNotFoundError: No module named 'rsl_rl'"**
> ```bash
> pip install -r requirements.txt --force-reinstall
> ```bash
   # Reinstalar dependencias
   pip install -r requirements.txt --force-reinstall
   ```

---

## Solución de Problemas

### Problema: El entrenamiento es muy lento

**Causa**: Probablemente estés usando CPU
**Solución**:
```bash
# Verificar disponibilidad de GPU
python -c "import torch; print(torch.cuda.is_available())"

# Si devuelve True, entrena con GPU
python train.py -d gpu
```

### Problema: Error de memoria insuficiente

**Causa**: Demasiados entornos paralelos para tu hardware
**Soluciones**:
```bash
# Reducir número de entornos
python train.py -n 64 -d gpu

# O usar CPU
python train.py -n 100 -d cpu
```

### Problema: El modelo no converge

**Causas posibles**:
- Tasa de aprendizaje muy alta
- Observaciones con distribución incorrecta
- Red neuronal inadecuada

**Solución**: Revisar logs en TensorBoard y ajustar hiperparámetros en `train.py`

### Problema: Evaluación sin modelos entrenados

**Error**: "Warning: No model files found"
**Solución**:
```bash
# Primero entrena un modelo
python train.py

# Luego evalúa
python eval.py
```

---

## Variables de Entorno

Puedes configurar variables de entorno para personalizar el comportamiento:

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

| Paquete | Versión | Propósito |
|---------|---------|----------|
| genesis-world | - | Simulador físico |
| torch | >= 2.0 | Framework de aprendizaje profundo |
| rsl-rl-lib | >= 2.2.4 | Algoritmo PPO |
| numpy | - | Operaciones numéricas |
| tensorboard | - | Visualización de métricas |

>[!WARNING]
> Algunos paquetes pueden tener conflictos de versión. Si ocurren problemas, usa el archivo `requirements.txt` original.

---

## Tips de Optimización

### Para Entrenamientos Más Rápidos

1. Aumentar número de entornos (si hay memoria disponible)
   ```bash
   python train.py -n 400 -d gpu
   ```

2. Usar GPU en lugar de CPU
   ```bash
   python train.py -d gpu
   ```

3. Reducir iteraciones si solo necesitas prototipado
   ```bash
   python train.py --max_iterations 500
   ```

### Para Mejores Resultados

1. Aumentar iteraciones de entrenamiento
   ```bash
   python train.py --max_iterations 3000
   ```

2. Reducir tasa de aprendizaje en `train.py` para mayor estabilidad

3. Monitoreizar en TensorBoard durante el entrenamiento

---

## Documentación de Archivos

### `train.py`
- **Función principal**: Maneja el entrenamiento con PPO
- **Salidas**: Modelos, videos y datos de TensorBoard en `logs/`
- **Configuración**: Define arquitectura de redes neuronales y parámetros de PPO

### `eval.py`
- **Función principal**: Carga y ejecuta un modelo entrenado
- **Entrada**: Modelos de `logs/`
- **Salida**: Comportamiento del robot en simulación

### `environment.py`
- **Función principal**: Define la tarea y el espacio de estados/acciones
- **Características**: Currículo de 4 fases, recompensas personalizadas

### `confi_cad.ipynb`
- **Propósito**: Notebook para visualizar y configurar el modelo CAD
- **Uso**: Exploración del modelo antes del entrenamiento

---

## Referencias

- **Genesis Simulator**: [Link a documentación]
- **RSL-RL Library**: [Link a documentación]
- **PPO Algorithm**: Schulman et al., 2017
- **OnShape CAD**: https://cad.onshape.com/

---

## Autor y Contacto

**Proyecto**: Robot Bípedo USC  
**Institución**: Universidad Santiago de Cali  
**Año**: 2024-2026

Para problemas o preguntas, revisa:
1. Este README
2. Los logs en `./logs/bipedo-usc-ppo-v1/`
3. La salida en terminal durante la ejecución

---

## Licencia

[Especificar licencia si aplica]

---

## Cambios Recientes

- **v1.0**: Versión inicial del proyecto
- **v1.1**: Adición de soporte para GPU
- **v1.2**: Implementación de currículo de aprendizaje

---

>[!NOTE]
> Última actualización: Mayo 2026  
> Estado: En desarrollo activo
