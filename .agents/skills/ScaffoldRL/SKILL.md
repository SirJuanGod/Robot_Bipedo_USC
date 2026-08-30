---
name: ScaffoldRL
description: "Genera la estructura de código para control de RL manteniendo métricas y entrenamiento separados. Usa cuando necesites crear un proyecto de aprendizaje por refuerzo en Genesis con políticas PPO u otros algoritmos, garantizando una arquitectura modular y limpia."
argument-hint: "Crea la estructura RL para un entorno con PPO, entrenamiento y métricas separadas"
user-invocable: true
---

# ScaffoldRL

## Cuándo usar esta skill

- Necesitas crear o reorganizar la estructura base de un proyecto de RL.
- Estás implementando un entorno de control con Genesis o un entorno similar.
- Quieres mantener una arquitectura limpia donde el entrenamiento y la lógica de métricas estén separados.
- Debes generar código para un agente o un entorno sin mezclar responsabilidades.

## Objetivo

Generar una estructura modular de Python para proyectos de RL en los que la política (por ejemplo PPO) vive en un script de entrenamiento separado y la actualización de métricas vive en otro script importado/invocando desde el entrenamiento.

## Procedimiento

1. Define el rol de cada archivo antes de escribir código.
2. Crea un script de entrenamiento principal para la política y el loop de actualización.
3. Crea un script separado para métricas, logging o dashboard de rendimiento.
4. Configura el script de entrenamiento para importar la lógica de métricas y llamarla explícitamente.
5. Mantén las clases de entrenamiento y de métricas desacopladas.
6. Diseña el entorno, observaciones, recompensas y terminaciones en un módulo separado si es necesario.
7. Verifica que no exista un bloque único que mezcle PPO, métricas y logging.

## Regla de arquitectura estricta

- Las clases de entrenamiento de la política y las clases de actualización de métricas deben crearse como scripts o módulos completamente separados.
- El script de entrenamiento debe invocar al script de métricas.
- Nunca devuelvas un único bloque unificado que combine lógica principal, métricas y reportes.
- Mantén la separación entre entrenamiento, entorno y observabilidad.

## Patrón recomendado

```python
# train_policy.py
from metrics import MetricsLogger

class PPOTrainer:
    def __init__(self):
        self.metrics = MetricsLogger()

    def train(self):
        for step in range(num_steps):
            # lógica principal del entrenamiento
            self.metrics.update(step, reward, loss)

# metrics.py
class MetricsLogger:
    def update(self, step, reward, loss):
        # logging, historial, TensorBoard, plot, etc.
        pass
```

## Criterios de calidad

La estructura es correcta si:

- Hay un módulo o script específico para el entrenamiento de la política.
- Hay un módulo o script específico para métricas.
- El entrenamiento importa e invoca el módulo de métricas sin integrar la lógica dentro del mismo archivo.
- El entorno y la política están separados de la lógica de observabilidad.
- Cada archivo tiene una responsabilidad única y clara.

## Formato de salida esperado

Cuando el usuario pida generar código, responde con:

- estructura de archivos sugerida
- código Python modular
- separación explícita entre entrenadores y métricas
- instrucciones para invocar el módulo de métricas desde el entrenamiento

## Criterio de finalización

La tarea se considera completa cuando:

- La solución genera una arquitectura modular de RL.
- El entrenamiento y las métricas están en archivos distintos.
- El flujo de importación/invocación entre ambos queda explícito.
- No existe un bloque único combinado en un solo archivo.
