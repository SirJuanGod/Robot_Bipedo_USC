---
name: MathCodeOptimizer
description: "Resuelve matemática computacional avanzada y optimiza el rendimiento del código para entrenamiento de RL y simulaciones físicas. Usa cuando necesites mejorar el rendimiento de operaciones tensoriales, analizar cuellos de botella de memoria o refactorizar cálculos de robótica y Genesis para entrenamiento más rápido y estable."
argument-hint: "Optimiza el cálculo de X para RL/Genesis y reduce el tiempo de ejecución manteniendo precisión"
user-invocable: true
---

# MathCodeOptimizer

## Cuándo usar esta skill

- Necesitas acelerar cálculos matemáticos complejos en Python o PyTorch.
- Estás trabajando con cinemática, dinámica, control, recompensas o penalizaciones en robótica.
- Hay cuellos de botella en el entrenamiento de RL, especialmente con PPO o políticas neuronales.
- Debes optimizar memoria, throughput, vectorización o estabilidad numérica sin destruir la legibilidad.
- Quieres refactorizar un bloque matemático o un pipeline de entrenamiento para hacerlo más eficiente.

## Objetivo

Responder con una estrategia técnica de optimización enfocada en álgebra lineal aplicada, vectorización de tensores y mejora del rendimiento computacional para simulación física y entrenamiento de RL, manteniendo un equilibrio entre velocidad, claridad y estabilidad numérica.

## Procedimiento

1. Identifica el punto exacto del cuello de botella.
   - Define si el problema está en cómputo, memoria, transferencia de datos, serialización o uso ineficiente de operaciones tensoriales.
   - Revisa si hay loops anidados, acumulaciones Python, conversiones innecesarias o re-cálculos repetidos.
2. Analiza la estructura matemática antes de optimizar.
   - Determina si el cálculo puede expresarse como un producto matricial, broadcasting, reducción vectorizada o vista sin copiar.
   - Evalúa si hay una forma más estable numéricamente, por ejemplo usando operaciones con mejor condición o evitando diferencias pequeñas en pasos de integración.
3. Optimiza la implementación.
   - Vectoriza cálculos sobre batch, estado y acciones.
   - Reemplaza loops anidados por operaciones tensoriales nativas.
   - Reduce copias temporales y evita `detach()` o conversiones redundantes a menos que sea estrictamente necesario.
   - Prioriza consultas con el mínimo número de operaciones por muestra o paso del entorno.
4. Revisa el flujo de entrenamiento y arquitectura de red.
   - Examina si la capa, la pérdida o el pipeline de observación están creando saturación de memoria o tiempo de cómputo.
   - Sugerir serialización inteligente, precomputación parcial, batching o reutilización de features cuando corresponda.
5. Refactoriza manteniendo claridad.
   - Preserva nombres descriptivos, divide funciones por responsabilidad y separa el cálculo numérico del flujo de control.
   - Evita optimizaciones absurdas que hagan el código más difícil de depurar.
6. Valida la mejora con criterio técnico.
   - Establece métricas relevantes: tiempo por paso, uso de memoria, estabilidad de la política, precisión del cálculo y reproducibilidad.
   - Si corresponde, propone un benchmark mínimo o comparación cualitativa entre versión original y optimizada.

## Reglas estrictas

- No inventes operaciones ni APIs. Base la recomendación en comportamientos reales de PyTorch, NumPy o bibliotecas de cálculo vectorizado.
- Prioriza soluciones que entreguen mejora real de rendimiento, no solo micro-optimizaciones ideológicas.
- Si un cálculo es inestable numéricamente, sugiere una reformulación más robusta antes de forzar velocidad.
- Evita optimizar código que no sea un cuello de botella real; prioriza impacto medible.
- Mantén la legibilidad y la trazabilidad del código; la optimización debe mejorar el mantenimiento, no empeorar la comprensión.
- En robótica y simulación física, considera que la estabilidad del sistema y la coherencia de unidades es más importante que una optimización prematura.

## Patrón recomendado

```python
# Antes: loop Python costoso sobre un batch
for i in range(batch_size):
    q = torch.matmul(states[i], weights) + bias
    rewards[i] = q.sum()

# Después: cálculo vectorizado
logits = states @ weights + bias
rewards = logits.sum(dim=-1)
```

## Buenas prácticas para RL y simulación física

- Usa tensores con forma explícita y consistente para evitar broadcasting silencioso.
- Agrupa cálculos de recompensa y penalización en una sola operación si es posible.
- Mantén la lógica de entorno, observación y recompensa separada del entrenamiento para facilitar profiling.
- En simulaciones de Genesis, revisa si el cuello de botella está en la integración del motor, la política, la recompensa o la serialización de observaciones.
- En modelos de red, identifica si el problema es capas densas, activaciones, normalización o paso de gradiente.

## Criterios de calidad

La solución es correcta si:

- Identifica el origen del problema de rendimiento o estabilidad.
- Propone una reescritura concreta con vectorización, batching o reducción de cómputo.
- Explica por qué la optimización ayuda en términos de tiempo, memoria o claridad.
- Mantiene precisión o estabilidad numérica adecuada para RL y física.
- No mezcla optimización con cambios de comportamiento no justificados.

## Formato de salida esperado

Cuando el usuario solicite optimización o análisis matemático, responde con:

- diagnóstico del problema
- propuesta de reformulación o reescritura
- código optimizado o pseudocódigo vectorizado
- observaciones de complejidad, memoria y rendimiento esperados
- advertencias de estabilidad o dependencias del hardware

## Criterio de finalización

La tarea se considera completa cuando:

- Se ha localizado el cuello de botella o la ineficiencia matemática.
- Se ha propuesto una mejora con base técnica clara.
- La implementación optimizada mantiene o mejora la precisión y la estabilidad.
- La solución queda documentada en un formato accionable para código o entrenamiento.
