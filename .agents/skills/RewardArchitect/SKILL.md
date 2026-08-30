---
name: RewardArchitect
description: "Diseña y equilibra componentes de recompensa y penalización para RL bípedo. Usa cuando necesites formular recompensas para locomoción, regular estabilidad y simetría de paso, o ajustar pesos para evitar colapsos, marcha rígida y contacto excesivo."
argument-hint: "Diseña la recompensa para locomoción bípeda con seguimiento de velocidad, estabilidad y penalizaciones de impacto"
user-invocable: true
---

# RewardArchitect

## Cuándo usar esta skill

- Necesitas diseñar una función de recompensa para un robot bípedo en RL.
- Estás ajustando locomoción, estabilidad, velocidad de referencia o postura corporal.
- Quieres equilibrar recompensas y penalizaciones para evitar comportamientos indeseados.
- Debes crear una política que camine sin colapsar, oscilar excesivamente o generar impactos violentos.

## Objetivo

Diseñar un conjunto de componentes de recompensa y penalización que guíen la locomoción bípeda hacia un comportamiento estable, eficiente y reproducible, manteniendo un equilibrio entre progreso del objetivo y control de energía, contacto y postura.

## Procedimiento

1. Define el objetivo principal de locomoción.
   - Establece si la política debe seguir velocidad, ir hacia un objetivo, mantener altura corporal o sostener una postura específica.
   - Asegúrate de que el objetivo principal sea observable desde la información del entorno.
2. Formula recompensas de progreso.
   - Utiliza recompensas por seguimiento de velocidad en dirección deseada.
   - Si aplica, incorpora recompensa por desplazamiento global o por referencia de velocidad.
   - Mantén la señal de progreso clara y no ambigua.
3. Añade recompensas de postura y estabilidad.
   - Incluye mantenimiento de altura del torso o centro de masa.
   - Recompensa orientación estable en roll y pitch.
   - Define qué tan tolerable es la inclinación antes de que la política pierda estabilidad.
4. Incorpora simetría del paso y patrones de marcha.
   - Considera recompensas por simetría en posiciones y velocidades de piernas.
   - Evita que la política aprenda un movimiento rígido o asimétrico.
   - Si hay una referencia de paso, úsala como guía, pero no la conviertas en una restricción artificial que rompa adaptabilidad.
5. Penaliza contacto y energía ineficiente.
   - Aplica penalizaciones por impactos de contacto fuertes o repentinos.
   - Penaliza uso excesivo de torque, esfuerzo muscular o potencia de actuación.
   - Reduce la tendencia a “forzar” la locomoción mediante golpes o vibración.
6. Ajusta pesos con criterio de equilibrio.
   - Si el agente colapsa, reforzar la recompensa de supervivencia o altura del torso.
   - Si camina rígido, redistribuir peso hacia simetría, suavidad o referencias de velocidad.
   - Si el robot vibra o impacta, aumentar penalizaciones de contacto y torque.
7. Revisa los mínimos locales.
   - No dejes que una sola recompensa domine la política.
   - Verifica que no aparezcan soluciones triviales como “sentarse”, “inclinarse”, o “aplastar el pie” solo para maximizar una recompensa parcial.
8. Valida en términos de comportamiento real.
   - Contrasta la política con indicadores de estabilidad, altura, velocidad, energía y contacto.
   - Ajusta gradualmente los pesos en lugar de hacer cambios exagerados.

## Reglas estrictas

- La recompensa debe guiar al robot hacia un comportamiento físico plausible, no solo a maximizar una métrica aislada.
- No utilices una recompensa excesivamente grande para un objetivo único; el agente puede explotarla.
- Prioriza estabilidad y continuidad por encima de velocidad máxima si la locomoción es inestable.
- Las penalizaciones por impacto y torque deben ser proporcionales al daño físico o a la ineficiencia del movimiento.
- Si una recompensa se presta a explotación, reformúlala o combínala con otra más robusta.

## Estructura recomendada

```python
reward = (
    w_vel * vel_tracking
    + w_height * torso_height_reward
    + w_orientation * orientation_stability
    + w_sym * gait_symmetry
    - w_impact * contact_impact_penalty
    - w_torque * torque_penalty
    - w_slip * slip_penalty
)
```

## Componentes típicos para locomoción bípedo

- Seguimiento de velocidad de referencia
- Simetría de paso
- Mantenimiento de altura del torso
- Estabilidad de roll/pitch
- Penalización por contacto brusco
- Penalización por torque excesivo
- Penalización por deslizamiento o pérdida de tracción
- Penalización por energía innecesaria o esfuerzo ineficiente

## Criterios de calidad

La recompensa es correcta si:

- Guía el movimiento hacia una locomoción sensible y estable.
- Equilibra progreso, postura y eficiencia.
- Penaliza impactos, energía excesiva y comportamientos inestables.
- Evita mínimos locales como colapsos, marcha rígida o zancadas violentas.
- Es fácil ajustar pesos con impacto predictivo.

## Formato de salida esperado

Cuando el usuario pida diseño de recompensa, responde con:

- objetivo de locomoción
- componentes de recompensa con explicación
- penalizaciones relevantes
- pesos sugeridos o estrategia de ajuste
- diagnósticos de fallos comunes y cómo corregirlos

## Criterio de finalización

La tarea se considera completa cuando:

- Se ha definido una estructura de recompensa clara para la locomoción bípedo.
- Los componentes relevantes están equilibrados y no favorecen un comportamiento degenerado.
- Se han identificado los riesgos principales de explotación o colapso.
- La política resultante puede ser ajustada mediante pesos sin requerir una reformulación completa.
