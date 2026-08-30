---
name: SimToRealInspector
description: "Inspecciona y valida modelos URDF y MJCF para simulaciones físicas en Genesis. Usa cuando necesites revisar límites articulares, matrices de inercia, cadenas cinemáticas, jerarquías de joints y errores comunes en geometrías o ejes de rotación."
argument-hint: "Revisa el modelo robótico X y valida URDF/MJCF para Genesis"
user-invocable: true
---

# RobotModelInspector

## Cuándo usar esta skill

- Necesitas revisar un modelo robótico en URDF o MJCF antes de entrenar o simular.
- Quieres validar que la cinemática, la jerarquía y los parámetros físicos sean coherentes.
- Estás importando un robot a Genesis y sospechas que hay problemas de inercia, articulaciones o colisiones.
- Debes detectar errores comunes antes de ejecutar simulaciones largas o políticas de RL.

## Objetivo

Analizar archivos de descripción robótica para identificar fallas estructurales y físicas que puedan afectar a la simulación, control o transferencia a hardware. La inspección debe centrarse en la validez del modelo, no solo en la apariencia visual.

## Procedimiento

1. Identifica el tipo de modelo y su estructura base.
   - Determina si es URDF o MJCF.
   - Revisa la jerarquía de links y joints, la raíz del robot y la relación entre cuerpos.
2. Verifica la cadena cinemática.
   - Comprueba que cada joint tenga un parent/child correcto.
   - Revisa que no existan enlaces desconectados, ciclos cinemáticos o ramas sin sentido.
   - Confirma que la secuencia del árbol sea física y consistente con la topología del robot.
3. Revisa los parámetros articulares.
   - Comprueba límites `lower`/`upper`, `damping`, `friction`, `armature` o equivalentes.
   - Asegúrate de que el rango de movimiento sea realista y que no haya valores nulos o inconsistentes.
   - Verifica que los ejes de rotación o translación correspondan a la dirección esperada.
4. Evalúa inercia y masas.
   - Revisa que cada link tenga masa positiva y no nula.
   - Validar matrices de inercia y su simetría.
   - Detecta valores muy pequeños, grandes, negativos o sin sentido físico.
5. Inspecciona geometría y colisiones.
   - Comprueba que las meshes o collision geometries estén bien alineadas con el origen del link.
   - Identifica solapamientos, offsets desbalanceados, escalas incorrectas o cuerpos que no coinciden con la estructura física.
   - Revisa si las colisiones son demasiado agresivas o demasiado débiles para la simulación.
6. Busca errores típicos de importación.
   - Ejes de rotación invertidos.
   - Inercias nulas o mal definidas.
   - Joints que no respetan la convención del modelo.
   - Geometrías de colisión mal posicionadas.
   - Unidades inconsistente entre masas, longitudes y tiempos.
7. Señala riesgos de simulación.
   - Determina qué problemas pueden producir inestabilidad, jitter, rigidez artificial o comportamiento no realista.
   - Prioriza los errores que harán fallar la simulación, antes que los que solo afectan al render.
8. Propón correcciones específicas.
   - Sugiere ajustes concretos a inercia, joint limits, ejes, origenes de enlace, mesh y colisiones.
   - Si aplica, indica cómo validarlo en Genesis con un paso mínimo de prueba.

## Reglas estrictas

- No inventes parámetros ni asumas que un modelo es correcto solo porque “parece bien”.
- Prioriza fallos que afecten al comportamiento físico real de la simulación.
- Si un modelo presenta inconsistencia geométrica o cinemática, indícala claramente antes de sugerir optimizaciones.
- No confundas visualización con física: una mesh bonita no asegura una simulación estable.
- En modelos URDF/MJCF, verifica origenes, ejes y unidades como parte central del análisis.

## Patrón recomendado

```xml
<!-- Revisión de joints y ejes -->
<joint name="hip_pitch" type="revolute">
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <parent link="pelvis"/>
  <child link="thigh"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="10" velocity="2.0"/>
</joint>
```

## Errores comunes que debe detectar

- `mass` igual a cero o demasiado pequeña.
- `inertia` no simétrica o con valores absurdos.
- `axis` apuntando en dirección incorrecta.
- `origin` con offsets no coherentes con la geometría.
- `collision` fuera de lugar respecto al link visual.
- joint con limits fuera de rango o valores de velocidad/torque no realistas.
- enlaces sin parent/child adecuados.
- árbol cinemático incompleto o con ciclos.

## Criterios de calidad

La inspección es correcta si:

- Verifica la estructura jerárquica del robot.
- Revisa físicos y cinemática relevantes para simulación.
- Detecta errores comunes con explicación técnica.
- Sugerir correcciones específicas y accionables.
- Prioriza problemas reales que afecten estabilidad o rendimiento del modelo.

## Formato de salida esperado

Cuando el usuario pida inspeccionar un modelo, responde con:

- resumen general del robot y su topología
- lista de validaciones por categoría (cinemática, inercia, articulaciones, geometría)
- hallazgos críticos con severidad
- recomendaciones concretas de corrección
- advertencias sobre riesgos en Genesis

## Criterio de finalización

La tarea se considera completa cuando:

- El modelo ha sido revisado en sus componentes físicos y cinemáticos esenciales.
- Se han identificado errores o riesgos reales con justificación técnica.
- Se presentan correcciones específicas y plausibles.
- La salida es útil para corregir el modelo antes de entrenamiento o simulación.
