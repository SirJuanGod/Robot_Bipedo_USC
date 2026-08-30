---
name: LaTeXReportGen
description: "Convierte ecuaciones de control, parámetros y arquitecturas de RL en código LaTeX. Usa cuando necesites transformar dinámicas de control, funciones de recompensa, hiperparámetros o diagramas de arquitectura en bloques LaTeX limpios y listos para informes técnicos o publicaciones IEEE."
argument-hint: "Genera ecuaciones LaTeX para control, recompensa y arquitectura del sistema"
user-invocable: true
---

# LaTeXReportGen

## Cuándo usar esta skill

- Necesitas convertir ecuaciones de control, estados o dinámicas en formato LaTeX limpio.
- Quieres documentar funciones de costo, recompensa o arquitectura de red en un paper o reporte técnico.
- Debes preparar tablas de hiperparámetros o resúmenes de configuración para IEEE, academia o ingeniería.
- Necesitas producir bloques LaTeX formateados y legibles sin errores de sintaxis.

## Objetivo

Transformar información técnica de robótica y aprendizaje por refuerzo en bloques LaTeX bien estructurados, consistentes y listos para incluir en memoria técnica, informes técnicos o publicaciones.

## Procedimiento

1. Identifica el tipo de contenido a convertir.
   - Ecuaciones de dinámica o control.
   - Funciones de costo, recompensa o penalización.
   - Tablas de hiperparámetros, arquitectura o métricas.
   - Diagramas y listas de componentes del sistema.
2. Selecciona el entorno matemático correcto.
   - Usa `equation`, `align`, `gather`, `cases` o `array` según el caso.
   - Mantén notación consistente con el dominio: $x$, $u$, $q$, $\theta$, $\pi$, $r$, etc.
3. Estructura la ecuación para legibilidad.
   - Ordena términos por relevancia o por flujo físico.
   - Mantén alineación clara y evita marcas visuales innecesarias.
   - Si hay múltiples componentes, separa por líneas con `&` para mejor lectura.
4. Traduce conceptos de ingeniería a notación formal.
   - Convierte explicaciones cualitativas en variables y ecuaciones formales.
   - Asegúrate de que cada símbolo tenga sentido en el contexto de control, RL o mecánica.
5. Genera tablas y resúmenes con formato IEEE.
   - Usa `tabular` o `array` con columnas compactas y legibles.
   - Ajusta alineación, unidades y valores numéricos para máxima claridad.
6. Valida la sintaxis LaTeX.
   - Revisa que no haya comillas mal cerradas, delimitadores faltantes, comandos de subíndice o superíndice mal usados.
   - Confirma que las ecuaciones se compilan bien en el contexto del documento.
7. Ajusta el nivel de detalle según el público.
   - Informe interno: más descriptivo y explícito.
   - Publicación: más compacto, formal y estético.

## Reglas estrictas

- No generes LaTeX ambiguo o visualmente desordenado.
- Mantén terminología técnica coherente en todo el documento.
- Evita usar símbolos que no estén claramente definidos.
- Para ecuaciones complejas, prioriza claridad sobre una notación excesivamente compacta.
- En una tabla, usa unidades y valores compatibles con el formato de la publicación.

## Patrones recomendados

```latex
\begin{equation}
\dot{x} = A x + B u, \quad y = C x + D u
\end{equation}
```

```latex
\begin{align}
J(\theta) &= \sum_{t=0}^{T} \gamma^t r_t \
&\quad - \lambda \|\theta\|_2^2
\end{align}
```

```latex
\begin{table}[h]
\centering
\caption{Hiperparámetros del entrenamiento.}
\begin{tabular}{l c c}
\hline
Parámetro & Valor & Descripción \\
\hline
\textit{lr} & 3e-4 & Tasa de aprendizaje \\
\textit{batch size} & 2048 & Tamaño de lote \\
\textit{gamma} & 0.99 & Factor de descuento \\
\hline
\end{tabular}
\end{table}
```

## Tipos de contenido que puede convertir

- Ecuaciones de dinámica del robot o del sistema
- Funciones de costo, recompensa y penalización
- Arquitecturas de política y red neuronal
- Tablas de hiperparámetros o resultados
- Resúmenes de métodos y experimentación

## Criterios de calidad

La salida es correcta si:

- La ecuación o tabla se entiende sin necesidad de interpretar el código.
- La notación es consistente y técnicamente apropiada.
- El bloque LaTeX está bien formateado y listo para compilar.
- El contenido refleja con precisión el sistema o método descrito.

## Formato de salida esperado

Cuando el usuario pida convertir contenido técnico a LaTeX, responde con:

- bloque(s) LaTeX formateados
- explicación breve de la elección de entorno (`equation`, `align`, `table`, etc.)
- si aplica, versión compacta y versión más detallada
- advertencias de notación o compatibilidad con IEEE si corresponde

## Criterio de finalización

La tarea se considera completa cuando:

- El contenido técnico ha sido transformado en LaTeX válido.
- La estructura visual es clara y profesional.
- La notación es consistente con la ingeniería o la publicación.
- La salida queda lista para incluirse en un informe o paper.
