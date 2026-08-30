---
name: AnalyzeSimToReal
description: "Verifica parámetros físicos en simulaciones de Genesis para evitar problemas al pasar al mundo real. Usa cuando necesites revisar fricción, inercias, contacto, calibración de pesos y discrepancias entre simulación y hardware físico."
argument-hint: "Analiza esta configuración sim-to-real para un robot bípedo en Genesis"
user-invocable: true
---

# AnalyzeSimToReal

## Cuándo usar esta skill

- Debes revisar si una simulación en Genesis será transferible a hardware real.
- Necesitas detectar discrepancias entre modelo simulado y robot físico.
- Estás ajustando contactos, fricción, masa, inercias o pesos de control.
- Quieres una auditoría rápida de la configuración antes de entrenar o desplegar.

## Objetivo

Identificar y clasificar los riesgos de transferencia sim-to-real en entornos físicos de Genesis, enfocándote en parámetros que afectan la dinámica del robot, la estabilidad del contacto y la alineación entre el comportamiento simulado y el comportamiento real.

## Procedimiento

1. Revisa la configuración del entorno y del robot simulado.
2. Inspecciona los parámetros del Contact Manager y la física del contacto.
3. Verifica la fricción de superficies, amortiguación, restitud y coeficientes de contacto.
4. Revisa masas, centros de masa e inercias de links y actuadores.
5. Comprueba la calibración de pesos del controlador y del sistema de comando.
6. Evalúa si los observables del entorno (IMU, contactos, posiciones, velocidades) son consistentes con lo disponible en hardware.
7. Señala discrepancias reales, riesgos de dominio y ajustes recomendados.

## Puntos de revisión

- Contact Manager: configuración de resolución, contacto, tolerancias y materiales.
- Fricción de suelo y materiales: si es demasiado baja o alta, puede generar locomoción artificial.
- Inercias y masas: desalineaciones pueden producir aceleraciones no realistas.
- Pesos del controlador: si están sobre-optimizados para la simulación, pueden no generalizar al hardware.
- Control del gait o referencias: comportamiento estable en simulación no garantiza viabilidad real.
- Sensorización: IMU, encoders, estados de contacto y observabilidad deben reflejar el sistema real.

## Regla de análisis

- No asumas que una configuración estable en simulación será válida en hardware.
- Busca diferencias estructurales entre el mundo simulado y la realidad física.
- Identifica qué parámetros pueden producir un efecto de overfitting a la simulación.
- Prioriza cambios con alto impacto realista: contacto, masas, fricción, observación y controlador.

## Formato de salida esperado

Devuelve una respuesta con estas secciones:

### 1. Configuración revisada

- Qué componentes del entorno y robot fueron analizados.

### 2. Riesgos sim-to-real detectados

- Lista de discrepancias y su impacto probable.

### 3. Parámetros críticos

- Contact Manager, fricción, inercias, pesos del controlador, calibración.

### 4. Ajustes recomendados

- Cambios específicos sugeridos para mejorar la transferencia a hardware.

### 5. Recomendación final

- Si el entorno es viable, requiere calibración o debe replantearse.

## Criterio de finalización

La revisión se considera completa cuando:

- Se han revisado los parámetros físicos y de control relevantes.
- Se han identificado los principales riesgos sim-to-real.
- Se proponen ajustes accionables.
- La conclusión refleja si la configuración es razonablemente transferible o necesita corrección.
