---
name: FetchGenesisDocs
description: "Busca en la web la documentación oficial de Genesis World y Genesis Forge. Usa cuando necesites consultar APIs, parámetros, ejemplos oficiales y referencias actualizadas sobre robótica bípeda, control dinámico y simulación física."
argument-hint: "Busca la documentación oficial de Genesis World/Forge para X"
user-invocable: true
---

# FetchGenesisDocs

## Cuándo usar esta skill

- Necesitas consultar la documentación oficial de Genesis World y Genesis Forge.
- Quieres confirmar la sintaxis actual de una API, parámetros, límites o ejemplos oficiales.
- Estás trabajando con robótica bípeda, control dinámico, locomoción o simulación física.
- Debes obtener información cruda y verificable para que otro agente o modelo la procese.

## Objetivo

Realizar una búsqueda web exhaustiva centrada exclusivamente en fuentes oficiales de Genesis World y Genesis Forge, extraer la información actualizada y devolverla en forma de evidencia utilizable para análisis posterior.

## Procedimiento

1. Localiza primero la documentación oficial de Genesis World y Genesis Forge.
2. Prioriza páginas oficiales, docs, tutoriales y ejemplos mantenidos por Genesis.
3. Busca específicamente referencias relacionadas con:
   - APIs relevantes
   - configuración del entorno
   - actuadores, físicas y control
   - robótica bípeda o locomoción
   - sim-to-real y entornos de entrenamiento
4. Extrae la sintaxis actualizada, parámetros, configuraciones y ejemplos oficiales.
5. Registra enlaces y secciones clave para mantener trazabilidad.
6. Devuelve la información en formato crudo, bien organizada y lista para ser procesada.

## Reglas estrictas

- Nunca uses fuentes no oficiales como base principal si existe documentación oficial disponible.
- No inventes APIs, nombres de parámetros ni comportamientos.
- Si la documentación no es clara, indica la ambigüedad y solicita verificación adicional.
- No conviertas la respuesta en un resumen final demasiado condensado; prioriza la información factual y cruda.
- Mantén separada la lectura de documentación de la inferencia conceptual.

## Formato de salida esperado

Devuelve un bloque con estas secciones:

### 1. Fuentes oficiales consultadas

- Lista de enlaces directos a la documentación relevante.

### 2. Sintaxis actualizada

- Snippets de API o ejemplos oficiales.
- Parámetros principales y su significado.

### 3. Ejemplos relevantes

- Casos de uso compatibles con robótica bípeda y control dinámico.

### 4. Observaciones clave

- Limitaciones, advertencias o requisitos importantes.

### 5. Información para procesamiento posterior

- Resumen técnico crudo que el agente pueda transformar en implementación o diagnóstico.

## Criterio de finalización

La tarea se considera completa cuando:

- Se han revisado fuentes oficiales de Genesis World y Genesis Forge.
- La respuesta incluye sintaxis actualizada y ejemplos oficiales.
- La información está suficientemente estructurada para su procesamiento posterior.
- No se presenta información no verificada o inventada.
