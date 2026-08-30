---
name: EnvironmentManager
description: "Gestiona la configuración del entorno de desarrollo, Pixi, WSL y dependencias de Python. Usa cuando necesites diagnosticar conflictos de paquetes, crear archivos de entorno, preparar virtualenvs o resolver problemas con PyTorch, Stable-Baselines3, CUDA y rutas de ejecución."
argument-hint: "Configura el entorno de Python/Pixi para RL con PyTorch, CUDA y WSL"
user-invocable: true
---

# EnvironmentManager

## Cuándo usar esta skill

- Necesitas preparar o corregir un entorno de desarrollo para Python, RL o simulación física.
- Estás trabajando con Pixi, venv, conda o WSL y quieres dejarlo reproducible.
- Hay conflictos entre dependencias como PyTorch, Stable-Baselines3, CUDA y bibliotecas de robótica.
- Debes resolver errores de importación, versiones incompatibles o rutas de ejecutables.

## Objetivo

Generar y revisar configuraciones de entorno que permitan ejecutar proyectos de RL, robótica y simulación de forma reproducible y con mínima fricción entre sistemas operativos, dependencias y herramientas de desarrollo.

## Procedimiento

1. Identifica el tipo de entorno necesario.
   - Determina si se usa Python global, venv, conda, Pixi o WSL.
   - Revisa si el proyecto requiere GPU, CUDA o dependencias específicas de sistema.
2. Define el conjunto mínimo de dependencias.
   - Prioriza librerías clave para RL y simulación: PyTorch, Stable-Baselines3, numpy, gymnasium, pandas, matplotlib, etc.
   - Verifica compatibilidad entre versiones, especialmente entre PyTorch, CUDA y paquetes de control.
3. Diagnostica conflictos reales.
   - Busca errores de importación, módulos faltantes, incompatibilidades de versión, `pip` vs `conda`, rutas de ejecutable o problemas de WSL.
   - Diferencia un problema de entorno de un problema de código.
4. Genera la configuración correcta.
   - Crea o modifica `pixi.toml`, `requirements.txt`, scripts de PowerShell o comandos reproducibles para WSL.
   - Asegúrate de separar entorno de desarrollo, entrenamiento y evaluación cuando sea necesario.
5. Resuelve rutas y ejecutables.
   - Verifica que `python`, `pip`, `pytest`, `torch`, `sb3` y cualquier ejecutable del proyecto apunten al entorno correcto.
   - Corrige problemas de PATH, instalación en WSL o rutas del sistema Windows/Linux.
6. Revisa la reproducibilidad.
   - Documenta versiones clave y comandos exactos para reinstalar o replicar el entorno.
   - Si conviene, incluye pasos para verificar la instalación con un comando mínimo de smoke test.
7. Realiza validación básica.
   - Comprueba que el entorno importa correctamente las librerías clave.
   - Verifica que la GPU/CUDA se detecta o que al menos no hay silencios de incompatibilidad.

## Reglas estrictas

- No asumas que una instalación funciona solo porque se ejecutó alguna vez; valida importaciones y versiones reales.
- Prioriza compatibilidad sobre “última versión disponible” si hay conflictos de dependencia.
- Separa claramente el entorno de desarrollo de la ejecución del proyecto; no mezcles scripts de configuración con lógica de aplicación.
- Si el problema es específico de WSL o Windows, documenta la ruta y el origen del conflicto antes de proponer una solución.
- Evita sugerir cambios de entorno demasiado complejos si un entorno virtual simple o Pixi básico resuelve el problema.

## Patrones recomendados

```toml
[project]
name = "robot-bipedo"
channels = ["conda-forge"]

[dependencies]
python = "3.11"
pytorch = ">=2.2"
stable-baselines3 = ">=2.3"
numpy = ">=1.26"
```

## Problemas típicos a diagnosticar

- `ModuleNotFoundError` para `torch`, `stable_baselines3`, `gymnasium` o librerías del proyecto.
- Incompatibilidad entre CUDA y PyTorch.
- `pip install` en un entorno equivocado.
- Conflictos de versión entre paquetes de compatibilidad.
- Scripts de PowerShell o WSL que ejecutan un intérprete incorrecto.
- Rutas de ejecutables o `PATH` mal configuradas.

## Criterios de calidad

La configuración es correcta si:

- El entorno se puede reproducir de manera clara.
- Las dependencias clave importan correctamente.
- Las versiones relevantes quedan definidas o documentadas.
- Las rutas y ejecutables apuntan al entorno correcto.
- La solución resuelve la causa real del conflicto y no solo el síntoma.

## Formato de salida esperado

Cuando el usuario pida configuración del entorno, responde con:

- diagnóstico del problema de entorno
- recomendaciones de dependencias o versions
- archivos de configuración sugeridos (`pixi.toml`, `requirements`, scripts)
- pasos de verificación mínimos para confirmar funcionamiento
- advertencias de WSL, CUDA o compatibilidad de hardware

## Criterio de finalización

La tarea se considera completa cuando:

- El entorno está descrito de forma reproducible.
- Las dependencias clave y rutas se han revisado.
- Se vuelve claro cómo reinstalar o validar la configuración.
- El problema de instalación, conflicto o ejecución queda resuelto en la raíz.
