# Experimentos de la revisión (camino A) — instrucciones

## Antes de empezar

1. Copia **todos** los archivos de esta carpeta a la carpeta donde hiciste los entrenamientos originales. Ahí deben estar también:
   - `modelos_reducidos.py` (el original, **no lo modifiques**)
   - `subset_grande_con_lesion_id.csv` (el manifest de las 2,000 imágenes)
   - la carpeta `resultados_split_final_v2\` (resultados originales)
   - la carpeta `modelos_split_final_v2\` (los modelos `.keras` originales)
2. Abre `CONFIGURAR.bat` con el Bloc de notas y pon tus rutas reales:
   - `MANIFEST`: el CSV de las 2,000 imágenes.
   - `METADATA`: el archivo de metadata de HAM10000. Si tu archivo no tiene extensión, escribe el nombre tal cual, sin `.csv`.
   - `HAM_DIRS`: la(s) carpeta(s) con **todas** las imágenes de HAM10000, separadas por coma.
3. Abre la terminal (cmd) en esa carpeta, con el mismo entorno de Python que usaste antes.

## Los 4 pasos

Ejecuta cada paso escribiendo su nombre, por ejemplo `PASO1_diagnostico.bat`. **Después de cada paso, mándame lo que indica al final**, antes de seguir con el siguiente.

| Paso | Qué hace | Duración aproximada |
|---|---|---|
| `PASO1_diagnostico.bat` | Verifica el ViT original; diagnóstico de CapsNet-R3 (G1–G5); rejilla de 6 configuraciones (solo validación); selección | Corta (CapsNet es rápido) |
| `PASO2_split_original.bat` | ViT con normalización oficial (congelado y fine-tuned), ViT fine-tuned con normalización ImageNet y CapsNet revisado, 5 semillas cada uno, en el split original | Media (15 entrenamientos de ViT) |
| `PASO3_splits_repetidos.bat` | 5 splits nuevos × 3 semillas × 5 pipelines | **Larga** (varios días) |
| `PASO4_confirmacion_y_analisis.bat` | Evalúa los modelos en las ~8,000 imágenes de HAM10000 que nunca se usaron, y hace todo el análisis estadístico | Corta (solo inferencia) |

- Si un paso se interrumpe (apagón, cierras la ventana), **vuelve a ejecutarlo**: lo que ya terminó se salta y continúa donde se quedó.
- Cada entrenamiento imprime cuántos minutos tardó (`train_minutes`). Con eso sabremos cuánto tardará el paso 3.
- Si aparece un error, el script se detiene y dice "Hubo un error". Mándame el mensaje.

## Reglas para no invalidar nada

- **No mires los resultados del test para decidir nada.** Las decisiones (por ejemplo, qué configuración de CapsNet usar) las toman los scripts con validación, según el protocolo (`PROTOCOLO_REVISION.md`).
- No cambies el código entre pasos sin consultarlo. Si hace falta un cambio, se anota en el protocolo con fecha y razón.
