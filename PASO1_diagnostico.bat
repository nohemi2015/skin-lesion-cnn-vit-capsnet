@echo off
REM PASO 1: verificar el ViT original + diagnostico de CapsNet-R3 (G1-G5) + rejilla G6 + seleccion
REM Usa solo entrenamiento y validacion. Al terminar, manda la carpeta diagnostico_capsnet
REM y el archivo verificacion_vit.txt.
call CONFIGURAR.bat
python verificar_vit_original.py > verificacion_vit.txt 2>&1 || goto :error
type verificacion_vit.txt
python capsnet_diagnostico.py --manifest_csv %MANIFEST% || goto :error
for %%L in (1e-4 3e-5 1e-5) do (
  for %%N in (none imagenet) do (
    python entrenar_v3.py --manifest_csv %MANIFEST% --pipeline capsnet_r3_revised --capsnet_lr %%L --capsnet_input_norm %%N --split_seed 2026 --seeds 2027 --no_test --tag capsnet_grid_lr%%L_%%N || goto :error
  )
)
python seleccionar_capsnet.py || goto :error
echo.
echo PASO 1 TERMINADO. Manda la carpeta diagnostico_capsnet y verificacion_vit.txt
goto :eof
:error
echo.
echo *** Hubo un error. Copia el mensaje de arriba y mandalo. ***
exit /b 1
