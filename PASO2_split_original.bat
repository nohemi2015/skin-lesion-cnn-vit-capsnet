@echo off
REM PASO 2: nuevos pipelines en el split original (2026), 5 semillas cada uno.
REM Si se interrumpe, vuelve a ejecutarlo: las semillas terminadas se saltan.
call CONFIGURAR.bat
set CHECK=--check_ledger resultados_split_final_v2\split_ledger.csv
for %%P in (vit32_frozen_ckpt vit32_ft_ckpt vit32_ft_imagenet) do (
  python entrenar_v3.py --manifest_csv %MANIFEST% --pipeline %%P --split_seed 2026 --seeds 2027,2028,2029,2030,2031 %CHECK% || goto :error
)
python entrenar_v3.py --manifest_csv %MANIFEST% --pipeline capsnet_r3_revised --capsnet_config diagnostico_capsnet\capsnet_seleccion.json --split_seed 2026 --seeds 2027,2028,2029,2030,2031 %CHECK% || goto :error
echo.
echo PASO 2 TERMINADO. Manda la carpeta resultados_v3\split_2026
goto :eof
:error
echo *** Hubo un error. Copia el mensaje de arriba y mandalo. ***
exit /b 1
