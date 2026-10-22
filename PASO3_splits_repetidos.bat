@echo off
REM PASO 3: 5 splits nuevos (3001-3005) x 3 semillas x 5 pipelines. Es el paso mas largo.
REM Si se interrumpe, vuelve a ejecutarlo: lo terminado se salta.
call CONFIGURAR.bat
for %%S in (3001 3002 3003 3004 3005) do (
  for %%P in (efficientnetb1_ft vit32_frozen_imagenet vit32_ft_ckpt capsnet_r3) do (
    python entrenar_v3.py --manifest_csv %MANIFEST% --pipeline %%P --split_seed %%S --seeds 2027,2028,2029 || goto :error
  )
  python entrenar_v3.py --manifest_csv %MANIFEST% --pipeline capsnet_r3_revised --capsnet_config diagnostico_capsnet\capsnet_seleccion.json --split_seed %%S --seeds 2027,2028,2029 || goto :error
)
echo.
echo PASO 3 TERMINADO. Manda la carpeta resultados_v3
goto :eof
:error
echo *** Hubo un error. Copia el mensaje de arriba y mandalo. ***
exit /b 1
