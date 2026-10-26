@echo off
REM PASO 5: busqueda de imagenes duplicadas con distinto lesion_id + copia de 12 ejemplos de errores.
REM Solo inferencia (no entrena nada). Tarda unos 10-30 minutos en CPU.
call CONFIGURAR.bat
python duplicados.py --metadata %METADATA% --ham_dirs "%HAM_DIRS%" || goto :error
python copiar_ejemplos.py --ham_dirs "%HAM_DIRS%" || goto :error
echo.
echo PASO 5 TERMINADO. Comprime y manda las carpetas duplicados y ejemplos_errores
echo (no hace falta mandar duplicados\embeddings.npy, pesa mucho).
goto :eof
:error
echo *** Hubo un error. Copia el mensaje de arriba y mandalo. ***
exit /b 1
