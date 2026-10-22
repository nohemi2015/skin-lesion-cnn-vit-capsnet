@echo off
REM PASO 4: cohorte no usada de HAM10000 (solo inferencia) + analisis de todo.
call CONFIGURAR.bat
python cohorte_confirmacion.py --metadata %METADATA% --ham_dirs "%HAM_DIRS%" --manifest_csv %MANIFEST% || goto :error
python analisis_revision.py || goto :error
echo.
echo PASO 4 TERMINADO. Manda las carpetas analisis_revision, confirmacion y resultados_v3
goto :eof
:error
echo *** Hubo un error. Copia el mensaje de arriba y mandalo. ***
exit /b 1
