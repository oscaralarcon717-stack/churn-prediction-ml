@echo off
REM ============================================
REM Reentrenamiento del modelo
REM Frecuencia: dia 1 de cada mes a las 5:00 AM
REM ============================================

cd /d C:\Users\HP\Desktop\churn-prediction-ml
call venv\Scripts\activate

if not exist logs mkdir logs

echo. >> logs\pipeline.log
echo [%date% %time%] ===== TAREA B: Reentrenamiento ===== >> logs\pipeline.log

echo [%date% %time%] Reentrenando modelo >> logs\pipeline.log
python python\02_entrenar_modelo.py >> logs\pipeline.log 2>&1

echo [%date% %time%] Regenerando predicciones >> logs\pipeline.log
python python\03_generar_predicciones.py >> logs\pipeline.log 2>&1

echo [%date% %time%] Tarea B completada >> logs\pipeline.log
deactivate