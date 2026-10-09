@echo off
REM ============================================
REM Pipeline: Simulaciones + Prediccion
REM Frecuencia: cada 9 dias a las 6:00 AM
REM ============================================

cd /d C:\Users\HP\Desktop\churn-prediction-ml
call venv\Scripts\activate

if not exist logs mkdir logs

echo. >> logs\pipeline.log
echo [%date% %time%] ===== TAREA A: Simulaciones + Prediccion ===== >> logs\pipeline.log

echo [%date% %time%] Paso 1: Simular clientes nuevos >> logs\pipeline.log
python python\simulate_new_clients.py >> logs\pipeline.log 2>&1

echo [%date% %time%] Paso 2: Simular eventos de churn >> logs\pipeline.log
python python\simulate_churn_events.py >> logs\pipeline.log 2>&1

echo [%date% %time%] Paso 3: Generar predicciones >> logs\pipeline.log
python python\03_generar_predicciones.py >> logs\pipeline.log 2>&1

echo [%date% %time%] Tarea A completada >> logs\pipeline.log
deactivate