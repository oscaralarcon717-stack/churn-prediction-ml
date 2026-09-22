-- ============================================
-- TABLA 1: clientes_raw
-- Guarda la info cruda de cada cliente
-- ============================================
CREATE TABLE IF NOT EXISTS clientes_raw (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    tenure INT,
    phone_service VARCHAR(5),
    multiple_lines VARCHAR(20),
    internet_service VARCHAR(20),
    online_security VARCHAR(20),
    online_backup VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support VARCHAR(20),
    streaming_tv VARCHAR(20),
    streaming_movies VARCHAR(20),
    contract VARCHAR(20),
    paperless_billing VARCHAR(5),
    payment_method VARCHAR(50),
    monthly_charges NUMERIC(10,2),
    total_charges NUMERIC(10,2),
    churn VARCHAR(5),
    fecha_ingesta TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- TABLA 2: predicciones_churn
-- Guardará las predicciones del modelo ML
-- ============================================
CREATE TABLE IF NOT EXISTS predicciones_churn (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES clientes_raw(customer_id),
    churn_probability NUMERIC(5,4),
    churn_prediction INT,
    risk_level VARCHAR(10),
    modelo_version VARCHAR(20),
    fecha_prediccion TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- ÍNDICES (hacen las búsquedas más rápidas)
-- ============================================
CREATE INDEX IF NOT EXISTS idx_pred_customer ON predicciones_churn(customer_id);
CREATE INDEX IF NOT EXISTS idx_pred_fecha ON predicciones_churn(fecha_prediccion);
CREATE INDEX IF NOT EXISTS idx_pred_risk ON predicciones_churn(risk_level);