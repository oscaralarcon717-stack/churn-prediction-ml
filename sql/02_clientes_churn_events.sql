-- Historial de bajas y reactivaciones.
-- clientes_raw solo guarda el estado ACTUAL de cada cliente (se sobrescribe);
-- esta tabla guarda cada cambio con fecha, para reconstruir la historia.
CREATE TABLE IF NOT EXISTS clientes_churn_events (
    id             BIGSERIAL PRIMARY KEY,
    customer_id    TEXT NOT NULL REFERENCES clientes_raw (customer_id),
    event_type     TEXT NOT NULL CHECK (event_type IN ('churn', 'reactivation')),
    occurred_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_clientes_churn_events_customer
    ON clientes_churn_events (customer_id);