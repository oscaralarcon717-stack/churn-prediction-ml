"""
Configuracion compartida por los scripts de entrenamiento y prediccion.
Un solo lugar para las columnas y la conexion.
"""
import os
from pathlib import Path

from sqlalchemy.engine import URL

# ============================================================
# RUTAS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_LATEST_PATH = MODEL_DIR / "churn_model_latest.pkl"

# ============================================================
# TABLAS
# ============================================================
TABLE_CLIENTES = "clientes_raw"
TABLE_PRED = "predicciones_churn"

ID_COL = "customer_id"
TARGET_COL = "churn"

# ============================================================
# COLUMNAS DEL MODELO
# ============================================================
NUMERICAL_COLS = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL_COLS = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method",
]
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# ============================================================
# NIVELES DE RIESGO
# ============================================================
RISK_LOW_MAX = 0.3
RISK_MED_MAX = 0.6


def build_db_url() -> URL:
    """Construye la URL de conexion a PostgreSQL."""
    required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f"Faltan variables de entorno: {', '.join(missing)}")

    return URL.create(
        drivername="postgresql",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        database=os.environ["DB_NAME"],
    )