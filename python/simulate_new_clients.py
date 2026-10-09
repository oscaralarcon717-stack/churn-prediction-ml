"""
Simula la llegada de clientes nuevos (campañas de marketing, portabilidad, etc.).

A diferencia de simulate_churn_events.py (que hace UPDATE sobre clientes
existentes), este script INSERTA filas nuevas en clientes_raw.

Cada corrida agrega un número variable de clientes nuevos (entre 10 y 30),
con valores plausibles.

"""
from __future__ import annotations

import logging
import random
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from config import ID_COL, TABLE_CLIENTES, TARGET_COL, build_db_url

logger = logging.getLogger("churn.new_clients")

MIN_NEW = 10
MAX_NEW = 30

CATEGORY_OPTIONS = {
    "gender": ["Male", "Female"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phone_service": ["Yes", "No"],
    "multiple_lines": ["Yes", "No", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["Yes", "No", "No internet service"],
    "online_backup": ["Yes", "No", "No internet service"],
    "device_protection": ["Yes", "No", "No internet service"],
    "tech_support": ["Yes", "No", "No internet service"],
    "streaming_tv": ["Yes", "No", "No internet service"],
    "streaming_movies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}


def generate_client(index: int, fecha: str) -> dict:
    """Genera un cliente nuevo con valores plausibles. Llegan recién
    contratados: tenure=0, total_charges=0."""
    return {
        ID_COL: f"NEW-{fecha}-{index:04d}",
        "gender": random.choice(CATEGORY_OPTIONS["gender"]),
        "senior_citizen": random.choice([0, 1]),
        "partner": random.choice(CATEGORY_OPTIONS["partner"]),
        "dependents": random.choice(CATEGORY_OPTIONS["dependents"]),
        "tenure": 0,
        "phone_service": random.choice(CATEGORY_OPTIONS["phone_service"]),
        "multiple_lines": random.choice(CATEGORY_OPTIONS["multiple_lines"]),
        "internet_service": random.choice(CATEGORY_OPTIONS["internet_service"]),
        "online_security": random.choice(CATEGORY_OPTIONS["online_security"]),
        "online_backup": random.choice(CATEGORY_OPTIONS["online_backup"]),
        "device_protection": random.choice(CATEGORY_OPTIONS["device_protection"]),
        "tech_support": random.choice(CATEGORY_OPTIONS["tech_support"]),
        "streaming_tv": random.choice(CATEGORY_OPTIONS["streaming_tv"]),
        "streaming_movies": random.choice(CATEGORY_OPTIONS["streaming_movies"]),
        "contract": random.choice(CATEGORY_OPTIONS["contract"]),
        "paperless_billing": random.choice(CATEGORY_OPTIONS["paperless_billing"]),
        "payment_method": random.choice(CATEGORY_OPTIONS["payment_method"]),
        "monthly_charges": round(random.uniform(18, 120), 2),
        "total_charges": 0.0,
        TARGET_COL: "No",
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        load_dotenv()
        n_nuevos = random.randint(MIN_NEW, MAX_NEW)
        fecha = datetime.now().strftime("%Y%m%d")

        logger.info("Generando %d clientes nuevos (fecha %s)...", n_nuevos, fecha)
        nuevos = [generate_client(i, fecha) for i in range(n_nuevos)]
        df = pd.DataFrame(nuevos)

        engine = create_engine(build_db_url())
        try:
            df.to_sql(
                TABLE_CLIENTES, engine, if_exists="append",
                index=False, method="multi", chunksize=1000,
            )
            with engine.connect() as conn:
                total = conn.execute(
                    text(f"SELECT COUNT(*) FROM {TABLE_CLIENTES}")
                ).scalar()
        finally:
            engine.dispose()

        logger.info("Total de clientes en BD: %d", total)
        logger.info("Clientes nuevos agregados correctamente")
        return 0
    except Exception:
        logger.exception("Falló la simulación de clientes nuevos")
        return 1


if __name__ == "__main__":
    sys.exit(main())