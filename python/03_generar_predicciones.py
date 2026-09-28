"""
Genera predicciones de churn para todos los clientes activos
y las escribe en la tabla predicciones_churn de PostgreSQL.

Flujo:
    1. Carga el modelo entrenado (churn_model_latest.pkl)
    2. Lee los clientes activos desde clientes_raw
    3. Genera probabilidad de churn para cada uno
    4. Clasifica en risk_level (Bajo / Medio / Alto), anclado al umbral del modelo
    5. Reemplaza el contenido de predicciones_churn dentro de una transacción,
       verificando que se insertaron todas las filas antes de confirmar

"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from config import (
    FEATURE_COLS,
    ID_COL,
    MODEL_LATEST_PATH,
    NUMERICAL_COLS,
    RISK_LOW_MAX,
    RISK_MED_MAX,
    TABLE_CLIENTES,
    TABLE_PRED,
    TARGET_COL,
    build_db_url,
)

logger = logging.getLogger("churn.predict")


# ============================================================
# 1. CARGAR MODELO
# ============================================================
def load_model():
    """Carga el pipeline, el threshold y la metadata desde el artefacto guardado."""
    if not MODEL_LATEST_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_LATEST_PATH}.\n"
            f"Ejecuta primero: python train_churn_model.py"
        )

    artefacto = joblib.load(MODEL_LATEST_PATH)
    pipeline = artefacto["pipeline"]
    threshold = artefacto["threshold"]
    metadata = artefacto["metadata"]

    logger.info("Modelo cargado: versión %s", metadata.get("version", "desconocida"))
    logger.info("Threshold: %.4f | ROC-AUC (test): %.4f",
                threshold, metadata["test_metrics"]["roc_auc"])

    # Aviso si el entorno actual difiere del que entrenó el modelo.
    # No es fatal, pero puede causar diferencias sutiles en las predicciones.
    trained_version = metadata.get("sklearn_version")
    if trained_version and trained_version != sklearn.__version__:
        logger.warning(
            "El modelo se entrenó con scikit-learn %s y aquí hay %s instalado",
            trained_version, sklearn.__version__,
        )

    # Si las columnas del modelo ya no coinciden con config.py, mejor fallar ya
    # que dejar que predict_proba() falle más abajo con un error menos claro.
    expected = metadata.get("features")
    if expected and list(expected) != list(FEATURE_COLS):
        raise ValueError(
            "Las columnas del modelo no coinciden con config.FEATURE_COLS. "
            "¿Se actualizó config.py después de entrenar? Reentrena el modelo."
        )

    return pipeline, threshold, metadata


# ============================================================
# 2. LEER CLIENTES
# ============================================================
def load_clients(db_url) -> pd.DataFrame:
    """
    Lee los clientes activos (no dados de baja). Puntuar a quien ya se fue
    no sirve para retención, y además esas filas suelen sesgar las métricas
    porque el modelo las vio de cerca en el histórico de entrenamiento.
    """
    cols = [ID_COL, *FEATURE_COLS]
    query = (
        f"SELECT {', '.join(cols)} FROM {TABLE_CLIENTES} "  # noqa: S608
        f"WHERE {TARGET_COL} IS DISTINCT FROM 'Yes'"
    )

    engine = create_engine(db_url)
    try:
        df = pd.read_sql(query, engine)
    finally:
        engine.dispose()

    logger.info("Clientes activos cargados: %d", len(df))
    return df


# ============================================================
# 3. GENERAR PREDICCIONES
# ============================================================
def predict(df: pd.DataFrame, pipeline, threshold: float) -> pd.DataFrame:
    """Genera probabilidades y clasifica el riesgo."""
    df = df.copy()

    # Mismos tipos que en entrenamiento
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Generando predicciones...")
    proba = pipeline.predict_proba(df[FEATURE_COLS])[:, 1]
    pred = (proba >= threshold).astype(int)

    # Los niveles de riesgo quedan anclados al umbral del modelo, para que
    # "Alto" siempre implique churn_prediction == 1 y nunca se contradigan.
    # RISK_LOW_MAX/RISK_MED_MAX se reescalan proporcionalmente al umbral.
    bins = sorted({0.0, threshold * (RISK_LOW_MAX / RISK_MED_MAX), threshold, 1.0})
    df["risk_level"] = pd.cut(
        proba,
        bins=[-float("inf"), *bins[1:-1], float("inf")],
        labels=["Bajo", "Medio", "Alto"][: len(bins) - 1] or ["Alto"],
        right=False,
    ).astype(str)

    df["churn_probability"] = proba
    df["churn_prediction"] = pred

    logger.info("Predicciones generadas: %d", len(df))
    logger.info("Distribución de riesgo:")
    for level in ["Bajo", "Medio", "Alto"]:
        n = (df["risk_level"] == level).sum()
        pct = 100 * n / len(df) if len(df) else 0.0
        logger.info("   %-6s: %5d (%.1f%%)", level, n, pct)

    return df


# ============================================================
# 4. ESCRIBIR A SQL
# ============================================================
def save_predictions(df: pd.DataFrame, db_url, version: str) -> int:
    """
    Reemplaza el contenido de predicciones_churn dentro de una transacción.
    Si el conteo final no coincide con lo que se intentó insertar, se lanza
    un error y la transacción hace rollback: la tabla queda como estaba.
    """
    output = df[[
        ID_COL, "churn_probability", "churn_prediction", "risk_level",
    ]].copy()
    output["modelo_version"] = version
    output["fecha_prediccion"] = datetime.now(timezone.utc)
    engine = create_engine(db_url)
    try:
        with engine.begin() as conn:
            # DELETE en vez de TRUNCATE: no toma un bloqueo exclusivo de tabla,
            # así que no bloquea lecturas de un dashboard mientras corre.
            conn.execute(text(f"DELETE FROM {TABLE_PRED}"))

            output.to_sql(
                TABLE_PRED, conn, if_exists="append", index=False,
                method="multi", chunksize=1000,
            )

            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_PRED}")).scalar()
            if count != len(output):
                raise RuntimeError(
                    f"Se esperaban {len(output)} filas insertadas y hay {count}. "
                    f"Se revierte la transacción."
                )
    finally:
        engine.dispose()

    logger.info("Predicciones insertadas: %d", count)
    return count


# ============================================================
# 5. ORQUESTACIÓN
# ============================================================
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        load_dotenv()

        pipeline, threshold, metadata = load_model()
        version = metadata.get("version", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

        db_url = build_db_url()
        df = load_clients(db_url)

        # Sin esto, una tabla clientes_raw vacía (por un fallo previo en el
        # pipeline de datos) borraría todas las predicciones existentes
        # dejando el dashboard en blanco, sin que nadie se entere.
        if df.empty:
            raise ValueError(
                f"{TABLE_CLIENTES} no devolvió clientes activos; se aborta "
                f"sin tocar {TABLE_PRED} para no perder las predicciones vigentes."
            )

        df = predict(df, pipeline, threshold)
        save_predictions(df, db_url, version)

        logger.info("Proceso de predicción completado")
        return 0
    except Exception:
        logger.exception("Falló el proceso de predicción")
        return 1


if __name__ == "__main__":
    sys.exit(main())