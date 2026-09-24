"""
Entrenamiento del modelo de predicción de churn.

Flujo:
    1. Lee la configuración desde variables de entorno (.env)
    2. Carga los datos desde PostgreSQL (tabla clientes_raw)
    3. Valida y limpia los datos
    4. Divide en train/test
    5. Elige el umbral de decisión usando validación cruzada (solo con train)
    6. Entrena el modelo final y lo evalúa en test (datos que nunca vio)
    7. Guarda el modelo + metadata versionados en models/


"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

logger = logging.getLogger("churn.train")

# ============================================================
# CONSTANTES (todo lo "configurable" vive aquí, arriba del archivo)
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
MIN_ROC_AUC_TO_PROMOTE = 0.75  # si el modelo rinde menos que esto, NO se marca como "latest"

MODEL_DIR = Path(__file__).resolve().parent / "models"

TABLE_NAME = "clientes_raw"
ID_COL = "customer_id"
TARGET_COL = "churn"

# Listas EXPLÍCITAS: si aparece una columna nueva en la BD, no entra al modelo por accidente.
NUMERICAL_COLS = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL_COLS = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method",
]
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


# ============================================================
# 1. CONFIGURACIÓN Y DATOS
# ============================================================
def build_db_url() -> URL:
    """Construye la URL de conexión validando que estén todas las variables."""
    required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f"Faltan variables de entorno: {', '.join(missing)}")

    # URL.create escapa bien caracteres raros en la contraseña (@, #, /, etc.)
    return URL.create(
        drivername="postgresql",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        database=os.environ["DB_NAME"],
    )


def load_data(db_url: URL) -> pd.DataFrame:
    """Lee la tabla de clientes desde PostgreSQL."""
    columns = [ID_COL, *FEATURE_COLS, TARGET_COL]
    query = f"SELECT {', '.join(columns)} FROM {TABLE_NAME}"  # noqa: S608 (solo constantes)

    engine = create_engine(db_url)
    try:
        df = pd.read_sql(query, engine)
    finally:
        engine.dispose()  # siempre cerramos las conexiones

    logger.info("Datos cargados: %d filas, %d columnas", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y limpia los datos. Falla con un error claro si algo está muy mal."""
    df = df.copy()

    # Duplicados por cliente
    n_before = len(df)
    df = df.drop_duplicates(subset=ID_COL)
    if len(df) < n_before:
        logger.warning("Se eliminaron %d clientes duplicados", n_before - len(df))

    # Target: Yes/No -> 1/0 (tolerante a mayúsculas y espacios)
    target = df[TARGET_COL].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    n_invalid = int(target.isna().sum())
    if n_invalid:
        logger.warning("Se eliminaron %d filas con 'churn' inválido o vacío", n_invalid)
    df[TARGET_COL] = target
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Numéricas: forzamos número; lo que no se pueda convertir queda NaN
    # (ej. total_charges vacío en clientes nuevos). El imputer lo rellenará luego.
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=[ID_COL])

    # Comprobaciones mínimas
    if df[TARGET_COL].nunique() < 2:
        raise ValueError("El target solo tiene una clase; no se puede entrenar.")
    if len(df) < 500:
        raise ValueError(f"Muy pocos datos para entrenar ({len(df)} filas).")

    churn_rate = df[TARGET_COL].mean()
    logger.info(
        "Datos limpios: %d filas | se van: %.1f%% | se quedan: %.1f%%",
        len(df), churn_rate * 100, (1 - churn_rate) * 100,
    )
    return df


# ============================================================
# 2. MODELO
# ============================================================
def build_pipeline() -> Pipeline:
    """Preprocesamiento + Random Forest en un solo objeto."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # rellena nulos con la mediana
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERICAL_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def cross_validated_probabilities(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Probabilidades "honestas" para cada fila de train: cada fila se predice con un
    modelo que NO la vio al entrenar (validación cruzada estratificada).
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]


def find_best_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Busca el umbral que maximiza F1 (equilibrio entre precision y recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1_scores)])


# ============================================================
# 3. EVALUACIÓN
# ============================================================
def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """Evalúa el modelo final sobre el test (datos que nunca vio)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }

    logger.info("Métricas en TEST (umbral=%.3f): %s", threshold,
                {k: round(v, 4) for k, v in metrics.items()})
    logger.info("Reporte:\n%s",
                classification_report(y_test, pred, target_names=["Se queda", "Se va"]))

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    logger.info("Matriz de confusión | VN=%d FP=%d FN=%d VP=%d", tn, fp, fn, tp)
    return metrics


def top_feature_importances(pipeline: Pipeline, top_n: int = 10) -> list[dict]:
    """Las variables que más pesan en las decisiones del modelo."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    ranked = sorted(zip(names, importances), key=lambda pair: pair[1], reverse=True)[:top_n]

    logger.info("Top %d variables más importantes:", top_n)
    for name, value in ranked:
        logger.info("   %-45s %.4f", name, value)
    return [{"feature": str(n), "importance": float(v)} for n, v in ranked]


# ============================================================
# 4. GUARDADO
# ============================================================
def save_artifact(pipeline: Pipeline, threshold: float, metadata: dict) -> Path:
    """Guarda modelo + umbral + metadata, con versión por fecha."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metadata = {**metadata, "version": version}

    model_path = MODEL_DIR / f"churn_model_{version}.pkl"
    meta_path = MODEL_DIR / f"churn_model_{version}.json"

    # Guardamos TODO lo necesario para predecir en un solo archivo
    joblib.dump(
        {"pipeline": pipeline, "threshold": threshold, "metadata": metadata},
        model_path,
    )
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Modelo guardado: %s", model_path)

    # "Quality gate": solo se promueve a 'latest' si supera el mínimo aceptable
    if metadata["test_metrics"]["roc_auc"] >= MIN_ROC_AUC_TO_PROMOTE:
        shutil.copy2(model_path, MODEL_DIR / "churn_model_latest.pkl")
        logger.info("Modelo promovido a churn_model_latest.pkl")
    else:
        logger.warning(
            "ROC-AUC %.3f < %.2f: NO se promueve a 'latest'. Revisa los datos.",
            metadata["test_metrics"]["roc_auc"], MIN_ROC_AUC_TO_PROMOTE,
        )
    return model_path


# ============================================================
# 5. ORQUESTACIÓN
# ============================================================
def train(df: pd.DataFrame) -> Path:
    """Recibe datos limpios y ejecuta todo el entrenamiento. (Separada para poder testearla sin BD.)"""
    X, y = df[FEATURE_COLS], df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Split: train=%d | test=%d", len(X_train), len(X_test))

    pipeline = build_pipeline()

    # Validación cruzada SOLO con train -> métricas estables + umbral óptimo
    logger.info("Validación cruzada (%d folds)...", N_FOLDS)
    oof_proba = cross_validated_probabilities(pipeline, X_train, y_train)
    cv_metrics = {
        "roc_auc": float(roc_auc_score(y_train, oof_proba)),
        "pr_auc": float(average_precision_score(y_train, oof_proba)),
    }
    logger.info("Métricas CV: %s", {k: round(v, 4) for k, v in cv_metrics.items()})
    threshold = find_best_threshold(y_train, oof_proba)
    logger.info("Umbral elegido: %.3f", threshold)

    # Entrenamiento final y evaluación en test
    logger.info("Entrenando modelo final...")
    pipeline.fit(X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test, threshold)
    importances = top_feature_importances(pipeline)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "n_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "features": FEATURE_COLS,
        "threshold": threshold,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "top_features": importances,
    }
    return save_artifact(pipeline, threshold, metadata)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        load_dotenv()
        raw = load_data(build_db_url())
        df = clean_data(raw)
        train(df)
        logger.info("Entrenamiento finalizado")
        return 0
    except Exception:
        logger.exception("El entrenamiento falló")
        return 1  # código de salida != 0 para que cron/Airflow/CI detecten el fallo


if __name__ == "__main__":
    sys.exit(main())