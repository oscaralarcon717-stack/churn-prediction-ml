"""
Simula eventos de churn sobre clientes YA existentes en clientes_raw:
un porcentaje de los activos se va (churn: No -> Yes) y un porcentaje
de los que ya se habían ido regresa (churn: Yes -> No).

A diferencia de simulate_new_clients.py (que INSERTA filas nuevas), este
script hace UPDATE sobre filas existentes.

Cada cambio además queda registrado en clientes_churn_events (ver
create_clientes_churn_events.sql), para no perder el historial de cuándo
pasó cada baja/reactivación: clientes_raw solo guarda el estado actual.

"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text

from config import ID_COL, TABLE_CLIENTES, TARGET_COL, build_db_url

logger = logging.getLogger("churn.eventos")

# % de clientes activos que se van, y % de clientes idos que regresan, en cada corrida.
LEAVE_RATE = 0.03    # 3% de los que tienen churn='No'
RETURN_RATE = 0.10   # 10% de los que tienen churn='Yes' (ese grupo suele ser más chico)

# Peso base que tiene TODO cliente activo de ser elegido, antes de aplicar el
# riesgo. Sin este piso, el sorteo ponderado tiende a excluir casi del todo a
# los de bajo riesgo (contrato largo, mucho tenure), lo que determiniza la
# relación riesgo->churn y le enseña al modelo una regla demasiado perfecta,
# que no existe en los datos reales (ahí también se va gente "de bajo riesgo").
RISK_BASE_WEIGHT = 0.3

EVENTS_TABLE = "clientes_churn_events"


# ============================================================
# 1. SELECCIÓN 
# ============================================================
def _risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Puntaje de riesgo (no es el modelo entrenado, solo una heurística para dar
    forma realista a los datos simulados): contrato mensual, poco tenure y
    cobro alto suben el riesgo. Las columnas que falten se ignoran.
    """
    score = pd.Series(0.0, index=df.index)
    if "contract" in df.columns:
        score += (df["contract"] == "Month-to-month").astype(float) * 0.5
    if "tenure" in df.columns:
        score += (1 - df["tenure"].clip(0, 72) / 72) * 0.3
    if "monthly_charges" in df.columns:
        score += (df["monthly_charges"].clip(0, 120) / 120) * 0.2
    return score


def select_departures(df: pd.DataFrame, leave_rate: float, random_state=None) -> list[str]:
    """
    Elige qué clientes activos ('No') se van ahora, con un SORTEO PONDERADO
    por riesgo : cada cliente activo tiene SIEMPRE una
    probabilidad de ser elegido (ver RISK_BASE_WEIGHT), y el riesgo solo la
    inclina. Esto preserva el ruido que existe en los datos reales -no todo
    el que se va es de alto riesgo, ni todo el de alto riesgo se va- y evita
    que el modelo aprenda una regla perfecta y artificial en vez del patrón
    real, más difuso.
    """
    leave_rate = max(0.0, min(1.0, leave_rate))
    active = df[df[TARGET_COL] == "No"]
    n = min(len(active), round(len(active) * leave_rate))
    if n == 0:
        return []
    weights = RISK_BASE_WEIGHT + _risk_score(active)
    return active.sample(n=n, weights=weights, random_state=random_state)[ID_COL].tolist()


def select_reactivations(df: pd.DataFrame, return_rate: float, random_state=None) -> list[str]:
    """Elige al azar qué clientes idos ('Yes') regresan ahora ('No')."""
    return_rate = max(0.0, min(1.0, return_rate))
    churned = df[df[TARGET_COL] == "Yes"]
    n = min(len(churned), round(len(churned) * return_rate))
    if n == 0:
        return []
    return churned.sample(n=n, random_state=random_state)[ID_COL].tolist()


# ============================================================
# 2. LECTURA
# ============================================================
def load_status(engine) -> pd.DataFrame:
    """Solo lee id + estado de churn; no hace falta traer todas las columnas."""
    query = f"SELECT {ID_COL}, {TARGET_COL} FROM {TABLE_CLIENTES}"  # noqa: S608
    df = pd.read_sql(query, engine)
    logger.info(
        "Estado actual: %d activos, %d idos",
        (df[TARGET_COL] == "No").sum(), (df[TARGET_COL] == "Yes").sum(),
    )
    return df


# ============================================================
# 3. ESCRITURA
# ============================================================
def apply_departures(conn, ids: list[str]) -> None:
    if not ids:
        return
    stmt = text(
        f"UPDATE {TABLE_CLIENTES} SET {TARGET_COL} = 'Yes' "
        f"WHERE {ID_COL} IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    conn.execute(stmt, {"ids": ids})
    _log_events(conn, ids, "churn")


def apply_reactivations(conn, ids: list[str]) -> None:
    if not ids:
        return
    # Al regresar, el cliente empieza de nuevo: reseteamos tenure y total_charges.
    # monthly_charges se deja igual (es la tarifa del plan, no hay cambios por irse y volver).
    stmt = text(
        f"UPDATE {TABLE_CLIENTES} SET {TARGET_COL} = 'No', tenure = 0, total_charges = 0 "
        f"WHERE {ID_COL} IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    conn.execute(stmt, {"ids": ids})
    _log_events(conn, ids, "reactivation")


def _log_events(conn, ids: list[str], event_type: str) -> None:
    """Registra cada cambio en la tabla de eventos (ver DDL adjunto). No falla
    el proceso si la tabla de eventos no existe todavía: solo avisa."""
    now = datetime.now(timezone.utc)
    rows = [{"customer_id": cid, "event_type": event_type, "occurred_at": now} for cid in ids]
    try:
        conn.execute(
            text(
                f"INSERT INTO {EVENTS_TABLE} (customer_id, event_type, occurred_at) "
                f"VALUES (:customer_id, :event_type, :occurred_at)"
            ),
            rows,
        )
    except Exception:
        logger.warning(
            "No se pudo escribir en %s (¿corriste create_clientes_churn_events.sql?). "
            "clientes_raw sí se actualizó; solo se perdió el registro del evento.",
            EVENTS_TABLE,
        )


# ============================================================
# 4. ORQUESTACIÓN
# ============================================================
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        load_dotenv()
        engine = create_engine(build_db_url())
        try:
            df = load_status(engine)
            if df.empty:
                logger.warning("%s está vacía; no hay nada que actualizar", TABLE_CLIENTES)
                return 0

            departures = select_departures(df, LEAVE_RATE)
            reactivations = select_reactivations(df, RETURN_RATE)
            logger.info(
                "Se van: %d clientes | Regresan: %d clientes", len(departures), len(reactivations)
            )

            with engine.begin() as conn:
                apply_departures(conn, departures)
                apply_reactivations(conn, reactivations)
        finally:
            engine.dispose()

        logger.info("Eventos de churn simulados completados")
        return 0
    except Exception:
        logger.exception("Falló la simulación de eventos de churn")
        return 1


if __name__ == "__main__":
    sys.exit(main())