"""
Script: Carga el dataset Telco Churn desde CSV a PostgreSQL.
- Lee data/telco_churn.csv
- Renombra columnas al formato de la tabla clientes_raw
- Inserta en PostgreSQL (Docker)
- Verifica que se cargaron los registros
"""
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# -----------------------------
# 1. Configuración
# -----------------------------
load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# Mapeo: nombre en el CSV  →  nombre en la tabla SQL
COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

# -----------------------------
# 2. Leer el CSV
# -----------------------------
print("📥 Leyendo CSV...")
df = pd.read_csv("data/telco_churn.csv")
print(f"   Filas: {len(df)}, Columnas: {len(df.columns)}")

# -----------------------------
# 3. Renombrar columnas
# -----------------------------
df = df.rename(columns=COLUMN_MAP)

# -----------------------------
# 4. Limpieza mínima
# -----------------------------
# 'TotalCharges' a veces viene con espacios en blanco → convertir a número
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
df["total_charges"] = df["total_charges"].fillna(0)

# -----------------------------
# 5. Conectar a PostgreSQL
# -----------------------------
print("🔌 Conectando a PostgreSQL...")
engine = create_engine(DB_URL)

# -----------------------------
# 6. Limpiar tabla antes de insertar (para poder re-ejecutar)
# -----------------------------
with engine.connect() as conn:
    conn.execute(text("TRUNCATE TABLE predicciones_churn CASCADE;"))
    conn.execute(text("TRUNCATE TABLE clientes_raw CASCADE;"))
    conn.commit()
print("🧹 Tablas limpiadas.")

# -----------------------------
# 7. Insertar datos
# -----------------------------
print("💾 Insertando en PostgreSQL...")
df.to_sql(
    "clientes_raw",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=1000,
)

# -----------------------------
# 8. Verificar
# -----------------------------
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM clientes_raw")).scalar()
    print(f"✅ Total de registros en la tabla: {count}")

    # Mostrar 3 ejemplos
    result = conn.execute(text(
        "SELECT customer_id, contract, monthly_charges, churn "
        "FROM clientes_raw LIMIT 3"
    ))
    print("\n📋 Muestra de registros:")
    for row in result:
        print(f"   {row}")