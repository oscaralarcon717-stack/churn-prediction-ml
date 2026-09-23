"""
Script: Entrena un modelo de Machine Learning para predecir churn.
- Lee datos de PostgreSQL (tabla clientes_raw)
- Preprocesa features (codifica variables categóricas)
- Divide en train/test
- Entrena un modelo Random Forest
- Evalúa con métricas
- Guarda el modelo en disco (models/churn_model.pkl)
"""
import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. CONFIGURACIÓN Y CONEXIÓN
# ============================================================
load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

print("=" * 60)
print("🧠 ENTRENAMIENTO DEL MODELO DE CHURN")
print("=" * 60)

# ============================================================
# 2. LEER DATOS DE POSTGRESQL
# ============================================================
print("\n📥 Leyendo datos de PostgreSQL...")
engine = create_engine(DB_URL)

query = """
    SELECT 
        customer_id,
        gender, senior_citizen, partner, dependents,
        tenure, phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges, churn
    FROM clientes_raw
"""

df = pd.read_sql(query, engine)
print(f"   ✅ {len(df)} registros cargados")
print(f"   ✅ {len(df.columns)} columnas")

# ============================================================
# 3. PREPARAR FEATURES Y TARGET
# ============================================================
print("\n🔧 Preparando datos...")

# Quitamos customer_id (es solo un identificador, no aporta al modelo)
df = df.drop(columns=["customer_id"])

# Convertimos el target 'churn' (Yes/No) a 1/0
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

# Separamos:
# X = features (todas las columnas menos 'churn')
# y = target (lo que queremos predecir)
X = df.drop(columns=["churn"])
y = df["churn"]

print(f"   ✅ Features (X): {X.shape}")
print(f"   ✅ Target (y): {y.shape}")
print(f"   ✅ Balance del target:")
print(f"      - Se quedan (0): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
print(f"      - Se van (1):    {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")

# ============================================================
# 4. IDENTIFICAR COLUMNAS NUMÉRICAS Y CATEGÓRICAS
# ============================================================
numerical_cols = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
categorical_cols = [col for col in X.columns if col not in numerical_cols]

print(f"\n   📊 Columnas numéricas ({len(numerical_cols)}):")
print(f"      {numerical_cols}")
print(f"\n   📊 Columnas categóricas ({len(categorical_cols)}):")
print(f"      {categorical_cols}")

# ============================================================
# 5. CREAR PIPELINE DE PREPROCESAMIENTO
# ============================================================
# Las columnas numéricas se escalan (para que tengan la misma magnitud)
# Las columnas categóricas se convierten en números (One-Hot Encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ]
)

# ============================================================
# 6. DIVIDIR EN TRAIN Y TEST
# ============================================================
print("\n✂️  Dividiendo en train (80%) y test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✅ Train: {X_train.shape[0]} registros")
print(f"   ✅ Test:  {X_test.shape[0]} registros")

# ============================================================
# 7. CREAR PIPELINE COMPLETO (PREPROCESAMIENTO + MODELO)
# ============================================================
print("\n🏗️  Creando pipeline con Random Forest...")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight="balanced",   # Importante: compensa el desbalance (73% vs 27%)
        random_state=42,
        n_jobs=-1,
    )),
])

# ============================================================
# 8. ENTRENAR
# ============================================================
print("\n🚀 Entrenando el modelo...")
model.fit(X_train, y_train)
print("   ✅ Entrenamiento completado")

# ============================================================
# 9. EVALUAR
# ============================================================
print("\n" + "=" * 60)
print("📊 RESULTADOS DEL MODELO")
print("=" * 60)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n✅ Accuracy:  {accuracy:.4f}  (de cada 100, acierta {accuracy*100:.1f})")
print(f"✅ Precision: {precision:.4f}  (de los que predijo 'se va', {precision*100:.1f}% acertó)")
print(f"✅ Recall:    {recall:.4f}  (detectó {recall*100:.1f}% de los que se fueron)")
print(f"✅ F1-Score:  {f1:.4f}")
print(f"✅ AUC-ROC:   {auc:.4f}  (1.0 = perfecto, 0.5 = azar)")

print("\n📋 Reporte completo:")
print(classification_report(y_test, y_pred, target_names=["Se queda", "Se va"]))

print("🧩 Matriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
print(f"   Verdaderos Negativos (acertó 'se queda'): {cm[0][0]}")
print(f"   Falsos Positivos (dijo 'se va' pero se quedó): {cm[0][1]}")
print(f"   Falsos Negativos (dijo 'se queda' pero se fue): {cm[1][0]}")
print(f"   Verdaderos Positivos (acertó 'se va'): {cm[1][1]}")

# ============================================================
# 10. GUARDAR EL MODELO
# ============================================================
os.makedirs("python/models", exist_ok=True)
model_path = "python/models/churn_model.pkl"
joblib.dump(model, model_path)
print(f"\n💾 Modelo guardado en: {model_path}")

print("\n" + "=" * 60)
print("🎉 ENTRENAMIENTO FINALIZADO")
print("=" * 60)