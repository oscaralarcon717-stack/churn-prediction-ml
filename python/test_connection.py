"""
Script de prueba: verifica que Python puede conectarse a PostgreSQL en Docker.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Carga las variables del archivo .env
load_dotenv()

# Construye la URL de conexión
DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

print("🔌 Intentando conectar a PostgreSQL...")

# Crea la conexión
engine = create_engine(DB_URL)

# Prueba una consulta simple
with engine.connect() as conn:
    # Versión de PostgreSQL
    result = conn.execute(text("SELECT version();"))
    version = result.fetchone()[0]
    print("✅ ¡Conexión exitosa!")
    print(f"📊 Versión: {version[:50]}...")

    # Contar tablas
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """))
    tablas = [row[0] for row in result]
    print(f"📋 Tablas encontradas: {tablas}")