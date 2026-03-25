import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def obtener_datos():
    response = supabase.table("calidad_agua").select("*").execute()
    df = pd.DataFrame(response.data)
    return df

def entrenar():
    print("Obteniendo datos de Supabase...")
    df = obtener_datos()

    # Features y variable objetivo
    features = ['alc_mg_l', 'conduct_ms_cm', 'sdt_mg_l', 'fluoruros_mg_l', 'dur_mg_l']
    objetivo = 'semaforo'

    # Eliminar filas donde semaforo esté vacío
    df = df[df[objetivo].notna()]
    df = df[df[objetivo].str.strip() != '']

    print(f"Registros para entrenamiento: {len(df)}")
    print(f"Distribución de clases:\n{df[objetivo].value_counts()}\n")

    X = df[features]
    y = df[objetivo]

    # División 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar modelo
    print("Entrenando Random Forest...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluación
    y_pred = modelo.predict(X_test)
    print("=== Resultados ===")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Importancia de features
    importancias = pd.Series(modelo.feature_importances_, index=features)
    print("\nImportancia de variables:")
    print(importancias.sort_values(ascending=False))

    # Guardar modelo
    os.makedirs("src/model", exist_ok=True)
    with open("src/model/modelo.pkl", "wb") as f:
        pickle.dump(modelo, f)
    print("\n✅ Modelo guardado en src/model/modelo.pkl")

if __name__ == "__main__":
    entrenar()