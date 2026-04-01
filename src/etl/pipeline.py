import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

# Conexión a Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Columnas que usamos
COLS_NUMERICAS   = ['ALC_mg/L', 'CONDUCT_mS/cm', 'SDT_mg/L', 'FLUORUROS_mg/L', 'DUR_mg/L']
COLS_CATEGORICAS = ['CALIDAD_ALC', 'CALIDAD_CONDUC', 'CALIDAD_SDT_ra', 'CALIDAD_FLUO', 'CALIDAD_DUR']

def cargar_datos(ruta):
    df = pd.read_excel(ruta, sheet_name='CASUB_12-2024')
    return df

def limpiar(df):
    # Convertir numéricas
    df[COLS_NUMERICAS] = df[COLS_NUMERICAS].apply(pd.to_numeric, errors='coerce')

    # Imputar con mediana (numéricas) y moda (categóricas)
    for col in COLS_NUMERICAS:
        df[col] = df[col].fillna(df[col].median())
    for col in COLS_CATEGORICAS:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def escalar(df):
    for col in COLS_NUMERICAS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def subir_a_supabase(df):
    registros = []
    for _, row in df.iterrows():
        registros.append({
            "clave_sitio":      str(row.get('CLAVE SITIO', '')),
            "sitio":            str(row.get('SITIO', '')),
            "estado":           str(row.get('ESTADO', '')),
            "municipio":        str(row.get('MUNICIPIO', '')),
            "acuifero":         str(row.get('ACUIFERO', '')),
            "periodo":          str(row.get('PERIODO', '')),
            "alc_mg_l":         float(row['ALC_mg/L']),
            "conduct_ms_cm":    float(row['CONDUCT_mS/cm']),
            "sdt_mg_l":         float(row['SDT_mg/L']),
            "fluoruros_mg_l":   float(row['FLUORUROS_mg/L']),
            "dur_mg_l":         float(row['DUR_mg/L']),
            "calidad_alc":      str(row['CALIDAD_ALC']),
            "calidad_conduc":   str(row['CALIDAD_CONDUC']),
            "calidad_sdt_ra":   str(row['CALIDAD_SDT_ra']),
            "calidad_fluo":     str(row['CALIDAD_FLUO']),
            "calidad_dur":      str(row['CALIDAD_DUR']),
            "semaforo":         str(row.get('SEMÁFORO', '')),
            "latitud":          float(row.get('LATITUD', 0)) if pd.notna(row.get('LATITUD')) else None,
            "longitud":         float(row.get('LONGITUD', 0)) if pd.notna(row.get('LONGITUD')) else None,
            "alc_original":       float(row['ALC_original']),
            "conduct_original":   float(row['CONDUCT_original']),
            "sdt_original":       float(row['SDT_original']),
            "fluoruros_original": float(row['FLUORUROS_original']),
            "dur_original":       float(row['DUR_original']),
        })

    # Subir en lotes de 500
    for i in range(0, len(registros), 500):
        lote = registros[i:i+500]
        supabase.table("calidad_agua").insert(lote).execute()
        print(f"  Subidos registros {i+1} a {i+len(lote)}")

def run():
    ruta = os.path.join("data", "raw", "Calidad_Agua_subterranea_2012_2024.xlsx")
    print("Cargando datos...")
    df = cargar_datos(ruta)
    print("Limpiando e imputando...")
    df = limpiar(df)
    
    # Guardar valores originales antes de escalar
    df['ALC_original']        = df['ALC_mg/L']
    df['CONDUCT_original']    = df['CONDUCT_mS/cm']
    df['SDT_original']        = df['SDT_mg/L']
    df['FLUORUROS_original']  = df['FLUORUROS_mg/L']
    df['DUR_original']        = df['DUR_mg/L']
    
    print("Escalando...")
    df = escalar(df)
    print("Subiendo a Supabase...")
    subir_a_supabase(df)
    print("✅ Pipeline completado.")

if __name__ == "__main__":
    run()