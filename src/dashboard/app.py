import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from supabase import create_client
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Cargar modelo
with open("src/model/modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Calidad del Agua Subterránea MX",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Calidad del Agua Subterránea — CONAGUA")
st.markdown("Análisis y predicción de calidad del agua subterránea en México (2012–2024)")
st.divider()

# ── Datos desde Supabase ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def obtener_datos():
    todos = []
    paso  = 1000
    inicio = 0
    while True:
        resp = supabase.table("calidad_agua").select(
            "id,estado,municipio,semaforo,alc_mg_l,conduct_ms_cm,sdt_mg_l,fluoruros_mg_l,dur_mg_l"
        ).range(inicio, inicio + paso - 1).execute()
        if not resp.data:
            break
        todos.extend(resp.data)
        if len(resp.data) < paso:
            break
        inicio += paso
    return pd.DataFrame(todos)

with st.spinner("Cargando datos desde Supabase..."):
    df = obtener_datos()

# ── KPIs ─────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total registros",   len(df))
col2.metric("Sitios VERDE 🟢",   len(df[df['semaforo'] == 'VERDE']))
col3.metric("Sitios AMARILLO 🟡", len(df[df['semaforo'] == 'AMARILLO']))
col4.metric("Sitios ROJO 🔴",    len(df[df['semaforo'] == 'ROJO']))

st.divider()

# ── Gráficas ──────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribución del semáforo de calidad")
    conteo = df['semaforo'].value_counts()
    colores = {'VERDE': 'seagreen', 'AMARILLO': 'gold', 'ROJO': 'tomato'}
    fig, ax = plt.subplots()
    ax.pie(conteo.values, labels=conteo.index, autopct='%1.1f%%',
           colors=[colores.get(c, 'gray') for c in conteo.index])
    st.pyplot(fig)

with col_b:
    st.subheader("Registros por estado")
    por_estado = df['estado'].value_counts().head(15)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    por_estado.sort_values().plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_xlabel("Cantidad de registros")
    st.pyplot(fig2)

st.divider()

# ── Predictor ────────────────────────────────────────────────────────────────
st.subheader("🔍 Predictor de calidad del agua")
st.markdown("Ingresa los parámetros de una muestra para predecir su calidad:")

col1, col2, col3, col4, col5 = st.columns(5)
alc         = col1.number_input("Alcalinidad",   value=0.0, format="%.3f")
conduct     = col2.number_input("Conductividad", value=0.0, format="%.3f")
sdt         = col3.number_input("SDT",           value=0.0, format="%.3f")
fluoruros   = col4.number_input("Fluoruros",     value=0.0, format="%.3f")
dur         = col5.number_input("Dureza",        value=0.0, format="%.3f")

if st.button("Predecir calidad"):
    muestra = np.array([[alc, conduct, sdt, fluoruros, dur]])
    pred    = modelo.predict(muestra)[0]
    proba   = modelo.predict_proba(muestra)[0]
    clases  = modelo.classes_

    if pred == 'VERDE':
        st.success(f"✅ Calidad: **{pred}** — Agua dentro de norma")
    elif pred == 'AMARILLO':
        st.warning(f"⚠️ Calidad: **{pred}** — Agua con parámetros límite")
    else:
        st.error(f"🚨 Calidad: **{pred}** — Agua fuera de norma")

    st.markdown("**Probabilidades por clase:**")
    for clase, p in sorted(zip(clases, proba), key=lambda x: -x[1]):
        st.progress(float(p), text=f"{clase}: {p*100:.1f}%")