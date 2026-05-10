# 💧 Calidad del Agua Subterránea — CONAGUA (2012–2024)

Pipeline de datos end-to-end para el análisis, predicción y visualización de la calidad del agua subterránea en México, usando datos oficiales de la Comisión Nacional del Agua (CONAGUA).

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Supabase](https://img.shields.io/badge/Database-Supabase-3ECF8E?logo=supabase)](https://supabase.com)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

---

## Descripción

Este proyecto toma el dataset de calidad del agua subterránea de CONAGUA (2,728 registros, 41 variables) y construye un pipeline completo que:

1. **Limpia y transforma** los datos aplicando imputación de valores faltantes y estandarización Z-score
2. **Almacena** los datos procesados en una base de datos PostgreSQL en la nube (Supabase)
3. **Entrena un modelo de Machine Learning** (Random Forest) para predecir el semáforo de calidad del agua (VERDE / AMARILLO / ROJO)
4. **Despliega un dashboard web** interactivo en Streamlit con KPIs, mapas y predictor en tiempo real
5. **Visualiza en Power BI** con 4 páginas de análisis conectadas directamente a Supabase
6. **Automatiza** todo el proceso con GitHub Actions para actualización diaria

---

## Arquitectura

```
Datos CONAGUA (.xlsx)
        │
        ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  ETL Pipeline   │─────▶│  Supabase (Cloud  │─────▶│  Modelo ML      │
│  (Python)       │      │  PostgreSQL)      │      │  Random Forest  │
│  · Limpieza     │      │  2,728 registros  │      │  Precisión: 73% │
│  · Imputación   │      └──────────────────┘      └─────────────────┘
│  · Z-score      │               │                        │
└─────────────────┘               │                        │
        │                         ▼                        ▼
        │              ┌──────────────────┐      ┌─────────────────┐
        │              │  Streamlit Cloud  │      │  Power BI       │
        │              │  Dashboard        │      │  Dashboard      │
        │              │  (público)        │      │  (4 páginas)    │
        │              └──────────────────┘      └─────────────────┘
        │
        ▼
┌─────────────────┐
│  GitHub Actions │
│  (diario 6am)   │
└─────────────────┘
```

---
## Estructura del proyecto

```
agua-calidad-mx/
├── data/
│   ├── raw/                    ← Dataset original de CONAGUA (.xlsx)
│   └── processed/              ← Datos limpios
├── src/
│   ├── etl/
│   │   └── pipeline.py         ← Limpieza, imputación, escalamiento y carga a Supabase
│   ├── model/
│   │   ├── train.py            ← Entrenamiento del modelo Random Forest
│   │   └── predict.py          ← Predicción sobre nuevas muestras
│   └── dashboard/
│       └── app.py              ← App Streamlit
├── docs/
│   ├── CONAGUA INFORMES.pbix   ← Dashboard Power BI
│   └── images/                 ← Capturas del dashboard
├── .github/
│   └── workflows/
│       └── pipeline.yml        ← Automatización diaria con GitHub Actions
├── .env.example                ← Plantilla de variables de entorno
├── requirements.txt
└── README.md
```

---

## Tecnologías

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| ETL | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Base de datos | Supabase (PostgreSQL) |
| Dashboard web | Streamlit |
| Visualización BI | Power BI Desktop |
| Automatización | GitHub Actions |
| Control de versiones | Git / GitHub |

---

## Instalación y uso local

### 1. Clonar el repositorio
```bash
git clone https://github.com/kuisval/agua-calidad-mx.git
cd agua-calidad-mx
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
Copia `.env.example` a `.env` y llena tus credenciales:
```bash
cp .env.example .env
```
```env
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu-anon-key
DB_PASSWORD=tu-password
```

### 4. Ejecutar el pipeline ETL
```bash
python -m src.etl.pipeline
```

### 5. Entrenar el modelo
```bash
python -m src.model.train
```

### 6. Correr el dashboard localmente
```bash
python -m streamlit run src/dashboard/app.py
```

---

## Pipeline ETL

El script `src/etl/pipeline.py` realiza las siguientes transformaciones sobre el dataset de CONAGUA:

- **Carga** el archivo `.xlsx` desde `data/raw/`
- **Conversión de tipos** con `pd.to_numeric(errors='coerce')`
- **Imputación de valores faltantes:**
  - Variables numéricas → mediana (robusta ante outliers)
  - Variables categóricas → moda
- **Escalamiento Z-score** sobre las 5 variables numéricas principales
- **Carga a Supabase** en lotes de 500 registros
- Cada ejecución **agrega 10 registros nuevos simulados** para simular actualización diaria

---

## Modelo de Machine Learning

**Algoritmo:** Random Forest Classifier (`sklearn`)

**Variables de entrada (features):**
- `alc_mg_l` — Alcalinidad
- `conduct_ms_cm` — Conductividad
- `sdt_mg_l` — Sólidos disueltos totales
- `fluoruros_mg_l` — Fluoruros *(variable más importante: 34% de importancia)*
- `dur_mg_l` — Dureza

**Variable objetivo:** `semaforo` (VERDE / AMARILLO / ROJO)

**Resultados:**
| Clase | Precisión |
|---|---|
| VERDE | 81% |
| ROJO | 74% |
| AMARILLO | 60% |
| **General** | **73%** |

División: 80% entrenamiento / 20% prueba (`random_state=42`)

---

## Dashboard Streamlit

La app web muestra:
- **KPIs:** Total de registros, sitios en VERDE / AMARILLO / ROJO
- **Gráfica de pie:** Distribución del semáforo de calidad
- **Barras:** Registros por estado
- **Predictor en tiempo real:** Ingresa parámetros fisicoquímicos y predice la calidad

---

## Dashboard Power BI

El archivo `docs/CONAGUA INFORMES.pbix` conecta directamente a Supabase vía API REST e incluye 4 páginas:

| Página | Contenido |
|---|---|
| Resumen General | KPIs totales, sitios por estado, % sitios en rojo |
| Calidad por Parámetro | Distribución de las 5 categorías de calidad con filtros |
| Semáforo de Calidad | Mapa de México con puntos coloreados, barras apiladas por estado |
| Parámetros Fisicoquímicos | Promedios por estado, scatter plot fluoruros vs conductividad |

---

## Automatización — GitHub Actions

El workflow `.github/workflows/pipeline.yml` se ejecuta:
- **Automáticamente** todos los días a las 6:00 AM UTC
- **Manualmente** desde la pestaña Actions en GitHub

Pasos del workflow:
1. Clonar repositorio
2. Configurar Python 3.11
3. Instalar dependencias
4. Correr pipeline ETL (actualiza Supabase)
5. Reentrenar el modelo con datos nuevos

Las credenciales se pasan como **GitHub Secrets** (nunca se exponen en el código).

---

## Variables de entorno requeridas

| Variable | Descripción |
|---|---|
| `SUPABASE_URL` | URL del proyecto en Supabase |
| `SUPABASE_KEY` | Clave anon/public de Supabase |
| `DB_PASSWORD` | Contraseña de la base de datos |

---

## Dataset

**Fuente:** Comisión Nacional del Agua (CONAGUA) — México  
**Período:** 2012 – 2024  
**Registros:** 2,728  
**Variables:** 41 (parámetros fisicoquímicos, microbiológicos y metales pesados)  
**Sitios de monitoreo:** 33 estados, 713 municipios

---

## Autor

**José Luis Valenzuela Araujo**  
Estudiante de Ingeniería en Sistemas Computacionales — TECNM Campus Culiacán  
[LinkedIn](https://www.linkedin.com/in/jose-luis-valenzuela-araujo8b6196298) · joselvalenzuela04@gmail.com
