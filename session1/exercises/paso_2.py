# ============================================================
# paso_2.py — Añade regresión y clustering al dashboard
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Ya tienes el clasificador funcionando del paso_1. Ahora vas
# a añadir DOS paneles más al mismo dashboard:
#
#   1. Regresión: predice el ACV cotizado del lead seleccionado.
#   2. Clustering: dice a qué cluster pertenece el lead.
#
# El esqueleto del clasificador (paso_1) ya está rellenado.
# Tú te encargas de los HUECOS de regresión y clustering.
#
# ── Huecos ──────────────────────────────────────────────────
#
# Hay CUATRO huecos marcados con `___`:
#   - HUECO 1: cargar el regressor.pkl
#   - HUECO 2: predecir el ACV (recuerda: el target estaba en
#     log space, hay que hacer np.exp para volver a euros).
#   - HUECO 3: cargar el clusterer.pkl
#   - HUECO 4: predecir el cluster (recuerda: hay que escalar
#     primero con el StandardScaler que viene en el pkl).
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session1/exercises/paso_2.py
#
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Cañadata — paso 2", page_icon="🎯", layout="wide")


# ── Carga de datos y modelos ───────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "canadata_leads_clean.csv")


@st.cache_resource
def load_classifier() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "classifier.pkl")


@st.cache_resource
def load_regressor() -> dict:
    # ── HUECO 1 ────────────────────────────────────────────
    # Carga ROOT / "session1" / "models" / "regressor.pkl"
    # con joblib.load(...).
    # ──────────────────────────────────────────────────────
    return joblib.load(ROOT / "session1" / "models" / "regressor.pkl")


@st.cache_resource
def load_clusterer() -> dict:
    # ── HUECO 3 ────────────────────────────────────────────
    # Carga ROOT / "session1" / "models" / "clusterer.pkl"
    # con joblib.load(...).
    # ──────────────────────────────────────────────────────
    return joblib.load(ROOT / "session1" / "models" / "clusterer.pkl")


# ── Helper: construye features (igual que en paso_1) ───────

CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]

REG_INPUT_COLS = [c for c in CLF_INPUT_COLS if c != "quoted_acv_eur"]


def build_X(lead: dict, columns: list[str], training_features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{k: lead[k] for k in columns}])
    cat_cols = [c for c in ["industry", "country", "source"] if c in columns]
    X = pd.get_dummies(df, columns=cat_cols)
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


# ── App ────────────────────────────────────────────────────

df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()

st.title("🎯 Cañadata — paso 2: clasificador + regresor + clusterer")
st.caption(f"{len(df):,} leads · 3 modelos servidos")

with st.sidebar:
    st.subheader("Selecciona un lead")
    lead_id = st.selectbox("lead_id", df["lead_id"].tolist())

lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

# ── Datos del lead ─────────────────────────────────────────
st.subheader("Datos del lead")
visible = ["company_name", "industry", "company_size", "country", "source",
           "demo_requested", "emails_opened", "response_time_hours",
           "n_meetings", "decision_maker_contacted", "quoted_acv_eur"]
st.json({k: lead[k] for k in visible})

# ── Tres paneles en columnas ───────────────────────────────
col_clf, col_reg, col_clu = st.columns(3)

# ── Clasificación (igual que paso_1) ──────────────────────
with col_clf:
    st.subheader("🎯 Clasificación")
    X_clf = build_X(lead, CLF_INPUT_COLS, classifier["feature_names"])
    proba = classifier["model"].predict_proba(X_clf)[0, 1]
    st.metric("P(convertir)", f"{proba:.1%}")
    st.caption(f"realidad: `{lead['converted']}`")

# ── Regresión (NUEVO) ──────────────────────────────────────
with col_reg:
    st.subheader("💰 Regresión (ACV)")
    X_reg = build_X(lead, REG_INPUT_COLS, regressor["feature_names"])

    # ── HUECO 2 ────────────────────────────────────────────
    # El regressor predice en log space (el warm-up entrenó
    # con `np.log(y_reg)`). Para volver a euros:
    #   1. predicción_log = regressor["model"].predict(X_reg)
    #   2. acv_pred = np.exp(predicción_log)[0]
    # Concéntralo en una sola línea o dos.
    # ──────────────────────────────────────────────────────
    acv_pred = np.exp(regressor["model"].predict(X_reg))[0]

    st.metric("ACV predicho", f"{acv_pred:,.0f} €")
    st.caption(f"cotizado real: `{lead['quoted_acv_eur']:,.0f} €`")

# ── Clustering (NUEVO) ─────────────────────────────────────
with col_clu:
    st.subheader("🔮 Clustering")
    # El clusterer guardó su scaler dentro del pkl
    X_clu = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])

    # ── HUECO 4 ────────────────────────────────────────────
    # El clusterer.pkl guarda dos cosas:
    #   - clusterer["scaler"] (un StandardScaler entrenado)
    #   - clusterer["model"] (un KMeans entrenado)
    #
    # Para predecir el cluster de UN lead:
    #   1. X_scaled = clusterer["scaler"].transform(X_clu)
    #   2. cluster_id = clusterer["model"].predict(X_scaled)[0]
    # ──────────────────────────────────────────────────────
    cluster_id = clusterer["model"].predict(clusterer["scaler"].transform(X_clu))[0]

    st.metric("Cluster", f"#{cluster_id}")
    st.caption(f"arquetipo plantado: `{lead['lead_segment_truth']}`")
    st.caption("(El cluster es solo un número. Mapearlo a un arquetipo lo dejamos para el reto 🚀)")

st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Mapea cluster → arquetipo**: para cada cluster id, calcula el arquetipo más frecuente entre todos los leads que cae en ese cluster, y muestra el nombre en vez del número. Pista: predict sobre todo `df` y haz un cross-tab con `lead_segment_truth`.
- **Banda de incertidumbre del ACV**: en vez del punto, muestra un intervalo (la `+- 1 std` de los árboles del RandomForest). Pista: `regressor["model"].estimators_` te da los 200 árboles individuales.
- **Top features que importaron en la predicción de conversión**: para el lead seleccionado, qué features pesaron más. Pista: usa `shap` o un fallback simple comparando la predicción del lead con la del lead "promedio".
- **Filtro inverso**: añade un selector de cluster en el sidebar y muestra todos los leads que caen en ese cluster.
"""
)
