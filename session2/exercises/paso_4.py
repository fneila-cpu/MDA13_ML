# ============================================================
# paso_4.py — Naive LLM: zero-shot scoring de un lead
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Ayer construimos cuatro herramientas. Hoy se las damos al LLM,
# en tres niveles. Empezamos por el primer nivel: **sin herramientas**.
#
# El LLM ve sólo la descripción libre del lead (`company_description`)
# y le pedimos un número 0-100. Sin features estructuradas, sin
# DataFrame, sin modelos entrenados. **Es el techo del LLM solo.**
#
# Por qué lo hacemos: para tener el SUELO de comparación. Cuando
# en paso_5 y paso_6 le demos herramientas al LLM, querrás saber
# cuánto suben respecto a este suelo. Sin esta medida, no sabes
# si darle herramientas vale la pena.
#
# ── Huecos ──────────────────────────────────────────────────
#
# TRES huecos marcados con `___`:
#   - HUECO 1: el prompt de scoring (sistema + user con la descripción)
#   - HUECO 2: la llamada a la API de OpenAI (chat.completions.create)
#   - HUECO 3: parsear el número del texto de respuesta
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session2/exercises/paso_4.py
#
# Necesitas .env con OPENAI_API_KEY válida.
#
# ============================================================

import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Cañadata — paso 4", page_icon="🤖", layout="wide")


# ── Carga (igual que paso_3, todo en cache) ───────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "canadata_leads_clean.csv")


@st.cache_resource
def load_classifier() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "classifier.pkl")


@st.cache_resource
def load_regressor() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "regressor.pkl")


@st.cache_resource
def load_clusterer() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "clusterer.pkl")


@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY no está. Crea `.env` en la raíz con la clave de Luis.")
        st.stop()
    return OpenAI(api_key=api_key)


# ── Helpers (de paso_3) ────────────────────────────────────

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


# ── LLM zero-shot: lo NUEVO ────────────────────────────────

MODEL = "gpt-4.1-mini"


# ── HUECO 1 ────────────────────────────────────────────────
# Construye el prompt de scoring. Le decimos al LLM:
#   - que es un analista comercial B2B,
#   - qué empresa estamos puntuando (la descripción),
#   - que devuelva SÓLO un número entero 0-100, sin explicación.
#
# Ejemplo (pista, pero úsalo a tu manera):
#   f"""Eres un analista comercial B2B en Cañadata, una SaaS de
#   gestión de pipeline. Lee la descripción y estima la probabilidad
#   (0-100) de que esta empresa se convierta en cliente.
#   Devuelve SÓLO el número entero, sin explicación.
#
#   Descripción: {description}"""
# ──────────────────────────────────────────────────────────
def build_scoring_prompt(description: str) -> str:
    return (
        f"Eres un analista comercial B2B en Cañadata, una SaaS de gestión de pipeline. "
        f"Lee la descripción de la empresa y estima la probabilidad (0-100) de que se convierta en cliente. "
        f"Devuelve SÓLO el número entero, sin explicación.\n\n"
        f"Descripción: {description}"
    )


# ── HUECO 2 ────────────────────────────────────────────────
# Llama a la API de OpenAI. Patrón:
#   response = _client.chat.completions.create(
#       model=MODEL,
#       messages=[{"role": "user", "content": prompt}],
#       max_tokens=10,
#   )
# Y luego: text = response.choices[0].message.content
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def llm_score_lead(_client, lead_id: str, description: str) -> int:
    """Cached por (lead_id, description). _client opta fuera del hash."""
    prompt = build_scoring_prompt(description)
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    text = response.choices[0].message.content

    match = re.search(r"\d+", text)
    score = int(match.group(0)) if match else -1
    return score


# ── App ────────────────────────────────────────────────────

df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
client = get_openai_client()

st.title("🤖 Cañadata — paso 4: clasificador clásico vs LLM zero-shot")
st.caption(f"{len(df):,} leads · clasificador `{classifier['kind']}` · LLM `{MODEL}`")

with st.sidebar:
    st.subheader("Selecciona un lead")
    lead_id = st.selectbox("lead_id", df["lead_id"].tolist())

lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

# Descripción visible primero — es lo que ve el LLM
st.subheader("Descripción de la empresa (lo único que verá el LLM)")
desc = lead.get("company_description") or "(sin descripción)"
st.info(f"_{desc}_")

# Cuatro paneles en columnas: clásicos + LLM zero-shot
col_clf, col_reg, col_clu, col_llm = st.columns(4)

with col_clf:
    st.subheader("🎯 Clasificador (clásico)")
    X_clf = build_X(lead, CLF_INPUT_COLS, classifier["feature_names"])
    proba_clf = classifier["model"].predict_proba(X_clf)[0, 1]
    st.metric("P(convertir)", f"{proba_clf:.1%}")
    st.caption("Entrenado sobre features estructuradas de 700 leads.")

with col_reg:
    st.subheader("💰 Regresión (clásica)")
    X_reg = build_X(lead, REG_INPUT_COLS, regressor["feature_names"])
    acv_pred = float(np.exp(regressor["model"].predict(X_reg))[0])
    st.metric("ACV predicho", f"{acv_pred:,.0f} €")

with col_clu:
    st.subheader("🔮 Cluster")
    X_clu = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
    cluster_id = int(clusterer["model"].predict(clusterer["scaler"].transform(X_clu))[0])
    st.metric("Cluster", f"#{cluster_id}")

with col_llm:
    st.subheader("🤖 LLM zero-shot")
    if not isinstance(desc, str) or not desc.strip() or desc == "(sin descripción)":
        st.warning("Sin descripción → no podemos preguntarle al LLM")
        proba_llm = None
    else:
        with st.spinner("Pregunto al LLM…"):
            score = llm_score_lead(client, lead_id, desc)
        if score < 0:
            st.error("No conseguí parsear un número. Revisa el HUECO 3.")
            proba_llm = None
        else:
            proba_llm = score / 100.0
            st.metric("P(convertir)", f"{score}%")
            st.caption(f"Sin entrenar. Sólo lee `company_description`.")

# Línea de comparación final
st.divider()
st.subheader("🪞 Lado a lado")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Clasificador entrenado", f"{proba_clf:.0%}")
col_b.metric("LLM zero-shot", f"{proba_llm:.0%}" if proba_llm is not None else "n/d")
col_c.metric("Realidad", "✓ convirtió" if lead["converted"] else "✗ no convirtió")

st.caption(
    f"Arquetipo plantado: `{lead['lead_segment_truth']}`. "
    "El clasificador acierta porque ha visto features. El LLM acierta (o falla) "
    "leyendo la descripción. Mira si están de acuerdo o si discrepan."
)

st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Cambia el lead** y observa cuándo el LLM y el clasificador discrepan más. ¿Hay un patrón? Pista: prueba con leads cuya descripción es vaga o está en inglés.
- **Mejora el prompt**: añade el contexto del negocio (precios, segmentos, qué hace que un lead sea bueno) y vuelve a comparar. ¿Mejora la correlación con el clasificador?
- **Modo confianza**: pídele al LLM que devuelva además su nivel de confianza ("alta/media/baja") y muéstralo. Útil para discutir cuándo confiar en el zero-shot.
- **Batch eval**: evalúa el LLM sobre 50 leads del holdout (`canadata_holdout.csv`) y compara su ROC-AUC con el del clasificador. Pista: ten cuidado con la tasa de llamadas (0.05 €/100 leads).
- **Coste real**: `response.usage` te da los tokens. Calcula el coste por lead a precios actuales de gpt-4.1-mini.
"""
)
