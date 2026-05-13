# ============================================================
# paso_6.py — LLM como operador del modelo entrenado
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# En paso_5 el LLM escribe pandas. Bien para análisis sobre los datos
# históricos. Pero si la pregunta es "¿qué probabilidad tiene este
# lead de convertir?", lo correcto NO es que el LLM escriba un
# clasificador inventado — es que cargue el clasificador entrenado
# en S1 y llame `predict_proba`.
#
# Aquí el LLM tiene acceso a TRES cosas:
#   1. el DataFrame `df` (como antes)
#   2. los modelos clásicos cargados (`classifier`, `regressor`,
#      `clusterer`, `timeseries_model`, `build_X`)
#   3. una guía sobre cómo usarlos
#
# Y termina con la VISTA DE COMPARACIÓN: misma pregunta, modo analista
# vs modo operador. Cuándo da igual, cuándo cambia la respuesta.
#
# ── Huecos ──────────────────────────────────────────────────
#
# CUATRO huecos marcados con `___`:
#   - HUECO 1: SYSTEM_PROMPT enriquecido con los modelos disponibles
#              y cómo invocarlos.
#   - HUECO 2: el namespace de exec() incluye los modelos cargados
#              (y la helper build_X).
#   - HUECO 3: la llamada al LLM (igual que en paso_5).
#   - HUECO 4: la vista de comparación: ejecuta la misma pregunta
#              en MODO_ANALISTA y MODO_OPERADOR y muestra los dos
#              resultados lado a lado.
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session2/exercises/paso_6.py
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

st.set_page_config(page_title="Cañadata — paso 6", page_icon="🧠", layout="wide")

MODEL = "gpt-4.1-mini"


# ── Carga ──────────────────────────────────────────────────

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
def load_timeseries() -> dict:
    return joblib.load(ROOT / "session1" / "models" / "timeseries.pkl")


@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY no está. Crea `.env` en la raíz con la clave de Luis.")
        st.stop()
    return OpenAI(api_key=api_key)


CLF_INPUT_COLS = [
    "industry", "company_size", "country", "source", "demo_requested",
    "emails_opened", "response_time_hours", "n_meetings",
    "decision_maker_contacted", "quoted_acv_eur",
]
REG_INPUT_COLS = [c for c in CLF_INPUT_COLS if c != "quoted_acv_eur"]


def build_X(lead: dict, columns: list[str], training_features: list[str]) -> pd.DataFrame:
    """Construye una matriz de features de 1 fila para un lead. El LLM puede llamarla."""
    d = pd.DataFrame([{k: lead[k] for k in columns}])
    cat_cols = [c for c in ["industry", "country", "source"] if c in columns]
    X = pd.get_dummies(d, columns=cat_cols)
    for col in ["demo_requested", "decision_maker_contacted"]:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X.reindex(columns=training_features, fill_value=0)


df = load_data()
classifier = load_classifier()
regressor = load_regressor()
clusterer = load_clusterer()
try:
    timeseries = load_timeseries()
except Exception as e:
    timeseries = None
    st.warning(f"⚠️ Series temporales no disponibles (Prophet): {e}")
client = get_openai_client()


# ── HUECO 1 ────────────────────────────────────────────────
# El SYSTEM_PROMPT del modo OPERADOR. Tiene que decirle al LLM:
#   - que es un asistente con acceso a `df` Y a 4 modelos entrenados,
#   - cómo invocar cada modelo,
#   - cuándo usar `df` puro vs cuándo usar los modelos entrenados,
#   - que el código debe terminar con `resultado = ...`.
#
# Plantilla:
# """Eres un analista que tiene acceso a:
#   - `df` (DataFrame de Cañadata, columnas: ...)
#   - `classifier` (dict con keys 'model', 'feature_names'). Para
#     predecir conversión sobre 1 lead:
#         X = build_X(lead_dict, CLF_INPUT_COLS, classifier['feature_names'])
#         proba = classifier['model'].predict_proba(X)[0, 1]
#   - `regressor` (igual estructura, target en log space; recuerda np.exp).
#   - `clusterer` (dict con 'model', 'scaler', 'feature_names').
#   - `timeseries` (dict con 'model' (Prophet entrenado), 'history' (Series mensual),
#     'validation_mape'). Para forecast:
#         future = timeseries['model'].make_future_dataframe(periods=N, freq='MS')
#         fc = timeseries['model'].predict(future)
#         forecast = fc.set_index('ds')['yhat'].iloc[-N:]
#   - `build_X(lead_dict, columns, training_features)` helper.
#   - `CLF_INPUT_COLS`, `REG_INPUT_COLS` constantes.
#
# Reglas:
#   - Para "¿qué probabilidad tiene este lead de convertir?" usa el clasificador.
#   - Para "¿cuál es la tasa de conversión por industria?" usa `df`.
#   - Para forecast usa timeseries.
#   - Termina con `resultado = ...`.
#   - Devuelve SÓLO ```python ... ```."""
# ──────────────────────────────────────────────────────────
SYSTEM_PROMPT_OPERADOR = """Eres un analista que tiene acceso a:
- `df` (DataFrame de Cañadata, columnas: lead_id, company_name, industry, company_size, country, signup_date, source, demo_requested, emails_opened, response_time_hours, n_meetings, decision_maker_contacted, quoted_acv_eur, company_description, converted, converted_within_days)
- `classifier` (dict con keys 'model', 'feature_names'). Para predecir conversión sobre 1 lead:
      lead_dict = df[df['lead_id'] == 'LXXXX'].iloc[0].to_dict()
      X = build_X(lead_dict, CLF_INPUT_COLS, classifier['feature_names'])
      proba = classifier['model'].predict_proba(X)[0, 1]
- `regressor` (igual estructura, target en log space; usa np.exp para convertir a euros).
- `clusterer` (dict con 'model', 'scaler', 'feature_names'). Para asignar cluster:
      X = build_X(lead_dict, REG_INPUT_COLS, clusterer['feature_names'])
      cluster_id = clusterer['model'].predict(clusterer['scaler'].transform(X))[0]
- `timeseries` (dict con 'model' (Prophet), 'history', 'validation_mape'). Para forecast:
      future = timeseries['model'].make_future_dataframe(periods=N, freq='MS')
      fc = timeseries['model'].predict(future)
      resultado = fc.set_index('ds')['yhat'].iloc[-N:]
- `build_X(lead_dict, columns, training_features)` helper para construir la matriz de features.
- `CLF_INPUT_COLS`, `REG_INPUT_COLS` listas de columnas para cada modelo.

Reglas:
- Para preguntas sobre probabilidad de conversión de un lead concreto: usa el clasificador.
- Para preguntas sobre tasas, distribuciones o estadísticas históricas: usa `df`.
- Para predicciones de ACV: usa el regresor (recuerda np.exp).
- Para preguntas de forecast futuro: usa timeseries.
- Para asignar un lead a un cluster: usa el clusterer.
- Termina asignando el resultado a una variable llamada `resultado`.
- Devuelve SÓLO un bloque ```python ... ```, sin explicación."""


# Mismo prompt simplificado para el modo ANALISTA (sólo `df`, sin modelos)
SYSTEM_PROMPT_ANALISTA = (
    "Eres un analista de datos. Tienes un DataFrame de pandas llamado `df` "
    "con datos de leads B2B (columnas: industry, company_size, country, "
    "source, demo_requested, emails_opened, response_time_hours, "
    "n_meetings, decision_maker_contacted, quoted_acv_eur, converted, ...).\n\n"
    "Genera código Python pandas para responder la pregunta del usuario.\n"
    "Termina asignando el resultado a una variable llamada `resultado`.\n"
    "Devuelve SÓLO un bloque ```python ... ```, sin explicación."
)


def ask_llm_for_code(pregunta: str, system_prompt: str) -> str:
    # ── HUECO 3 ────────────────────────────────────────────
    # Llama al LLM con system + user. Igual que paso_5.
    # ──────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pregunta},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    return text


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text.strip()


def run_code(code: str, mode: str):
    """Ejecuta `code` en un namespace que depende del modo."""
    # ── HUECO 2 ────────────────────────────────────────────
    # En modo "analista": ns = {"df": df, "pd": pd, "np": np}
    # En modo "operador": añade al ns: classifier, regressor, clusterer,
    #     timeseries, build_X, CLF_INPUT_COLS, REG_INPUT_COLS.
    # Luego: exec(code, ns); return ns.get("resultado", ...)
    # ──────────────────────────────────────────────────────
    ns = {"df": df, "pd": pd, "np": np}
    if mode == "operador":
        ns.update({
            "classifier": classifier,
            "regressor": regressor,
            "clusterer": clusterer,
            "timeseries": timeseries,
            "build_X": build_X,
            "CLF_INPUT_COLS": CLF_INPUT_COLS,
            "REG_INPUT_COLS": REG_INPUT_COLS,
        })

    try:
        exec(code, ns)
        return ns.get("resultado", "(no se asignó `resultado`)"), None
    except Exception as e:
        return None, str(e)


# ── App ────────────────────────────────────────────────────

st.title("🧠 Cañadata — paso 6: LLM operador del modelo entrenado")
st.caption(
    f"{len(df):,} leads · 4 modelos clásicos disponibles · modelo LLM `{MODEL}`"
)

st.subheader("Hazle una pregunta")
st.caption(
    "El LLM puede usar `df` para análisis Y los 4 modelos entrenados para "
    "predicciones individuales. Comparamos modo analista (sólo df) vs modo operador "
    "(df + modelos)."
)

ejemplos = [
    "¿Cuál es la probabilidad de conversión del lead L0001 según el clasificador?",
    "¿Cuál es la tasa de conversión por industria en df?",
    "Predice el ACV de un lead 'fintech' de 200 empleados con 4 reuniones.",
    "Forecast de conversiones para los próximos 4 meses.",
    "¿En qué cluster cae el lead L0050?",
]

if "pregunta_guardada_6" not in st.session_state:
    st.session_state["pregunta_guardada_6"] = ""

cols = st.columns(len(ejemplos))
for i, ej in enumerate(ejemplos):
    if cols[i].button(ej, key=f"ej_{i}", use_container_width=True):
        st.session_state["pregunta_guardada_6"] = ej

pregunta = st.text_area("O escribe la tuya:", key="pregunta_guardada_6", height=80)


if st.button("Preguntar (modo OPERADOR)", type="primary", disabled=not pregunta.strip()):
    with st.spinner("LLM redactando código (modo operador)…"):
        raw = ask_llm_for_code(pregunta, SYSTEM_PROMPT_OPERADOR)
        code = extract_code(raw)
    with st.expander("👁 Código (operador)"):
        st.code(code, language="python")
    resultado, error = run_code(code, "operador")
    if error:
        st.error(f"Error: {error}")
    else:
        st.subheader("Resultado (operador)")
        if isinstance(resultado, (pd.DataFrame, pd.Series)):
            st.dataframe(resultado, use_container_width=True)
        else:
            st.write(resultado)


# ── HUECO 4 ────────────────────────────────────────────────
# Vista de comparación: corre la misma pregunta en MODO_ANALISTA y
# en MODO_OPERADOR, lado a lado, y muestra ambos resultados.
#
# Patrón:
#   if st.button("Comparar modos", disabled=not pregunta.strip()):
#       col_an, col_op = st.columns(2)
#       with col_an:
#           st.markdown("### Modo analista (sólo df)")
#           code_an = extract_code(ask_llm_for_code(pregunta, SYSTEM_PROMPT_ANALISTA))
#           res_an, err_an = run_code(code_an, "analista")
#           # render
#       with col_op:
#           st.markdown("### Modo operador (df + modelos)")
#           code_op = extract_code(ask_llm_for_code(pregunta, SYSTEM_PROMPT_OPERADOR))
#           res_op, err_op = run_code(code_op, "operador")
#           # render
# ──────────────────────────────────────────────────────────
if st.button("Comparar modos", disabled=not pregunta.strip()):
    col_an, col_op = st.columns(2)
    with col_an:
        st.markdown("### Modo analista (sólo df)")
        with st.spinner("LLM modo analista…"):
            code_an = extract_code(ask_llm_for_code(pregunta, SYSTEM_PROMPT_ANALISTA))
        with st.expander("👁 Código (analista)"):
            st.code(code_an, language="python")
        res_an, err_an = run_code(code_an, "analista")
        if err_an:
            st.error(f"Error: {err_an}")
        elif isinstance(res_an, (pd.DataFrame, pd.Series)):
            st.dataframe(res_an, use_container_width=True)
        else:
            st.write(res_an)
    with col_op:
        st.markdown("### Modo operador (df + modelos)")
        with st.spinner("LLM modo operador…"):
            code_op = extract_code(ask_llm_for_code(pregunta, SYSTEM_PROMPT_OPERADOR))
        with st.expander("👁 Código (operador)"):
            st.code(code_op, language="python")
        res_op, err_op = run_code(code_op, "operador")
        if err_op:
            st.error(f"Error: {err_op}")
        elif isinstance(res_op, (pd.DataFrame, pd.Series)):
            st.dataframe(res_op, use_container_width=True)
        else:
            st.write(res_op)


st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Tres modos en paralelo**: añade el modo NAIVE (zero-shot del paso_4) a la comparación. Para cada pregunta verás 3 respuestas + (cuando aplica) la respuesta clásica directa.
- **Sándbox real**: en vez de exec(), usa `subprocess.run([sys.executable, "-c", code])` con timeout. Más seguro frente a código malicioso/erróneo.
- **Function calling**: re-implementa esto con OpenAI function calling tipado (`predict_conversion(lead_dict)`, `forecast(months: int)`). Compara robustez vs text-to-code.
- **Latency + coste**: mide tiempo y tokens en cada modo. ¿Cuál es el más caro? ¿Vale la pena?
- **Memoria conversacional**: el modo operador con memoria es más útil para iterar ("y para fintech?").
- **Validación cruzada**: para una pregunta numérica, ejecuta el modo operador 3 veces. ¿Da el mismo número? Si no, hay no determinismo en el código generado (aunque temperature=0).
"""
)
