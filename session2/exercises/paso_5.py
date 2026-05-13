# ============================================================
# paso_5.py — LLM como analista (text-to-code sobre los datos)
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# El LLM zero-shot del paso_4 sólo veía 1-2 frases. Bien para una
# corazonada, mal para "¿cuál es la conversión por industria?".
#
# Aquí el LLM va a ESCRIBIR código pandas que ejecutamos contra el
# DataFrame real de Cañadata. La pregunta del usuario en español →
# código Python → resultado real.
#
# Es el patrón text-to-code del taller anterior FitLife, aplicado
# a un dataset distinto.
#
# ── Huecos ──────────────────────────────────────────────────
#
# CUATRO huecos marcados con `___`:
#   - HUECO 1: SYSTEM_PROMPT con el schema del DataFrame
#   - HUECO 2: la llamada a la API para generar código
#   - HUECO 3: extraer el código del bloque markdown que devuelve el LLM
#   - HUECO 4: ejecutar el código contra `df` con exec() y capturar
#              el resultado en una variable `resultado`
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session2/exercises/paso_5.py
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

st.set_page_config(page_title="Cañadata — paso 5", page_icon="🧮", layout="wide")

MODEL = "gpt-4.1-mini"


# ── Carga ──────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "canadata_leads_clean.csv")


@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY no está. Crea `.env` en la raíz con la clave de Luis.")
        st.stop()
    return OpenAI(api_key=api_key)


df = load_data()
client = get_openai_client()


# ── HUECO 1 ────────────────────────────────────────────────
# El prompt de sistema. Tiene que decirle al LLM:
#   - que es un asistente de análisis sobre un DataFrame `df`
#   - el schema (columnas + tipos)
#   - que el código debe terminar con `resultado = ...`
#   - que devuelva SÓLO un bloque ```python ... ```
#
# Pista: Cañadata tiene estas columnas (omite el target oculto):
#   lead_id (str), company_name (str), industry (str ∈ {SaaS, fintech,
#   retail, logistics, healthcare, unknown}), company_size (int),
#   country (str ∈ {ES, FR, DE, UK, IT, PT, unknown}), signup_date
#   (datetime-string YYYY-MM-DD), source (str ∈ {organic, paid, referral,
#   conference, outbound, unknown}), demo_requested (bool),
#   emails_opened (int), response_time_hours (float), n_meetings (int),
#   decision_maker_contacted (bool), quoted_acv_eur (float),
#   company_description (str), converted (bool), converted_within_days
#   (float, NaN si no convirtió).
#
# Plantilla mínima:
#   """Eres un analista de datos. Tienes un DataFrame de pandas
#   llamado `df` con estas columnas: ...
#   Genera código Python pandas para responder la pregunta del usuario.
#   Termina asignando el resultado a una variable llamada `resultado`.
#   Devuelve SÓLO un bloque ```python ... ```, sin explicación."""
# ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista de datos. Tienes un DataFrame de pandas llamado `df` con estas columnas:
- lead_id (str): identificador único del lead
- company_name (str): nombre de la empresa
- industry (str): sector — SaaS, fintech, retail, logistics, healthcare, unknown
- company_size (int): número de empleados
- country (str): ES, FR, DE, UK, IT, PT, unknown
- signup_date (str): fecha de registro, formato YYYY-MM-DD
- source (str): canal de adquisición — organic, paid, referral, conference, outbound, unknown
- demo_requested (bool): si solicitó demo
- emails_opened (int): emails abiertos
- response_time_hours (float): horas hasta respuesta
- n_meetings (int): número de reuniones
- decision_maker_contacted (bool): si se contactó al decisor
- quoted_acv_eur (float): ACV cotizado en euros
- company_description (str): descripción libre de la empresa
- converted (bool): si se convirtió en cliente
- converted_within_days (float): días hasta conversión, NaN si no convirtió

Genera código Python pandas para responder la pregunta del usuario.
Termina asignando el resultado a una variable llamada `resultado`.
Devuelve SÓLO un bloque ```python ... ```, sin explicación."""


# ── HUECO 2 ────────────────────────────────────────────────
# Llama al LLM con system + user. Patrón:
#   response = client.chat.completions.create(
#       model=MODEL,
#       messages=[
#           {"role": "system", "content": SYSTEM_PROMPT},
#           {"role": "user", "content": pregunta},
#       ],
#       temperature=0.0,   # queremos código determinista
#   )
# Devuelve response.choices[0].message.content.
# ──────────────────────────────────────────────────────────
def ask_llm_for_code(pregunta: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pregunta},
        ],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    return text


# ── HUECO 3 ────────────────────────────────────────────────
# El LLM devuelve algo así:
#   ```python
#   resultado = df.groupby('industry')['converted'].mean()
#   ```
# Saca SÓLO el código de dentro de los backticks.
# Pista: re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
#        y luego .group(1) si encuentra match.
# Si no encuentra, devuelve el texto pelado (puede que el LLM no haya
# usado backticks).
# ──────────────────────────────────────────────────────────
def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    return code


# ── HUECO 4 ────────────────────────────────────────────────
# Ejecuta el código en un namespace controlado y devuelve el valor
# de `resultado`. Patrón:
#   ns = {"df": df, "pd": pd, "np": np}
#   exec(code, ns)
#   return ns.get("resultado", "(no se asignó `resultado`)")
#
# Envuélvelo en try/except para que un fallo no rompa la app.
# ──────────────────────────────────────────────────────────
def run_code(code: str, df: pd.DataFrame):
    try:
        ns = {"df": df, "pd": pd, "np": np}
        exec(code, ns)
        result = ns.get("resultado", "(no se asignó `resultado`)")
        return result, None
    except Exception as e:
        return None, str(e)


# ── App ────────────────────────────────────────────────────

st.title("🧮 Cañadata — paso 5: el LLM como analista (text-to-code)")
st.caption(f"{len(df):,} leads · {len(df.columns)} columnas · modelo `{MODEL}`")

with st.expander("📋 Schema del DataFrame `df` (lo que ve el LLM)"):
    st.dataframe(df.dtypes.rename("dtype").to_frame(), use_container_width=True)

# Ejemplos clicables
st.subheader("Hazle una pregunta a los datos")

ejemplos = [
    "¿Cuál es la tasa de conversión por industria?",
    "¿Cuál es el ACV medio por arquetipo (lead_segment_truth)?",
    "Top 10 leads de mayor tamaño que NO convirtieron.",
    "¿Cómo varía la conversión entre los que pidieron demo vs los que no?",
    "Distribución de conversiones por mes de signup.",
]

if "pregunta_guardada" not in st.session_state:
    st.session_state["pregunta_guardada"] = ""

cols = st.columns(len(ejemplos))
for i, ej in enumerate(ejemplos):
    if cols[i].button(ej, key=f"ej_{i}", use_container_width=True):
        st.session_state["pregunta_guardada"] = ej

pregunta = st.text_area("O escribe la tuya:", key="pregunta_guardada", height=80)

if st.button("Preguntar", type="primary", disabled=not pregunta.strip()):
    with st.spinner("LLM redactando código…"):
        raw = ask_llm_for_code(pregunta)
        code = extract_code(raw)

    with st.expander("👁 Código que generó el LLM"):
        st.code(code, language="python")

    resultado, error = run_code(code, df)

    if error:
        st.error(f"Error al ejecutar el código:\n\n{error}")
        st.caption("El LLM puede equivocarse. Mira el código arriba — ¿la columna existe? ¿el filtro tiene sentido?")
    else:
        st.subheader("Resultado")
        if isinstance(resultado, pd.DataFrame):
            st.dataframe(resultado, use_container_width=True)
        elif isinstance(resultado, pd.Series):
            st.dataframe(resultado.to_frame("valor"), use_container_width=True)
        else:
            st.write(resultado)

st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Memoria conversacional**: en vez de una pregunta única, mantén el historial en `st.session_state.messages` y deja que el alumno haga preguntas de seguimiento ("y filtrado por España"). Pista: pasa el historial al LLM como `messages`.
- **Auto-reparación**: si el código falla, vuelve a llamar al LLM con el error y pídele que corrija. Hasta 2 intentos. Es el patrón paso_11 de FitLife.
- **Validación del resultado**: añade reglas de sanidad (¿el porcentaje está entre 0 y 100? ¿la cuenta de filas tiene sentido?). Si no, avisa al usuario.
- **Restricciones de seguridad**: bloquea `import`, `open`, `os.system` en el código antes de exec(). En este taller no hace falta porque corremos local, en prod sí.
- **Few-shot**: añade 2-3 ejemplos de pregunta → código al system prompt. Mejora mucho la calidad sobre preguntas que requieren cruzar columnas.
"""
)
