# ============================================================
# paso_3.py — Cierra el dashboard con series temporales
# ============================================================
#
# ── Reto ────────────────────────────────────────────────────
#
# Tienes el dashboard del paso_2 con 3 paneles (clasificación,
# regresión, clustering). Falta el cuarto: series temporales.
#
# Vas a añadir:
#   1. Una sección "Histórico mensual de conversiones" que
#      muestra la serie histórica que aprendió el modelo.
#   2. Un forecast para los próximos N meses, con N controlado
#      por un slider.
#
# Esta es la parte INDEPENDIENTE de la sesión: prácticamente
# todo es nuevo. Yo paso a echar un cable cuando me llames y
# al final enseño cómo debería quedar.
#
# ── Huecos ──────────────────────────────────────────────────
#
# SEIS huecos marcados con `___`:
#   - HUECO 1: cargar timeseries.pkl
#   - HUECO 2: extraer la serie histórica del pkl
#   - HUECO 3: extraer el modelo Prophet del pkl
#   - HUECO 4: extraer la serie de forecast (sólo yhat) del DataFrame fc
#   - HUECO 5: el bridge para que histórico y forecast conecten visualmente
#   - HUECO 6: el cono de incertidumbre (banda yhat_lower / yhat_upper en plotly)
#
# ── Cómo ejecutar ──────────────────────────────────────────
#
#   streamlit run session1/exercises/paso_3.py
#
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Cañadata — paso 3", page_icon="🎯", layout="wide")


# ── Carga de todo ──────────────────────────────────────────

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
    # ── HUECO 1 ────────────────────────────────────────────
    # Carga ROOT / "session1" / "models" / "timeseries.pkl"
    # con joblib.load(...).
    # El pkl tiene tres cosas: 'model' (Prophet ya entrenado),
    # 'history' (pd.Series con la historia mensual) y
    # 'validation_mape' (MAPE en hold-out de los últimos 3 meses).
    # ──────────────────────────────────────────────────────
    return joblib.load(ROOT / "session1" / "models" / "timeseries.pkl")


# ── Helpers (igual que en paso_2) ──────────────────────────

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
ts = load_timeseries()

st.title("🎯 Cañadata — paso 3: dashboard completo")
st.caption(f"{len(df):,} leads · 4 modelos servidos · ML clásico end-to-end")

with st.sidebar:
    st.subheader("Lead")
    lead_id = st.selectbox("lead_id", df["lead_id"].tolist())
    st.divider()
    st.subheader("Forecast")
    forecast_months = st.slider("Meses a predecir", min_value=1, max_value=12, value=6)

lead = df[df["lead_id"] == lead_id].iloc[0].to_dict()

# ── Datos del lead ─────────────────────────────────────────
st.subheader("Datos del lead")
visible = ["company_name", "industry", "company_size", "country", "source",
           "demo_requested", "emails_opened", "response_time_hours",
           "n_meetings", "decision_maker_contacted", "quoted_acv_eur"]
st.json({k: lead[k] for k in visible})

# ── Tres paneles del paso_2 ────────────────────────────────
col_clf, col_reg, col_clu = st.columns(3)

with col_clf:
    st.subheader("🎯 Clasificación")
    X_clf = build_X(lead, CLF_INPUT_COLS, classifier["feature_names"])
    proba = classifier["model"].predict_proba(X_clf)[0, 1]
    st.metric("P(convertir)", f"{proba:.1%}")
    st.caption(f"realidad: `{lead['converted']}`")

with col_reg:
    st.subheader("💰 Regresión (ACV)")
    X_reg = build_X(lead, REG_INPUT_COLS, regressor["feature_names"])
    acv_pred = np.exp(regressor["model"].predict(X_reg))[0]
    st.metric("ACV predicho", f"{acv_pred:,.0f} €")
    st.caption(f"cotizado real: `{lead['quoted_acv_eur']:,.0f} €`")

with col_clu:
    st.subheader("🔮 Clustering")
    X_clu = build_X(lead, REG_INPUT_COLS, clusterer["feature_names"])
    X_scaled = clusterer["scaler"].transform(X_clu)
    cluster_id = clusterer["model"].predict(X_scaled)[0]
    st.metric("Cluster", f"#{cluster_id}")
    st.caption(f"arquetipo plantado: `{lead['lead_segment_truth']}`")

# ── Series temporales (NUEVO) ──────────────────────────────
st.divider()
st.subheader("📈 Histórico de conversiones + forecast")

# ── HUECO 2 ────────────────────────────────────────────────
# Saca la historia mensual del pkl. Es una pd.Series con
# index de fechas mensuales y valores = nº de conversiones.
#   ts["history"]
# ──────────────────────────────────────────────────────────
history: pd.Series = ts["history"]

# ── HUECO 3 ────────────────────────────────────────────────
# Saca el modelo Prophet del pkl.
#   ts["model"]  →  Prophet ya entrenado
# ──────────────────────────────────────────────────────────
ts_model = ts["model"]

# Estos dos pasos están pre-rellenados — son mecánicos y no son la lección.
# `fc` es el DataFrame completo que devuelve Prophet, con columnas
# ds, yhat, yhat_lower, yhat_upper, trend, ...
future = ts_model.make_future_dataframe(periods=forecast_months, freq="MS")
fc = ts_model.predict(future)

# ── HUECO 4 ────────────────────────────────────────────────
# Saca la serie de forecast (sólo los valores centrales de yhat, los
# últimos forecast_months meses) como pd.Series con index de fechas.
#
# Pista — UNA LÍNEA:
#   forecast = fc.set_index("ds")["yhat"].iloc[-forecast_months:]
# ──────────────────────────────────────────────────────────
forecast: pd.Series = fc.set_index("ds")["yhat"].iloc[-forecast_months:]

# ── HUECO 5 ────────────────────────────────────────────────
# El DataFrame ya está montado. Lo que falta es el **bridge**: una
# línea que conecte el último valor histórico con el primero del
# forecast. Sin ella, la gráfica tiene un hueco visual (porque la
# última fecha histórica tiene forecast=NaN y la primera fecha del
# forecast tiene histórico=NaN).
#
# Pista — UNA LÍNEA exacta:
#
#   chart_df.loc[history.index[-1], "forecast"] = history.iloc[-1]
#
# Repite el último histórico en la columna forecast en esa misma
# fecha. Streamlit dibuja las dos columnas y la transición queda
# continua.
# ──────────────────────────────────────────────────────────
chart_df = pd.DataFrame({"histórico": history, "forecast": forecast})
chart_df.loc[history.index[-1], "forecast"] = history.iloc[-1]

st.line_chart(chart_df, height=300)
st.caption(
    f"Historia: {len(history)} meses · "
    f"Forecast: {forecast_months} meses · "
    f"Total proyectado: {forecast.sum():.0f} conversiones · "
    f"MAPE en hold-out: {ts.get('validation_mape', 0):.1f}%"
)

# ── El cono de incertidumbre ──────────────────────────────
#
# Una predicción de futuro NO es una línea, es un cono. La línea
# de arriba era el centro. Cuanto más lejos miras, más se abre el
# cono. Prophet ya te calculó los bordes en `fc["yhat_lower"]` y
# `fc["yhat_upper"]` (intervalo al 80% por defecto). Sólo hay que
# pintarlos.
st.divider()
st.subheader("🎯 El cono de incertidumbre")

forecast_dates = forecast.index
yhat_upper = fc.set_index("ds")["yhat_upper"].iloc[-forecast_months:]
yhat_lower = fc.set_index("ds")["yhat_lower"].iloc[-forecast_months:]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history.index, y=history.values,
    name="histórico", line=dict(color="steelblue"),
))
fig.add_trace(go.Scatter(
    x=forecast_dates, y=forecast.values,
    name="forecast (yhat)", line=dict(color="darkorange"),
))

# ── HUECO 6 ────────────────────────────────────────────────
# Falta la traza que dibuja la BANDA (el cono). Plotly hace esto
# con un único Scatter de fill="toself" y un truco: la x va "ida y
# vuelta" (fechas hacia adelante + las MISMAS fechas hacia atrás), y
# la y va "upper hacia adelante + lower hacia atrás". Cuando rellenas
# con `fill="toself"`, queda un polígono cerrado entre las dos curvas.
#
# Pista — copia esta traza tal cual:
#
#   fig.add_trace(go.Scatter(
#       x=list(forecast_dates) + list(forecast_dates[::-1]),
#       y=list(yhat_upper) + list(yhat_lower[::-1]),
#       fill="toself",
#       fillcolor="rgba(255,165,0,0.15)",
#       line=dict(color="rgba(0,0,0,0)"),
#       name="banda 80%",
#       showlegend=True,
#   ))
# ──────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=list(forecast_dates) + list(forecast_dates[::-1]),
    y=list(yhat_upper) + list(yhat_lower[::-1]),
    fill="toself",
    fillcolor="rgba(255,165,0,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="banda 80%",
    showlegend=True,
))

st.plotly_chart(fig, use_container_width=True)
st.caption(
    "El cono se abre con el horizonte: a 1 mes vista el modelo es "
    "razonablemente seguro, a 12 meses vista admite que no lo sabe. "
    "Si el cono es estrecho todo el rato, el modelo está sobreseguro "
    "(probablemente mal calibrado). Si se abre demasiado, no es útil "
    "para planificar."
)

st.divider()
st.subheader("🚀 Si te quedas con ganas")
st.markdown(
    """
- **Compara el cono vs realidad** sobre los últimos 3 meses: usa `model.predict()` sobre las fechas que ya están en `history` y mira si los valores reales caen DENTRO o FUERA del intervalo `[yhat_lower, yhat_upper]`. Si caen fuera mucho, el modelo está mal calibrado.
- **Forecast por arquetipo**: separa la serie histórica por `lead_segment_truth` y entrena tres Prophet. ¿La forma del forecast cambia mucho?
- **Componentes del modelo**: `model.plot_components(forecast)` te separa tendencia, estacionalidad anual, festivos. Útil para entender qué aprendió Prophet.
- **Cross-validation interna**: `from prophet.diagnostics import cross_validation`. Te da MAPE / RMSE rolling, mucho más honesto que un único hold-out.
"""
)
