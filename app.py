import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

plt.style.use("default")
matplotlib.rcParams.update({
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#f8fafc",
    "grid.color": "#e2e8f0",
    "text.color": "#1e293b",
    "axes.labelcolor": "#1e293b",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b"
})

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Sri Lanka Rainfall Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Modern CSS UI
# ----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f8fafc !important;
}

/* Increase overall app width/reduce margins */
[data-testid="stAppViewContainer"] > .main > div {
    max-width: 95% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
[data-testid="stHeader"]{
    background:#ffffff !important;
    border-bottom:1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"]{
    background:#ffffff !important;
    border-right:1px solid #e2e8f0 !important;
}

/* Force readable text */
[data-testid="stAppViewContainer"] *{
    color:#0f172a !important;
    opacity:1 !important;
}

/* ---------- Typography ---------- */
.big-title{
    font-size:2.0rem;
    font-weight:900;
    letter-spacing:-0.5px;
    margin-bottom:0.1rem;
}
.subtle{
    color:#475569 !important;
    font-weight:600;
}

/* ---------- Cards ---------- */
.card{
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:18px !important;
    padding:16px 16px !important;
    box-shadow:0 6px 18px rgba(15,23,42,0.05) !important;
}
.card-title{
    font-weight:900 !important;
    font-size:0.95rem !important;
    color:#0f172a !important;
    margin-bottom:8px !important;
}
.kpi{
    font-size:1.8rem !important;
    font-weight:900 !important;
    margin-top:-6px !important;
}
.kpi-sub{
    color:#475569 !important;
    font-weight:700 !important;
    font-size:0.9rem !important;
}

/* ---------- Hero ---------- */
.hero{
    border-radius:28px !important;
    padding:45px 50px !important; /* Significantly increased padding */
    position:relative !important;
    overflow:hidden !important;
    border:1px solid rgba(255,255,255,0.18) !important;
    box-shadow:
      0 20px 50px rgba(37,99,235,0.25),
      inset 0 1px 0 rgba(255,255,255,0.45) !important;
    min-height: 180px; /* Force more height */
}
.hero:before{
    content:"";
    position:absolute;
    inset:-40% -35% auto -35%;
    height:80%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), rgba(255,255,255,0) 65%);
    transform: rotate(-8deg);
}
.hero *{
    position:relative;
    color:#ffffff !important;
    text-shadow:0 2px 12px rgba(0,0,0,0.2);
}
.hero-top{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:20px;
}
.hero-label{
    text-transform:uppercase;
    letter-spacing:3px;
    font-weight:900;
    font-size:1rem;
    opacity:0.95;
    margin-bottom: 10px;
}
.hero-value{
    font-size:4.5rem; /* Larger value */
    font-weight:900;
    line-height:0.9;
}
.pill{
    display:inline-block;
    padding:8px 14px;
    border-radius:999px;
    background: rgba(255,255,255,0.22);
    border:1px solid rgba(255,255,255,0.30);
    font-weight:900;
}

/* ---------- Sidebar Widgets ---------- */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
    font-weight:900 !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] > details > summary{
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:14px !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] > details > summary:hover{
    background:#e0f2fe !important;
    border-color:#bae6fd !important;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]{
    background:#ffffff !important;
    border:1px solid #cbd5e1 !important;
    border-radius:12px !important;
}
input{
    background:#ffffff !important;
    color:#0f172a !important;
    font-weight:700 !important;
}

/* ---------- Dropdown Popover (The List of Options) ---------- */
div[data-baseweb="popover"] {
    background: transparent !important;
}

div[data-baseweb="popover"] ul {
    background-color: #f0f9ff !important; /* Light sky blue */
    border: 1px solid #bae6fd !important;
    border-radius: 12px !important;
}

div[data-baseweb="popover"] li {
    background-color: transparent !important;
    color: #0369a1 !important; /* Azure blue text */
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #e0f2fe !important;
    color: #02507d !important;
}

/* Selected item in the list */
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #bae6fd !important;
    color: #0c4a6e !important;
}

/* Fix visibility of + and - marks in Number Inputs */
.stNumberInput button {
    background-color: #e0f2fe !important;
    border: 1px solid #bae6fd !important;
    color: #0369a1 !important;
    border-radius: 8px !important;
}
.stNumberInput svg {
    fill: #0369a1 !important;
}

/* Buttons */
.stButton button{
    background:#e0f2fe !important;
    border:1px solid #bae6fd !important;
    color:#0369a1 !important;
    font-weight:900 !important;
    border-radius:12px !important;
}

/* Tabs */
button[data-baseweb="tab"]{
    color:#2563eb !important;
    font-weight:900 !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
    border-bottom:3px solid #2563eb !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Assets
# ----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("xgb_rain_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_columns, explainer

try:
    model, feature_columns, explainer = load_assets()
except Exception as e:
    st.error(f"❌ Error loading model files: {e}\n\nRun training first to create: xgb_rain_model.pkl + feature_columns.pkl")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"]
DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
DAY_MAP = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}

def rain_style(pred):
    if pred <= 0:
        return ("linear-gradient(135deg, #10b981 0%, #059669 100%)", "Clear", "☀️", "Zero")
    if pred < 2.5:
        return ("linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)", "Light Rain", "🌦️", "Light")
    if pred < 10:
        return ("linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)", "Moderate Rain", "🌧️", "Moderate")
    return ("linear-gradient(135deg, #4338ca 0%, #312e81 100%)", "Heavy Rain", "⛈️", "Heavy")

def build_input_df(feature_cols):
    df = pd.DataFrame(columns=feature_cols)
    df.loc[0] = 0.0
    return df

def set_val(df, col, val):
    if col in df.columns:
        df.at[0, col] = float(val)

@st.cache_data
def get_sample_for_global(feature_cols, nrows=10000, sample_n=1200):
    df = pd.read_csv("SriLanka_Weather_Dataset.csv", nrows=nrows)
    df = df.sample(min(sample_n, len(df)), random_state=42)

    # time -> month/dayofweek
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["month"] = df["time"].dt.month.fillna(1).astype(int)
    df["dayofweek"] = df["time"].dt.dayofweek.fillna(0).astype(int)

    # one-hot city
    if "city" in df.columns:
        df = pd.get_dummies(df, columns=["city"], prefix="city", drop_first=True)

    X = pd.DataFrame(0.0, index=np.arange(len(df)), columns=feature_cols)
    for c in feature_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")
    X = X.fillna(0).astype(float)
    return X

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.markdown("## 🎛️ Dashboard Controls")

with st.sidebar:
    with st.expander("🌡️ Temperature", expanded=True):
        temp_mean = st.slider("Mean Temp (°C)", 15.0, 40.0, 28.0)
        temp_max  = st.slider("Max Temp (°C)",  15.0, 45.0, 32.0)
        temp_min  = st.slider("Min Temp (°C)",  10.0, 35.0, 24.0)

    with st.expander("💨 Atmosphere & Wind", expanded=False):
        wind_max  = st.number_input("Max Wind Speed (m/s)", 0.0, 30.0, 8.0, step=0.5)
        radiation = st.number_input("Radiation (MJ/m²)",     0.0, 40.0, 20.0, step=0.5)
        evapo     = st.number_input("Evapotranspiration (mm)",0.0, 15.0, 4.0, step=0.2)

    with st.expander("📍 Location & Date", expanded=True):
        month_name = st.selectbox("Month", MONTH_NAMES, index=0)
        month = MONTH_NAMES.index(month_name) + 1

        day = st.selectbox("Day of Week", DAYS, index=0)

        available_cities = [c.replace("city_", "") for c in feature_columns if c.startswith("city_")]
        city = st.selectbox("City", available_cities + ["Other"], index=0)

    st.markdown("---")
    st.caption("Tip: Change inputs and observe SHAP + PDP/LIME for explainability.")

# ----------------------------
# Build Input Vector
# ----------------------------
input_df = build_input_df(feature_columns)
set_val(input_df, "temperature_2m_mean", temp_mean)
set_val(input_df, "temperature_2m_max", temp_max)
set_val(input_df, "temperature_2m_min", temp_min)
set_val(input_df, "windspeed_10m_max", wind_max)
set_val(input_df, "shortwave_radiation_sum", radiation)
set_val(input_df, "et0_fao_evapotranspiration", evapo)
set_val(input_df, "month", month)
set_val(input_df, "dayofweek", DAY_MAP[day])

if city != "Other":
    set_val(input_df, f"city_{city}", 1.0)

# ----------------------------
# Predict
# ----------------------------
try:
    prediction = float(model.predict(input_df)[0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

prediction = max(0.0, prediction)
bg, condition, icon, intensity = rain_style(prediction)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="big-title">🌧️ Sri Lanka Rainfall Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown("")

# ----------------------------
# KPI Row
# ----------------------------
k1, k2, k3, k4 = st.columns([1.1, 1.1, 1.1, 1.2])

with k1:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Predicted Rainfall</div>
      <div class="kpi">{prediction:.2f} mm</div>
    
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Condition</div>
      <div class="kpi">{icon} {condition}</div>
 
    </div>
    """, unsafe_allow_html=True)

with k3:
    loc = city if city != "Other" else "Sri Lanka"
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Location</div>
      <div class="kpi">📍 {loc}</div>

    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Time Context</div>
      <div class="kpi">🗓️ {month_name} • {day}</div>
    
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["🎯 Prediction", "🔬 Model Intelligence"])

# ----------------------------
# Tab 1: Prediction UI
# ----------------------------
with tab1:
    st.markdown(f"""
    <div class="hero" style="background: {bg};">
      <div class="hero-top">
        <div>
          <div class="hero-label">Today’s Weather Outlook</div>
          <div style="margin-top:10px;">
            <span class="pill">{loc} • {month_name} • {day}</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div class="hero-value">{prediction:.2f} <span style="font-size:1.7rem; font-weight:900;">mm</span></div>
          <div style="font-weight:900; opacity:0.95; font-size:1.05rem;">{icon} {condition}</div>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([1.0, 1.2], gap="large")

    # Gauge + quick insights
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📟 Intensity</div>', unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={"font": {"size": 42, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": "#334155"},
                "bar": {"thickness": 0.60},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 5],  "color": "#dcfce7"},
                    {"range": [5, 15], "color": "#dbeafe"},
                    {"range": [15, 50],"color": "#e0e7ff"},
                ],
            },
        ))
        fig_gauge.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#0f172a", "family": "Inter, sans-serif"},
            template="plotly_white"
        )
        fig_gauge.update_traces(gauge_axis_tickfont={"color": "#0f172a", "size": 12})
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🌍 Environmental Snapshot</div>', unsafe_allow_html=True)
        cA, cB = st.columns(2)

        with cA:
            st.metric("Mean Temp (°C)", f"{temp_mean:.1f}")
            st.metric("Wind Max (m/s)", f"{wind_max:.1f}")
        with cB:
            st.metric("Radiation (MJ/m²)", f"{radiation:.1f}")
            st.metric("Evapo (mm)", f"{evapo:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # SHAP Local + Drivers
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧠 Local Explainability (SHAP Waterfall)</div>', unsafe_allow_html=True)

        shap_vals = explainer.shap_values(input_df)

        # Top positive and negative
        top_pos = feature_columns[int(np.argmax(shap_vals[0]))]
        top_neg = feature_columns[int(np.argmin(shap_vals[0]))]

        st.markdown(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px;">
          <span class="pill" style="background:rgba(34,197,94,0.12); border-color:rgba(34,197,94,0.25); color:#14532d !important;">
            🚀 Driver: {top_pos.replace('_',' ').capitalize()}
          </span>
          <span class="pill" style="background:rgba(245,158,11,0.12); border-color:rgba(245,158,11,0.25); color:#7c2d12 !important;">
            ⚠️ Limiter: {top_neg.replace('_',' ').capitalize()}
          </span>
          <span class="pill" style="background:rgba(59,130,246,0.12); border-color:rgba(59,130,246,0.25); color:#1e3a8a !important;">
            📊 Class: {intensity}
          </span>
        </div>
        """, unsafe_allow_html=True)

        plt.clf()
        fig = plt.figure(figsize=(8, 4), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_columns,
        )
        shap.plots.waterfall(explanation, max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Session state toggles
# ----------------------------
if "show_pdp" not in st.session_state:
    st.session_state.show_pdp = False
if "show_lime" not in st.session_state:
    st.session_state.show_lime = False
if "last_pdp_feature" not in st.session_state:
    st.session_state.last_pdp_feature = feature_columns[0] if len(feature_columns) else ""

# ----------------------------
# Tab 2: Model Intelligence
# ----------------------------
with tab2:
    topA, topB = st.columns([1.0, 1.0], gap="large")

    # Global shap
    with topA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🌐 Global Feature Impact (Top 10)</div>', unsafe_allow_html=True)

        try:
            X_sample = get_sample_for_global(feature_columns, nrows=10000, sample_n=1400)
            g_shap = explainer.shap_values(X_sample)

            fig_g = plt.figure(figsize=(7, 4))
            shap.summary_plot(g_shap, X_sample, plot_type="bar", show=False, max_display=10)
            plt.title("Global Feature Importance (SHAP)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_g, clear_figure=True)
            plt.close(fig_g)

        except Exception as e:
            st.warning(f"Global SHAP could not be generated: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Feature explorer (simple distribution view)
    with topB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Feature Explorer (Dataset Sample)</div>', unsafe_allow_html=True)

        try:
            X_sample = get_sample_for_global(feature_columns, nrows=10000, sample_n=1400)
            numeric_cols = [c for c in feature_columns if c in X_sample.columns][:20]
            chosen = st.selectbox("Select feature to view distribution", numeric_cols)

            fig_hist = px.histogram(X_sample, x=chosen, nbins=30, title=f"Distribution: {chosen}", template="plotly_white")
            fig_hist.update_layout(
                height=350, 
                margin=dict(l=40, r=20, t=50, b=40),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font={"color": "#0f172a", "family": "Inter, sans-serif"},
                title_font={"size": 16, "color": "#0f172a"}
            )
            fig_hist.update_xaxes(
                showline=True, linewidth=1, linecolor='#cbd5e1', 
                gridcolor='#f1f5f9', title_font={"color": "#0f172a"},
                tickfont={"color": "#0f172a"}
            )
            fig_hist.update_yaxes(
                showline=True, linewidth=1, linecolor='#cbd5e1', 
                gridcolor='#f1f5f9', title_font={"color": "#0f172a"},
                tickfont={"color": "#0f172a"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        except Exception as e:
            st.warning(f"Feature distribution view failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    c1, c2 = st.columns([1, 1], gap="large")

    # PDP / ICE
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧩 Partial Dependence (PDP & ICE)</div>', unsafe_allow_html=True)
        st.write("Shows how the prediction changes with a feature (avg PDP) + individual ICE lines.")

        pdp_feature = st.selectbox("Select feature (PDP/ICE)", feature_columns[: min(20, len(feature_columns))])

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Generate PDP/ICE"):
                st.session_state.show_pdp = True
                st.session_state.last_pdp_feature = pdp_feature
        with btn2:
            if st.button("Hide PDP/ICE"):
                st.session_state.show_pdp = False

        if st.session_state.show_pdp:
            try:
                from sklearn.inspection import PartialDependenceDisplay
                X_sample = get_sample_for_global(feature_columns, nrows=10000, sample_n=1400)

                fig_p = plt.figure(figsize=(7.5, 5.2))
                axp = fig_p.add_subplot(111)
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_sample,
                    [st.session_state.last_pdp_feature],
                    kind="both",
                    subsample=60,
                    ax=axp
                )
                axp.grid(True, alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig_p, clear_figure=True)
                plt.close(fig_p)
            except Exception as e:
                st.warning(f"PDP/ICE failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # LIME
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧪 Local Surrogate (LIME)</div>', unsafe_allow_html=True)
        st.write("Explains this specific case using a simple local model around the instance.")

        btn3, btn4 = st.columns(2)
        with btn3:
            if st.button("Generate LIME"):
                st.session_state.show_lime = True
        with btn4:
            if st.button("Hide LIME"):
                st.session_state.show_lime = False

        if st.session_state.show_lime:
            try:
                from lime.lime_tabular import LimeTabularExplainer
                X_sample = get_sample_for_global(feature_columns, nrows=10000, sample_n=1400)

                lx = LimeTabularExplainer(
                    X_sample.values,
                    feature_names=feature_columns,
                    mode="regression",
                    discretize_continuous=True
                )
                lexp = lx.explain_instance(input_df.values[0], model.predict)
                fig_l = lexp.as_pyplot_figure()
                plt.tight_layout()
                st.pyplot(fig_l, clear_figure=True)
                plt.close(fig_l)
            except Exception as e:
                st.warning(f"LIME failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

