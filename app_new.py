
# retail_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from scipy.stats import norm

from xai_framework import ExplainerFactory
from xai_framework import BaseExplainer

# Page config
st.set_page_config(
    page_title="Retail ML Dashboard",
    layout="wide",
    page_icon=":bar_chart:"
)

st.title("Explainable ML Framework for Retail")
st.markdown("Integrating Demand Forecasting, Dynamic Pricing, and Inventory Optimization")

# Sidebar selection
view = st.sidebar.selectbox("View", [
    "EDA",
    "Demand Forecasting",
    "Dynamic Pricing",
    "Inventory Optimization",
    "Explainability"
])

@st.cache_data
def load_data():
    df = pd.read_csv("retail_clean.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    return df

@st.cache_resource
def load_models():
    return {
        "forecast": joblib.load("models/forecast_model.pkl"),
        "pricing":  joblib.load("models/pricing_model.pkl"),
        "order":    joblib.load("models/order_quantity_model.pkl"),
        "scaler":   joblib.load("preprocessor/scaler.pkl"),
    }

# Load data and models once
df = load_data()
models = load_models()

# --- EDA view ---
if view == "EDA":
    st.header("Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Units Sold Over Time")
        fig = px.line(df, x="Date", y="Units Sold", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Price Distribution")
        fig = px.histogram(df, x="Price", nbins=50, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Feature Correlation")
    corr = df.select_dtypes(include=[float]).corr()
    fig = px.imshow(corr, text_auto=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- Demand Forecasting view ---
elif view == "Demand Forecasting":
    st.header("Demand Forecasting")
    df_f = df.copy()
    df_f["Lag_1"]       = df_f["Units Sold"].shift(1)
    df_f["Lag_7"]       = df_f["Units Sold"].shift(7)
    df_f["RollMean_7"]  = df_f["Units Sold"].rolling(7).mean()
    df_f["RollMean_14"] = df_f["Units Sold"].rolling(14).mean()
    subset = ["Lag_1","Lag_7","RollMean_7","RollMean_14","Price","Discount","Units Sold"]
    df_f = df_f.dropna(subset=subset)
    X_f = df_f[["Lag_1","Lag_7","RollMean_7","RollMean_14","Price","Discount"]]
    y_f = df_f["Units Sold"]
    m = models["forecast"]
    df_f["Forecast"] = m.predict(X_f)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f["Date"], y=y_f, name="Actual"))
    fig.add_trace(go.Scatter(x=df_f["Date"], y=df_f["Forecast"], name="Forecast"))
    fig.update_layout(title="Actual vs Forecast", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    mse = mean_squared_error(y_f, df_f["Forecast"])
    rmse = np.sqrt(mse)
    r2   = r2_score(y_f, df_f["Forecast"])
    st.write(f"**RMSE:** {rmse:.3f}   **R²:** {r2:.3f}")

# --- Dynamic Pricing view ---
elif view == "Dynamic Pricing":
    st.header("Dynamic Pricing")
    df_p = df[(df["Units Sold"] > 0) & (df["Price"] > 0)].copy()
    df_p["ln_Q"] = np.log(df_p["Units Sold"])
    df_p["ln_P"] = np.log(df_p["Price"])
    X_p = df_p[["ln_P","Discount"]]
    y_p = df_p["ln_Q"]
    m = models["pricing"]
    df_p["Pred_lnQ"] = m.predict(X_p)
    df_p["Pred_Q"]   = np.exp(df_p["Pred_lnQ"])

    med = df_p["Discount"].median()
    grid = np.linspace(df_p["ln_P"].min(), df_p["ln_P"].max(), 100)
    df_curve = pd.DataFrame({"ln_P": grid, "Discount": med})
    df_curve["Pred_lnQ"] = m.predict(df_curve)
    df_curve["Pred_Q"]   = np.exp(df_curve["Pred_lnQ"])
    fig = px.line(
        df_curve,
        x=np.exp(df_curve["ln_P"]),
        y="Pred_Q",
        labels={"x":"Price","Pred_Q":"Demand"},
        title="Demand Curve (Median Discount)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    elasticity = (df_p["Pred_lnQ"].diff() / df_p["ln_P"].diff()).mean()
    st.write(f"**Estimated Elasticity:** {elasticity:.2f}")

# --- Inventory Optimization view ---
elif view == "Inventory Optimization":
    st.header("Inventory Optimization")
    df_i = df.copy()
    df_i["Lag_1"]      = df_i["Units Sold"].shift(1)
    df_i["Lag_7"]      = df_i["Units Sold"].shift(7)
    df_i["RollMean_7"] = df_i["Units Sold"].rolling(7).mean()
    df_i = df_i.dropna(subset=["Lag_1","Lag_7","RollMean_7"])
    X_i = df_i[["Lag_1","Lag_7","RollMean_7","Price","Discount","Units Sold"]]
    m = models["order"]
    pred_scaled = m.predict(X_i)
    sc = models["scaler"]
    idx = list(sc.feature_names_in_).index("Units Ordered")
    df_i["Pred_Units_Ordered"] = pred_scaled * sc.scale_[idx] + sc.mean_[idx]

    avg_d = df_i["Pred_Units_Ordered"].mean()
    std_d = df_i["Pred_Units_Ordered"].std()
    LT, OC, HR = 10, 50, 0.02
    UC = df_i["Price"].mean()
    EOQ = np.sqrt(2 * avg_d * OC / (HR * UC))
    safety = norm.ppf(0.95) * std_d * np.sqrt(LT)
    rp = avg_d * LT + safety

    st.metric("EOQ", f"{EOQ:.0f} units")
    st.metric("Safety Stock", f"{safety:.0f} units")
    st.metric("Reorder Point", f"{rp:.0f} units")
    fig = px.line(df_i, x="Date", y="Pred_Units_Ordered", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- Explainability view ---
elif view == "Explainability":
    st.header("Explainability")
    # Prepare features same as forecasting
    df_f = df.copy()
    for lag in [1, 7]: df_f[f"Lag_{lag}"] = df_f["Units Sold"].shift(lag)
    df_f["RollMean_7"] = df_f["Units Sold"].rolling(7).mean()
    df_f["RollMean_14"] = df_f["Units Sold"].rolling(14).mean()
    df_f = df_f.dropna(subset=["Lag_1","Lag_7","RollMean_7","RollMean_14"])
    X_f = df_f[["Lag_1","Lag_7","RollMean_7","RollMean_14","Price","Discount"]]

    method = st.selectbox('XAI Method', ['SHAP', 'LIME', 'PDP'])
    explainer: BaseExplainer = ExplainerFactory.get_explainer(method, models['forecast'], X_f)

    if method == 'SHAP':
        plot_type = st.selectbox('SHAP plot type', ['summary', 'dependence', 'force'])
        params = {'plot_type': plot_type}
        if plot_type == 'dependence':
            params['feature'] = st.selectbox('Feature for dependence', X_f.columns)
        if plot_type == 'force':
            params['instance_idx'] = st.number_input('Instance index', 0, len(X_f)-1, len(X_f)-1)
        fig = explainer.explain(X_f, **params)
        st.pyplot(fig)

    elif method == 'LIME':
        idx = st.number_input('Instance index', 0, len(X_f)-1, len(X_f)-1)
        exp = explainer.explain(X_f, instance_idx=idx)
        from streamlit.components.v1 import html
        html(exp.as_html(), height=400)

    elif method == 'PDP':
        features = st.multiselect('Features', X_f.columns, X_f.columns[:2])
        fig = explainer.explain(X_f, features=features)
        st.pyplot(fig)

