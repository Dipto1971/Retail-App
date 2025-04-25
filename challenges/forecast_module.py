import streamlit as st
import matplotlib.pyplot as plt

def run_demand_forecast_section(data):
    st.header("Challenge 2: Demand Forecasting with Machine Learning Models")
    st.write(
        "This section demonstrates forecasting demand using machine learning techniques.")

    # Product selection
    selected_product = st.selectbox(
        "Select a Product", data['Product ID'].unique())

    # Forecast horizon
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
    st.write(
        f"Forecasting for {selected_product} over {forecast_horizon} days (model implementation pending).")

    # Example plot (replace with actual forecast later)
    product_data = data[data['Product ID'] == selected_product]
    fig, ax = plt.subplots()
    ax.plot(product_data['Date'], product_data['Demand Forecast'])
    ax.set_title(f"Demand Forecast for {selected_product}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecasted Demand")
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)
