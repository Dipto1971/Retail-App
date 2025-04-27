import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def run_demand_forecast_section(data):
    st.header("Challenge 2: Demand Forecasting with Machine Learning Models")
    st.write("This section forecasts demand using an ARIMA model for the selected product over a user-specified horizon.")

    # Product selection
    selected_product = st.selectbox("Select a Product", data['Product ID'].unique())

    # Forecast horizon
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)

    # Historical data range
    historical_days = st.slider("Historical Data Range (days)", 30, 180, 90, help="Number of days of historical data to display.")

    # Plot options
    show_historical = st.checkbox("Show Historical Data", value=True)
    show_forecast = st.checkbox("Show Forecast", value=True)

    # Filter and prepare data
    product_data = data[data['Product ID'] == selected_product][['Date', 'Units Sold']].copy()
    product_data = product_data.sort_values('Date').dropna()
    
    if product_data.empty:
        st.error(f"No data available for Product ID {selected_product}.")
        return

    # Limit historical data to user-specified range
    min_date = product_data['Date'].max() - pd.Timedelta(days=historical_days)
    product_data = product_data[product_data['Date'] >= min_date]

    # Prepare forecast
    if show_forecast:
        try:
            # Fit ARIMA model on Units Sold
            model = ARIMA(product_data['Units Sold'], order=(5, 1, 0))  # Simple ARIMA(5,1,0)
            model_fit = model.fit()

            # Generate forecast
            forecast_dates = pd.date_range(start=product_data['Date'].max() + pd.Timedelta(days=1), 
                                         periods=forecast_horizon, freq='D')
            forecast = model_fit.forecast(steps=forecast_horizon)
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Units Sold': forecast})

        except Exception as e:
            st.error(f"Forecasting error: {str(e)}")
            return
    else:
        forecast_df = pd.DataFrame()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    if show_historical and not product_data.empty:
        ax.plot(product_data['Date'], product_data['Units Sold'], 
                label='Historical Units Sold', color='blue', marker='o', markersize=4, linestyle='-')

    if show_forecast and not forecast_df.empty:
        ax.plot(forecast_df['Date'], forecast_df['Units Sold'], 
                label='Forecasted Units Sold', color='red', marker='x', markersize=6, linestyle='--')

    # Customize plot
    ax.set_title(f"Demand Forecast for Product {selected_product}", fontsize=14, pad=10)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Units Sold", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose date spacing
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format as YYYY-MM-DD
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Display plot
    st.pyplot(fig)

    # Display forecast summary
    if show_forecast and not forecast_df.empty:
        st.subheader("Forecast Summary")
        st.write(f"Forecast for the next {forecast_horizon} days:")
        forecast_summary = forecast_df[['Date', 'Units Sold']].copy()
        forecast_summary['Date'] = forecast_summary['Date'].dt.strftime('%Y-%m-%d')
        forecast_summary['Units Sold'] = forecast_summary['Units Sold'].round(2)
        st.dataframe(forecast_summary)
        