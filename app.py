import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from challenges import run_product_eda, forecast_module, clusturing_module

# Cache data loading for performance
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('retail_store_inventory.csv')
        # Convert 'Date' to datetime, handling any errors
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'retail_store_inventory.csv' is in the correct directory.")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Load data
data = load_data()

# Check if data is loaded successfully
if data.empty:
    st.stop()  # Stop execution if data loading fails

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Section", ["Home", "Challenge 1: EDA", "Challenge 2: Forecasting", "Challenge 3: Clustering"])

# Home Page
if page == "Home":
    st.title("Retail Store Inventory Forecasting Dashboard")
    st.markdown("""
    Welcome to the Retail Store Inventory Forecasting Dashboard! This application showcases research on optimizing supply chain management through statistical data analysis. The project addresses three key challenges:
    - **Challenge 1**: Data Preprocessing and Exploratory Data Analysis
    - **Challenge 2**: Demand Forecasting with Machine Learning Models
    - **Challenge 3**: Clustering for Inventory Insights
    Use the sidebar to explore each challenge's findings.
    """)

# Challenge 1: EDA
if page == "Challenge 1: EDA":
    run_product_eda.run_product_eda(data)
    
# Challenge 2: Forecasting
elif page == "Challenge 2: Forecasting":
    forecast_module.run_demand_forecast_section(data)

# Challenge 3: Clustering
elif page == "Challenge 3: Clustering":
    clusturing_module.run_inventory_clustering(data)