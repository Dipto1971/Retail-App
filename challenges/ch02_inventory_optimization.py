import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import ch02_xgboost_model
# Import your pre-trained XGBoost model from models.py

# Cache data loading for performance
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('retail_store_inventory.csv')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Sort by Store ID, Product ID, and Date for correct feature generation
        df = df.sort_values(['Store ID', 'Product ID', 'Date'])
        
        # Generate lag features
        for lag in [1, 7]:
            df[f'Sales_Lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(lag)
        
        # Generate rolling mean features
        for window in [7, 14]:
            df[f'Sales_RollingMean_{window}'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # Add DayOfWeek and Month features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        
        # Drop rows with missing values after feature generation
        df = df.dropna()
        
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'retail_store_inventory.csv' is in the correct directory.")
        return pd.DataFrame()

# Inventory optimization function
def run_inventory_optimization(data):
    st.title("Challenge 2: Inventory Optimization")
    st.markdown("""
    Use this tool to predict **Units Sold** and get an optimal **Inventory Level** recommendation based on your inputs. 
    The XGBoost model uses historical sales data to minimize stockouts and overstock.
    """)

    # Define features used in the XGBoost model
    features = [
        'Inventory Level', 'Units Ordered', 'Demand Forecast', 'Price',
        'Discount', 'Competitor Pricing', 'Sales_Lag_1', 'Sales_Lag_7',
        'Sales_RollingMean_7', 'Sales_RollingMean_14', 'DayOfWeek', 'Month'
    ]

    # Load the pre-trained XGBoost model
    try:
        model = ch02_xgboost_model  # From models.py
    except Exception as e:
        st.error(f"Error loading XGBoost model: {str(e)}")
        return

    # Input form for user to provide feature values
    st.subheader("Enter Feature Values")
    with st.form(key='inventory_form'):
        input_data = {}

        # Numerical inputs in two columns for better layout
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                if feature in ['Inventory Level', 'Units Ordered', 'Demand Forecast']:
                    input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=100.0, step=1.0, format="%.0f", key=feature)
                elif feature in ['Sales_Lag_1', 'Sales_Lag_7']:
                    input_data[feature] = st.number_input(f"{feature} (Recent Sales)", min_value=0.0, value=90.0, step=1.0, format="%.0f", key=feature)
                elif feature in ['Sales_RollingMean_7', 'Sales_RollingMean_14']:
                    input_data[feature] = st.number_input(f"{feature} (Avg Sales)", min_value=0.0, value=85.0, step=1.0, format="%.0f", key=feature)
                elif feature in ['Price', 'Competitor Pricing']:
                    input_data[feature] = st.number_input(f"{feature}", min_value=0.01, value=10.0, step=0.01, format="%.2f", key=feature)
                elif feature == 'Discount':
                    input_data[feature] = st.number_input(f"{feature} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, format="%.0f", key=feature)
                elif feature == 'DayOfWeek':
                    input_data[feature] = st.selectbox("Day of Week", options=range(7), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x], key=feature)
                elif feature == 'Month':
                    input_data[feature] = st.selectbox("Month", options=range(1, 13), format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1], key=feature)

        submit_button = st.form_submit_button("Generate Recommendation")

    # Process form submission
    if submit_button:
        try:
            # Create DataFrame from user inputs
            input_df = pd.DataFrame([input_data])

            # Ensure all model features are present and in correct order
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0  # Default to 0 for missing features
            input_df = input_df[model.feature_names_in_]

            # Predict Units Sold
            predicted_sales = model.predict(input_df)[0]

            # Calculate recommended inventory level
            safety_stock = 0.2 * predicted_sales  # 20% buffer to avoid stockouts
            recommended_inventory = predicted_sales + safety_stock
            recommended_inventory = min(recommended_inventory, 2 * predicted_sales)  # Cap at 2x to avoid overstock

            # Display results
            st.success(f"**Predicted Units Sold**: {predicted_sales:.2f}")
            st.success(f"**Recommended Inventory Level**: {recommended_inventory:.2f}")
            st.info(f"Safety Stock Buffer: {safety_stock:.2f} (20% of predicted sales)")

            # Feature importance visualization
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.dataframe(importance_df)

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for Units Sold Prediction')
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# For standalone testing
if __name__ == "__main__":
    data = load_data()
    if not data.empty:
        run_inventory_optimization(data)