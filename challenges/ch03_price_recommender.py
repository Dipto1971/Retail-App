import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from models import gradient_boosting_model, xgboost_model
import matplotlib.pyplot as plt

# Cache data loading for performance
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('retail_store_inventory.csv')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'retail_store_inventory.csv' is in the correct directory.")
        return pd.DataFrame()

# Function to encode categorical variables
def encode_input_data(input_df, categorical_cols, encoders):
    input_encoded = input_df.copy()
    for col in categorical_cols:
        if col in encoders:
            input_encoded[col] = encoders[col].transform(input_df[col])
    input_encoded = pd.get_dummies(input_encoded, columns=categorical_cols, drop_first=True)
    return input_encoded

# Function to recommend price based on enhanced tuning logic
def recommend_price(input_data):
    """
    Tune the user-provided price based on Demand_to_Inventory_Ratio and Competitor Pricing.
    Returns the recommended price in dollars.
    """
    user_price = input_data['Price'].iloc[0]
    demand_ratio = input_data['Demand_to_Inventory_Ratio'].iloc[0]
    competitor_price = input_data['Competitor Pricing'].iloc[0]

    # Base adjustment based on Demand_to_Inventory_Ratio
    adjustment_factor = 1.1 if demand_ratio > 1 else 0.9

    # Dynamic adjustment based on Price vs. Competitor Pricing
    price_ratio = user_price / competitor_price if competitor_price > 0 else 1.0
    if price_ratio < 0.8:  # User price is much lower than competitor
        adjustment_factor *= 1.05  # Allow slightly larger increase
    elif price_ratio > 1.2:  # User price is much higher than competitor
        adjustment_factor *= 0.95  # Reduce adjustment to stay competitive

    # Calculate recommended price
    recommended_price = user_price * adjustment_factor

    # Cap the adjustment to stay within 20% of Competitor Pricing
    if competitor_price > 0:
        max_price = competitor_price * 1.2
        min_price = competitor_price * 0.8
        recommended_price = np.clip(recommended_price, min_price, max_price)

    # Ensure minimum price
    recommended_price = max(recommended_price, 0.50)  # Minimum price $0.50

    return recommended_price

# Price recommendation function
def run_price_recommendation(data):
    st.title("Challenge 3: Price Recommendation")
    st.markdown("""
    This section tunes a user-provided price for a product based on input features, including Demand-to-Inventory Ratio 
    (Demand Forecast / Inventory Level) and Competitor Pricing. Enter the required features and your proposed price below 
    to generate a tuned price recommendation.
    """)

    # Define categorical and numerical columns
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 
                     'Discount', 'Holiday/Promotion', 'Competitor Pricing', 'Price']

    # Initialize LabelEncoders for categorical columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(data[col])
        encoders[col] = le

    # Load Gradient Boosting model for feature importance
    try:
        model = gradient_boosting_model
        st.write("Model features for importance analysis:", model.feature_names_in_)  # Display for debugging
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Input form
    st.subheader("Input Features")
    with st.form(key='price_recommendation_form'):
        input_data = {}
        
        # Numerical inputs
        cols = st.columns(2)
        for i, col in enumerate(numerical_cols):
            with cols[i % 2]:
                if col == 'Inventory Level':
                    input_data[col] = st.number_input(f"{col} (must be positive for ratio)", 
                                                    min_value=0.01, value=100.0, step=0.01, format="%.2f", key=col)
                elif col == 'Demand Forecast':
                    input_data[col] = st.number_input(f"{col}", 
                                                    min_value=0.0, value=50.0, step=0.01, format="%.2f", key=col)
                elif col == 'Price':
                    input_data[col] = st.number_input(f"{col} (your proposed price)", 
                                                     min_value=0.01, value=10.0, step=0.01, format="%.2f", key=col)
                elif col == 'Competitor Pricing':
                    input_data[col] = st.number_input(f"{col}", 
                                                     min_value=0.0, value=10.0, step=0.01, format="%.2f", key=col)
                else:
                    input_data[col] = st.number_input(f"{col}", 
                                                     value=0.0, step=0.01, format="%.2f", key=col)

        # Categorical inputs
        for col in categorical_cols:
            options = data[col].unique().tolist()
            input_data[col] = st.selectbox(f"{col}", options, key=col)

        submit_button = st.form_submit_button("Tune Price")

    if submit_button:
        try:
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])

            # Calculate Demand_to_Inventory_Ratio
            if input_df['Inventory Level'].iloc[0] == 0:
                st.error("Inventory Level cannot be zero for Demand-to-Inventory Ratio calculation.")
                return
            input_df['Demand_to_Inventory_Ratio'] = input_df['Demand Forecast'] / input_df['Inventory Level']

            # Encode categorical variables for feature importance
            input_encoded = encode_input_data(input_df, categorical_cols, encoders)

            # Get expected features from model
            expected_cols = model.feature_names_in_

            # Ensure all expected columns are present
            for col in expected_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            # Reorder columns to match training data
            input_encoded = input_encoded[expected_cols]

            # Tune price
            recommended_price = recommend_price(input_df)

            # Display results
            st.success(f"User-Provided Price: ${input_df['Price'].iloc[0]:.2f}")
            st.success(f"Recommended Price (after Tuning): ${recommended_price:.2f}")
            st.info(f"Demand-to-Inventory Ratio: {input_df['Demand_to_Inventory_Ratio'].iloc[0]:.2f}")
            st.info(f"Competitor Pricing: ${input_df['Competitor Pricing'].iloc[0]:.2f}")

            # Explain adjustment
            demand_ratio = input_df['Demand_to_Inventory_Ratio'].iloc[0]
            if demand_ratio > 1:
                st.write("Price increased due to high demand (Demand-to-Inventory Ratio > 1).")
            else:
                st.write("Price decreased due to low demand (Demand-to-Inventory Ratio ≤ 1).")
            if abs(recommended_price - input_df['Competitor Pricing'].iloc[0]) > 0.2 * input_df['Competitor Pricing'].iloc[0]:
                st.write("Adjustment capped to stay competitive with Competitor Pricing.")

            # Display feature importance
            st.subheader("Feature Importance (from Gradient Boosting Model)")
            feature_importance = pd.DataFrame({
                'Feature': expected_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            st.dataframe(feature_importance)

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for Price Context')
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in tuning price: {str(e)}")