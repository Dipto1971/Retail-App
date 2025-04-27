import streamlit as st
import pandas as pd
from models import lstm_model, scaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
torch.classes.__path__ = []

# Cache data loading for performance
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('retail_store_inventory.csv')
        # Convert 'Date' to datetime, handling any errors
        print(df['Date'])
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'retail_store_inventory.csv' is in the correct directory.")
        return pd.DataFrame()

# Load data
data = load_data()

# Check if data is loaded successfully
if data.empty:
    st.stop()

# Function to encode categorical variables
def encode_input_data(input_df, categorical_cols, encoders):
    input_encoded = input_df.copy()
    for col in categorical_cols:
        if col in encoders:
            input_encoded[col] = encoders[col].transform(input_df[col])
    return pd.get_dummies(input_encoded, columns=categorical_cols, drop_first=True)

# Function to run hybrid model predictions
def run_hybrid_model(data):
    st.title("Hybrid ARIMA + LSTM Forecasting for Retail Sales")
    st.markdown("""
    This app uses a hybrid ARIMA + LSTM model to forecast Units Sold based on input features and historical sales data.
    Provide the required features and past Units Sold values to generate a prediction.
    """)

    # Define categorical and numerical columns
    categorical_cols = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    numerical_cols = ['Inventory Level', 'Units Ordered', 'Demand Forecast', 'Price', 
                     'Discount', 'Holiday/Promotion', 'Competitor Pricing']

    # Initialize LabelEncoders for categorical columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(data[col])
        encoders[col] = le

    # Input form
    st.subheader("Input Features")
    with st.form(key='input_form'):
        input_data = {}
        
        # Numerical inputs
        cols = st.columns(2)
        for i, col in enumerate(numerical_cols):
            with cols[i % 2]:
                input_data[col] = st.number_input(f"{col}", value=0.0, step=0.01, format="%.2f", key=col)

        # Categorical inputs
        for col in categorical_cols:
            options = data[col].unique().tolist()
            input_data[col] = st.selectbox(f"{col}", options, key=col)

        # Past Units Sold input with date
        st.subheader("Past Units Sold (with Dates)")
        st.markdown("Enter past Units Sold values with corresponding dates (YYYY-MM-DD, Units Sold), one per line.")
        past_sales_input = st.text_area(
            "Format: YYYY-MM-DD,Units Sold\nExample:\n2022-01-01,127\n2022-01-02,150\n2022-01-03,65",
            value="2022-01-01,127\n2022-01-02,150\n2022-01-03,65\n2022-01-04,61\n2022-01-05,14",
            height=150
        )

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        try:
            # Parse past sales input
            past_sales_data = []
            for line in past_sales_input.split('\n'):
                if line.strip():
                    date_str, units = line.split(',')
                    date = pd.to_datetime(date_str.strip(), errors='coerce')
                    units = float(units.strip())
                    if pd.isna(date):
                        st.error(f"Invalid date format: {date_str}")
                        return
                    past_sales_data.append((date, units))
            
            if len(past_sales_data) < 5:
                st.error("Please provide at least 5 past Units Sold values with valid dates.")
                return

            # Create a DataFrame for past sales with datetime index
            past_sales_df = pd.DataFrame(past_sales_data, columns=['Date', 'Units Sold'])
            past_sales_df.set_index('Date', inplace=True)
            past_sales_df = past_sales_df.sort_index()  # Ensure chronological order

            # Fit ARIMA model on past sales
            arima_model_local = ARIMA(past_sales_df['Units Sold'], order=(0,0,0)).fit()

            # Forecast with ARIMA
            arima_forecast = arima_model_local.forecast(steps=1).iloc[0]

            # Create input DataFrame for LSTM
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables
            input_encoded = encode_input_data(input_df, categorical_cols, encoders)

            # Ensure all expected columns are present
            expected_cols = scaler.feature_names_in_
            for col in expected_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            # Reorder columns to match training data
            input_encoded = input_encoded[expected_cols]

            # Scale input features for LSTM
            input_scaled = scaler.transform(input_encoded)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(-1)

            # LSTM prediction (residuals)
            with torch.no_grad():
                lstm_pred = lstm_model(input_tensor).squeeze().numpy()

            # Hybrid prediction
            hybrid_pred = arima_forecast + lstm_pred

            # Display result
            st.success(f"Predicted Units Sold: {hybrid_pred:.2f}")

            # Plot past sales and prediction
            st.subheader("Prediction vs. Past Sales")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(past_sales_df.index, past_sales_df['Units Sold'], label="Past Units Sold", marker='o')
            # Predict one step ahead from the last date
            next_date = past_sales_df.index[-1] + pd.Timedelta(days=1)
            ax.plot([next_date], [hybrid_pred], 'r*', label="Predicted Units Sold", markersize=15)
            ax.set_xlabel("Date")
            ax.set_ylabel("Units Sold")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Display model details
            st.subheader("Model Details")
            st.write(f"ARIMA Order: (0,0,0)")  # Assuming a simple order; adjust if using pre-trained model
            st.write("LSTM Architecture: 2 layers, 64 hidden units")

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")