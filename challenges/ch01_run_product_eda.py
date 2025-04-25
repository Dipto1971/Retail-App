# data_analysis.py (or any module you'd like)
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

def run_product_eda(data):
    st.header("Challenge 1: Data Preprocessing and EDA")
    st.write("This section explores the dataset, cleans it, and visualizes key patterns.")

    # Interactive product selection
    products = data['Product ID'].unique()
    selected_product = st.selectbox("Select a Product", products)

    # Filter data for the selected product
    product_data = data[data['Product ID'] == selected_product]

    # Aggregate 'Units Sold' by date (sum across all stores)
    product_data_grouped = product_data.groupby('Date')['Units Sold'].sum().reset_index()

    # Plot total sales trend
    fig, ax = plt.subplots(figsize=(12, 6))  # Larger figure for better readability
    ax.plot(product_data_grouped['Date'], product_data_grouped['Units Sold'], linestyle='-', color='b')
    ax.set_title(f"Total Sales Trend for {selected_product}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")

    # Set date locator and formatter for clean x-axis
    locator = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45, ha='right')  # Rotate and align ticks

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Summary statistics for the selected product - Enhanced Version
    st.subheader("Detailed Statistics for Selected Product")

    # Select relevant numeric columns for statistics
    numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered',
                    'Demand Forecast', 'Price', 'Discount', 'Competitor Pricing']

    # Generate statistics and format
    stats = product_data[numeric_cols].describe().T
    stats = stats.round(2)
    stats['IQR'] = stats['75%'] - stats['25%']  # Add interquartile range
    stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'IQR', 'max']]

    # Rename columns for better readability
    stats.columns = ['Count', 'Mean', 'Std Dev', 'Min',
                     '25th %ile', 'Median', '75th %ile', 'IQR', 'Max']

    # Display formatted table with caption
    st.dataframe(stats.style.format("{:.2f}"), height=400)
    st.caption(f"Statistical summary for {selected_product} across {len(product_data)} store-day entries")
