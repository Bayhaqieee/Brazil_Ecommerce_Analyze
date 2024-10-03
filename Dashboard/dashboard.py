import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import FastMarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency

# Load dataset
data = pd.read_csv('Dashboard/combines_data.csv')

# Sidebar
with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    # Assuming the dataset has date columns
    min_date = pd.to_datetime(data['order_approved_at']).min()
    max_date = pd.to_datetime(data['order_approved_at']).max()
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Tabs
tab1, tab2, tab3 = st.tabs(["Customer Analysis", "Geographic & Time Insights", "Product Insights"])

# Helper functions
def calculate_rfm(data):
    """Calculate RFM (Recency, Frequency, Monetary) for customers."""
    # Convert to datetime
    data['order_approved_at'] = pd.to_datetime(data['order_approved_at'])
    
    # Recency: Difference between the last purchase and the end of the period (max date)
    max_date = data['order_approved_at'].max()
    rfm = data.groupby('customer_id').agg({
        'order_approved_at': lambda x: (max_date - x.max()).days,  # Recency
        'order_id': 'count',                                       # Frequency
        'payment_value': 'sum'                                      # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Handle cases where qcut may fail due to duplicates
    rfm['R'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
    rfm['F'] = pd.qcut(rfm['frequency'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    rfm['M'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    
    # Combine RFM scores into a single RFM Score column
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    
    return rfm

def calculate_monthly_sales(data):
    """Calculate monthly sales growth."""
    data['order_approved_at'] = pd.to_datetime(data['order_approved_at'])
    data['month_year'] = data['order_approved_at'].dt.to_period('M')
    
    monthly_sales = data.groupby('month_year').agg({
        'payment_value': 'sum'
    }).reset_index()
    
    monthly_sales['sales_growth'] = monthly_sales['payment_value'].pct_change() * 100  # Percentage change
    return monthly_sales

def daily_orders_overview(data):
    """Calculate daily orders overview."""
    data['order_approved_at'] = pd.to_datetime(data['order_approved_at'])
    daily_orders = data.groupby(data['order_approved_at'].dt.date).agg({
        'order_id': 'count'
    }).reset_index()
    return daily_orders

# Tab 1 Content
with tab1:
    st.header("RFM Analysis of Customers")
    # RFM Analysis
    rfm_data = calculate_rfm(data)
    st.write(rfm_data.head())  # Show a preview of the RFM data
    
    # Visualize RFM segments
    st.subheader("RFM Segment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=rfm_data, x='RFM_Score', ax=ax)
    st.pyplot(fig)

    st.header("Monthly Sales Growth")
    # Monthly Sales Growth
    monthly_sales_data = calculate_monthly_sales(data)
    st.write(monthly_sales_data)
    
    # Visualize Monthly Sales Growth
    st.subheader("Sales Growth (Monthly)")
    fig, ax = plt.subplots()
    sns.lineplot(data=monthly_sales_data, x='month_year', y='sales_growth', ax=ax)
    ax.set_xlabel('Month-Year')
    ax.set_ylabel('Sales Growth (%)')
    st.pyplot(fig)

    st.header("Daily Orders Overview")
    # Daily Orders Overview
    daily_orders_data = daily_orders_overview(data)
    st.write(daily_orders_data)
    
    # Visualize Daily Orders
    st.subheader("Daily Orders Count")
    fig, ax = plt.subplots()
    sns.lineplot(data=daily_orders_data, x='order_approved_at', y='order_id', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Orders')
    st.pyplot(fig)

# Tab 2: Geographic & Time Insights
with tab2:
    st.header("Customer 'Prime Time' and 'Dead Time'")
    # Time analysis code here
    st.write("Implement customer prime and dead time.")

    st.header("Customer and Seller Geolocation")
    # Create a map using Folium
    m = folium.Map(location=[-23.55, -46.63], zoom_start=4)
    
    # Add customer and seller points from the dataset
    for i, row in data.iterrows():
        folium.Marker([row['geolocation_lat_customer'], row['geolocation_lng_customer']], 
                      popup=f"Customer: {row['customer_city']}, {row['customer_state']}").add_to(m)
        folium.Marker([row['geolocation_lat_seller'], row['geolocation_lng_seller']], 
                      popup=f"Seller: {row['seller_city']}, {row['seller_state']}").add_to(m)
    
    st_folium(m, width=700, height=500)
    
    st.header("Payment Method by Customers")
    # Payment method visualization code
    payment_types = data['payment_type'].value_counts()
    st.bar_chart(payment_types)

# Tab 3: Product Insights
with tab3:
    st.header("Product Flow (Shipped to Delivered)")
    # Product flow analysis code
    st.write("Analyze product flow from shipping to delivery.")

    st.header("Best and Worst Performing Products")
    # Performance analysis code
    st.write("Show best and worst performing products.")

    st.header("Product Reviews")
    # Product reviews analysis code
    review_scores = data['review_score'].value_counts()
    st.bar_chart(review_scores)
