import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster
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
tab1, tab2, tab3, tab4 = st.tabs(["Data Conclusion","Customer Analysis", "Geographic & Time Insights", "Product Insights"])

data['order_approved_at'] = pd.to_datetime(data['order_approved_at'])
filtered_data = data[(data['order_approved_at'] >= pd.to_datetime(start_date)) & 
                     (data['order_approved_at'] <= pd.to_datetime(end_date))]

# Function for sales growth analysis
def sales_growth(df, freq='D'):
    # Ensure order_approved_at is in datetime format
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'], errors='coerce')
    
    # Drop rows where the date conversion failed
    df = df.dropna(subset=['order_approved_at'])
    
    # Ensure payment_value is numeric and handle errors
    df['payment_value'] = pd.to_numeric(df['payment_value'], errors='coerce')
    
    # Drop rows where payment_value conversion failed (i.e., non-numeric values)
    df = df.dropna(subset=['payment_value'])
    
    # Resample by the frequency (D=Day, M=Month, Y=Year) and sum the payment_value
    df = df.set_index('order_approved_at').resample(freq).sum(numeric_only=True)
    
    return df['payment_value']

def daily_orders_overview(data):
    """Calculate daily orders overview."""
    data['order_approved_at'] = pd.to_datetime(data['order_approved_at'])
    daily_orders = data.groupby(data['order_approved_at'].dt.date).agg({
        'order_id': 'count'
    }).reset_index()
    return daily_orders

def plot_top_cities_by_purchase(data):
    # Filter for successful orders (delivered)
    successful_orders = data[data['order_delivered_customer_date'].notnull()]

    # Group by city and count the number of purchases
    top_cities = successful_orders.groupby('customer_city').size().sort_values(ascending=False).head(10)

    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_cities.index, y=top_cities.values, palette='rocket', ax=ax)
    ax.set_title('Top 10 Cities with Highest Purchase Activity')
    ax.set_xlabel('Cities')
    ax.set_ylabel('Number of Purchases')
    plt.xticks(rotation=45)
    
    return fig

def create_geolocation_map_customer(data):
    # Limit the data to the first 1000 rows for performance
    data = data.head(5000)
    
    # Create a Folium map centered at the average location of customers
    avg_lat = data['geolocation_lat_customer'].mean()
    avg_lng = data['geolocation_lng_customer'].mean()
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=5)
    
    # Initialize marker clusters for customers and sellers
    customer_cluster = MarkerCluster().add_to(m)
    
    # Add customer locations to the customer cluster
    for i, row in data.iterrows():
        folium.Marker(
            [row['geolocation_lat_customer'], row['geolocation_lng_customer']],
            popup=f"Customer: {row['customer_city']}, {row['customer_state']}"
        ).add_to(customer_cluster)
        
    return m

def create_geolocation_map_seller(data):
    # Limit the data to the first 1000 rows for performance
    data = data.head(5000)
    
    # Create a Folium map centered at the average location of customers
    avg_lat = data['geolocation_lat_customer'].mean()
    avg_lng = data['geolocation_lng_customer'].mean()
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=5)
    
    # Initialize marker clusters for customers and sellers
    seller_cluster = MarkerCluster().add_to(m)
    
    # Add seller locations to the seller cluster
    for i, row in data.iterrows():
        folium.Marker(
            [row['geolocation_lat_seller'], row['geolocation_lng_seller']],
            popup=f"Seller: {row['seller_city']}, {row['seller_state']}"
        ).add_to(seller_cluster)
    
    return m
    
def calculate_product_performance(data):
    # Group by product and calculate total sales and number of purchases
    product_performance = data.groupby('product_category_name_english').agg(
        total_sales=pd.NamedAgg(column='payment_value', aggfunc='sum'),
        total_orders=pd.NamedAgg(column='order_id', aggfunc='count')
    ).reset_index()
    return product_performance

performance_data = calculate_product_performance(filtered_data)

def calculate_top_sellers(data):
    # Calculate total sales per seller
    top_sellers = data.groupby('seller_id').agg({'price': 'sum'}).reset_index()
    top_sellers.columns = ['seller_id', 'total_sales']

    # Merge to get seller cities and states
    top_sellers = top_sellers.merge(data[['seller_id', 'seller_city', 'seller_state']], on='seller_id', how='left').drop_duplicates()

    # Sort and select top 10 sellers
    top_seller = top_sellers.sort_values(by='total_sales', ascending=False).head(10)
    
    return top_seller

with tab1:
    # Streamlit section to display the results
    st.header('Highest Frequency on Buying Customer by City in our E-Commerce')
    st.subheader('Top 10 Cities with Highest Purchase Activity')

    # Call the function and display the plot in Streamlit
    fig = plot_top_cities_by_purchase(data)
    st.pyplot(fig)
    
    st.subheader("Customer Geolocation")
    # Generate the map with customer and seller geolocation data
    geolocation_map_customer = create_geolocation_map_customer(data)
    
    # Display the map using Streamlit Folium integration
    st_folium(geolocation_map_customer, width=700, height=500)
    st.write(f"Only Display 5000 Customers Dataset")
    
    st.header('Top Product and Top Seller in Our E-Commerce')
    st.subheader('Top 5 Product')
    
    # Top 5 best-performing products by total sales
    best_products = performance_data.sort_values(by='total_sales', ascending=False).head(5)
    fig, ax = plt.subplots()

    # Create pie chart
    ax.pie(best_products['total_sales'], labels=best_products['product_category_name_english'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    ax.axis('equal')
    st.pyplot(fig)
    
    st.subheader("Top 10 Sellers by Total Sales")

    top_sellers = calculate_top_sellers(data)

    # Display results in Streamlit
    st.subheader("Top 10 Sellers")
    st.table(top_sellers)

    # Optional: Visualizing total sales of top sellers
    fig, ax = plt.subplots()
    ax.bar(top_sellers['seller_id'].astype(str), top_sellers['total_sales'], color='skyblue')
    ax.set_xlabel('Seller ID')
    ax.set_ylabel('Total Sales')
    ax.set_title('Top 10 Sellers by Total Sales')
    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    

# Tab 2 Content
with tab2:
    st.header("Sales Growth (Day, Month, Year)")
    
    sales_monthly = sales_growth(filtered_data, 'M')
    sales_yearly = sales_growth(filtered_data, 'Y')
    
    st.subheader("Monthly Sales Growth")
    st.line_chart(sales_monthly)

    st.subheader("Yearly Sales Growth")
    st.line_chart(sales_yearly)

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

# Tab 3 : Geographic & Time Insights
with tab3:
    st.header("Customer 'Prime Time' and 'Dead Time'")
    
    def analyze_prime_and_dead_time(data):
        # Convert 'order_approved_at' to datetime
        data['order_approved_at'] = pd.to_datetime(data['order_approved_at'])
        
        # Extract the hour from the order approval time
        data['order_hour'] = data['order_approved_at'].dt.hour
        
        # Group by hour to find the count of orders for each hour
        hourly_orders = data.groupby('order_hour').size()
        
        # Find prime time and dead time (e.g., most and least popular shopping hours)
        prime_time = hourly_orders.idxmax()
        dead_time = hourly_orders.idxmin()
        
        return hourly_orders, prime_time, dead_time
    
    # Analyze prime and dead times for shopping
    hourly_orders, prime_time, dead_time = analyze_prime_and_dead_time(data)
    
    # Display the result
    st.write(f"Prime Time: {prime_time}:00 (Highest Orders)")
    st.write(f"Dead Time: {dead_time}:00 (Lowest Orders)")
    
    # Plot hourly orders to visualize prime and dead time
    fig, ax = plt.subplots()
    sns.lineplot(x=hourly_orders.index, y=hourly_orders.values, ax=ax)
    ax.set_title('Orders by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Orders')
    st.pyplot(fig)

    st.header("Customer and Seller Geolocation")

    # Generate the map with customer and seller geolocation data
    geolocation_map_customer = create_geolocation_map_customer(data)
    
    # Display the map using Streamlit Folium integration
    st_folium(geolocation_map_customer, width=700, height=500)
    st.write(f"Only Display 5000 Customers Dataset")
    
    # Generate the map with customer and seller geolocation data
    geolocation_map_seller = create_geolocation_map_seller(data)
    
    # Display the map using Streamlit Folium integration
    st_folium(geolocation_map_seller, width=700, height=500)
    st.write(f"Only Display 5000 Seller Dataset")

    st.header("Payment Method by Customers")
    
    def visualize_payment_methods(data):
        # Count the number of occurrences of each payment method
        payment_counts = data['payment_type'].value_counts()
        
        # Plot the payment method distribution
        fig, ax = plt.subplots()
        sns.barplot(x=payment_counts.index, y=payment_counts.values, ax=ax)
        ax.set_title('Payment Method Distribution')
        ax.set_xlabel('Payment Method')
        ax.set_ylabel('Number of Transactions')
        return fig
    
    # Display payment method distribution
    payment_method_fig = visualize_payment_methods(data)
    st.pyplot(payment_method_fig)


# Tab 4 : Product Insights
with tab4:
    st.header("Product Flow (Shipped to Delivered)")

    def calculate_delivery_time(data):
        # Convert the necessary columns to datetime
        data['order_delivered_customer_date'] = pd.to_datetime(data['order_delivered_customer_date'], errors='coerce')
        data['order_delivered_carrier_date'] = pd.to_datetime(data['order_delivered_carrier_date'], errors='coerce')
        
        # Calculate time from carrier shipping to customer delivery
        data['shipping_to_delivery_days'] = (data['order_delivered_customer_date'] - 
                                            data['order_delivered_carrier_date']).dt.days
        return data

    # Call the function and show the average delivery time
    delivery_data = calculate_delivery_time(filtered_data)
    avg_delivery_time = delivery_data['shipping_to_delivery_days'].mean()
    
    st.write(f"Average time from shipping to delivery: {avg_delivery_time:.2f} days")
    
    # Plot the distribution of shipping to delivery times
    plt.figure(figsize=(10, 6))
    sns.histplot(delivery_data['shipping_to_delivery_days'].dropna(), bins=20, kde=True)
    plt.title("Distribution of Shipping to Delivery Times (in days)")
    plt.xlabel("Days")
    plt.ylabel("Frequency")
    st.pyplot(plt)

    st.header("Best and Worst Performing Products")

    # Top 5 best performing products by total sales
    best_products = performance_data.sort_values(by='total_sales', ascending=False).head(5)
    st.subheader("Best Performing Products")
    st.table(best_products[['product_category_name_english', 'total_sales']])
    
    # Top 5 worst performing products by number of orders
    worst_products = performance_data.sort_values(by='total_orders', ascending=True).head(5)
    st.subheader("Worst Performing Products")
    st.table(worst_products[['product_category_name_english', 'total_orders']])

    st.header("Product Reviews")

    def visualize_review_scores(data):
        # Visualize product reviews
        review_counts = data['review_score'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=review_counts.index, y=review_counts.values, palette='viridis')
        plt.title("Distribution of Product Review Scores")
        plt.xlabel("Review Score")
        plt.ylabel("Count")
        st.pyplot(plt)

    visualize_review_scores(filtered_data)
