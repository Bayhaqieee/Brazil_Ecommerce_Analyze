import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency
import plotly.express as px
import plotly.graph_objects as go

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
    # Group by city and count the number of purchases
    top_cities = data.groupby('customer_city').size().reset_index(name='Number of Purchases')
    top_cities = top_cities.sort_values(by='Number of Purchases', ascending=False).head(10)
    
    # Create an interactive bar chart
    fig = px.bar(top_cities, x='customer_city', y='Number of Purchases', 
                 title='Top 10 Cities with Highest Purchase Activity',
                 labels={'customer_city': 'City', 'Number of Purchases': 'Number of Purchases'},
                 color='Number of Purchases', 
                 color_continuous_scale=px.colors.sequential.Rainbow)

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

def calculate_monthly_growth(data):
    # Convert to datetime if not already
    data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
    
    # Extract year and month from purchase timestamp
    data['purchase_month'] = data['order_purchase_timestamp'].dt.to_period('M')
    
    # Group by month and count the number of orders
    purchase_growth = data.groupby('purchase_month').size()
    
    # Convert index to string format
    purchase_growth.index = purchase_growth.index.astype(str)  # Convert Period to string
    
    # Calculate the overall percentage growth or decline
    initial_orders = purchase_growth.iloc[0]  # Orders in the first month
    final_orders = purchase_growth.iloc[-1]   # Orders in the last month
    overall_percentage_change = ((final_orders - initial_orders) / initial_orders) * 100

    return purchase_growth, overall_percentage_change

def analyze_delivery_efficiency(data):
    # Ensure the correct columns are in the dataset
    if 'order_approved_at' not in data.columns or 'order_delivered_customer_date' not in data.columns:
        raise ValueError("Data must contain 'order_approved_at' and 'order_delivered_customer_date' columns.")

    # Convert to datetime if not already
    data['order_purchase_timestamp'] = pd.to_datetime(data['order_approved_at'], errors='coerce')
    data['order_delivered_customer_date'] = pd.to_datetime(data['order_delivered_customer_date'], errors='coerce')

    # Filter for delivered orders and calculate the time taken
    data['delivery_duration'] = (data['order_delivered_customer_date'] - data['order_purchase_timestamp']).dt.days

    # Filter out any rows with NaT in delivery_duration
    data = data[data['delivery_duration'].notna()]

    # Summary statistics on delivery time
    delivery_summary = data['delivery_duration'].describe()

    # Calculate min and max delivery durations
    min_delivery_time = delivery_summary['min']
    max_delivery_time = delivery_summary['max']

    # Define efficiency thresholds
    threshold_min = min_delivery_time + (max_delivery_time - min_delivery_time) * 0.25  # Top 25% efficient
    threshold_max = max_delivery_time - (max_delivery_time - min_delivery_time) * 0.25  # Bottom 25% inefficient

    # Count the number of deliveries in each efficiency category
    efficient_deliveries = data[data['delivery_duration'] <= threshold_min].shape[0]
    inefficient_deliveries = data[data['delivery_duration'] >= threshold_max].shape[0]
    total_deliveries = data.shape[0]

    # Calculate the efficiency percentages for efficient and inefficient deliveries
    efficient_percentage = (efficient_deliveries / total_deliveries) * 100 if total_deliveries > 0 else 0
    inefficient_percentage = (inefficient_deliveries / total_deliveries) * 100 if total_deliveries > 0 else 0

    # General efficiency rates based on average delivery time
    average_delivery_time = delivery_summary['mean']
    general_efficient_deliveries = data[data['delivery_duration'] <= average_delivery_time].shape[0]
    general_inefficient_deliveries = data[data['delivery_duration'] > average_delivery_time].shape[0]

    # Calculate the general efficiency percentages
    general_efficient_percentage = (general_efficient_deliveries / total_deliveries) * 100 if total_deliveries > 0 else 0
    general_inefficient_percentage = (general_inefficient_deliveries / total_deliveries) * 100 if total_deliveries > 0 else 0

    return {
        "total_deliveries": total_deliveries,
        "efficient_deliveries": efficient_deliveries,
        "inefficient_deliveries": inefficient_deliveries,
        "efficient_percentage": efficient_percentage,
        "inefficient_percentage": inefficient_percentage,
        "general_efficient_percentage": general_efficient_percentage,
        "general_inefficient_percentage": general_inefficient_percentage
    }

def analyze_purchase_activity(data):
    # Convert 'order_purchase_timestamp' to datetime
    data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])

    # Extract hour from purchase timestamp
    data['purchase_hour'] = data['order_purchase_timestamp'].dt.hour

    # Group by hour to see purchase activity
    purchase_activity = data.groupby('purchase_hour').size()

    # Find the Prime Time (hour with the highest purchase activity)
    prime_time_hour = purchase_activity.idxmax()
    prime_time_count = purchase_activity.max()

    # Find the Dead Time (hour with the lowest purchase activity)
    dead_time_hour = purchase_activity.idxmin()
    dead_time_count = purchase_activity.min()

    return purchase_activity, prime_time_hour, prime_time_count, dead_time_hour, dead_time_count

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

with tab1:
    # Streamlit section to display the results
    st.header('Highest Frequency on Buying Customer by City in our E-Commerce')

    # Call the function and display the plot in Streamlit
    fig_cities = plot_top_cities_by_purchase(data)
    st.plotly_chart(fig_cities)
    
    st.header('Top Product and Top Seller in Our E-Commerce')
    
    best_products = performance_data.sort_values(by='total_sales', ascending=False).head(5)
    total_sales_all = performance_data['total_sales'].sum()
    total_sales_best = best_products['total_sales'].sum()
    total_sales_other = total_sales_all - total_sales_best
    pie_data = pd.DataFrame({
        'Category': list(best_products['product_category_name_english']) + ['Other'],
        'Sales': list(best_products['total_sales']) + [total_sales_other]
    })
    fig = px.pie(pie_data, names='Category', values='Sales',
                title='Top 5 Best-Performing Products vs Other Products',
                hole=0.3)  # This creates a donut chart
    st.plotly_chart(fig)
    st.subheader("Top 10 Sellers by Total Sales")

    top_sellers = calculate_top_sellers(data)
    st.table(top_sellers)
    
    st.header('Product Shipment Flow Effectivity')
    analysis_results = analyze_delivery_efficiency(data)


    # Optional: Visualizing delivery efficiency
    labels = ['Efficient Deliveries', 'Inefficient Deliveries']
    sizes = [analysis_results['general_efficient_percentage'], analysis_results['general_inefficient_percentage']]

    # Create an interactive pie chart for delivery efficiency
    fig_efficiency = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3,
                                            marker=dict(colors=['#66c2a5', '#fc8d62']),
                                            textinfo='percent+label')])
    fig_efficiency.update_layout(title_text='Delivery Efficiency Summary')
    st.plotly_chart(fig_efficiency)

    st.write(f"Total Deliveries: {analysis_results['total_deliveries']}")
    st.write(f"General Efficient Deliveries: {analysis_results['general_efficient_percentage']:.2f}%")
    st.write(f"General Inefficient Deliveries: {analysis_results['general_inefficient_percentage']:.2f}%")
    
    st.header('Most Payment Method Used')
    payment_methods = data.groupby('payment_type').size().sort_values(ascending=False)

    pull = [0.1 if payment_methods.index[i] == payment_methods.idxmax() else 0 for i in range(len(payment_methods))]

    # Use a color palette for better visual appeal
    colors = px.colors.qualitative.Set3  # A color set that works well with categorical data

    fig_payment = go.Figure(data=[go.Pie(labels=payment_methods.index, 
                                        values=payment_methods.values,
                                        pull=pull,
                                        hole=.3,
                                        marker=dict(colors=colors),
                                        textinfo='percent+label')])

    fig_payment.update_layout(title_text='Most Used Payment Methods')
    st.plotly_chart(fig_payment)
    
    st.header('Customer Prime and Dead Time')
    purchase_activity, prime_time_hour, prime_time_count, dead_time_hour, dead_time_count = analyze_purchase_activity(data)

    # Set up the Streamlit app
    st.subheader('Purchase Activity Insights')
    fig_activity = go.Figure()

    # Add a line trace to the figure
    fig_activity.add_trace(go.Scatter(
        x=purchase_activity.index,
        y=purchase_activity.values,
        mode='lines+markers',
        name='Number of Purchases',
        line=dict(color='royalblue'),
        marker=dict(size=8)
    ))

    # Update layout
    fig_activity.update_layout(
        title='Purchase Activity by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Number of Purchases',
        xaxis=dict(
            tickmode='linear',  # Ensures ticks are shown linearly
            tickvals=list(range(24)),  # Ensure all hours (0-23) are shown
        ),
        template='plotly_white'  # Choose a clean layout template
    )

    # Show the interactive plot in Streamlit
    st.plotly_chart(fig_activity)

    # Display the Prime Time and Dead Time
    st.write(f"**Prime Time:** {prime_time_hour}:00 with {prime_time_count} purchases.")
    st.write(f"**Dead Time:** {dead_time_hour}:00 with {dead_time_count} purchases.")
    
    purchase_growth, overall_percentage_change = calculate_monthly_growth(data)

    # Set up the Streamlit app
    st.header('Monthly Purchase Growth Analysis')

    # Create an interactive line plot
    fig_growth = go.Figure()

    # Add a line trace to the figure
    fig_growth.add_trace(go.Scatter(
        x=purchase_growth.index,
        y=purchase_growth.values,
        mode='lines+markers',
        name='Number of Purchases',
        line=dict(color='royalblue'),
        marker=dict(size=8)
    ))

    fig_growth.update_layout(
        title='Growth of Purchases Over Time',
        xaxis_title='',  # Leave the title blank to hide the title
        yaxis_title='Number of Purchases',
        xaxis=dict(
            tickmode='linear',  # Ensures ticks are shown linearly
            showticklabels=False,  # Hide x-axis labels
        ),
        xaxis_tickangle=-45,  # Rotate x-axis labels (will not be visible now)
        template='plotly_white'  # Choose a clean layout template
    )

    # Show the interactive plot in Streamlit
    st.plotly_chart(fig_growth)

    # Display overall purchase growth
    st.subheader('Overall Purchase Growth')
    st.write(f"**Overall Purchase Growth from first to last month:** {overall_percentage_change:.2f}%")

    st.subheader("Customer Geolocation")
    # Generate the map with customer and seller geolocation data
    geolocation_map_customer = create_geolocation_map_customer(data)
    st_folium(geolocation_map_customer, width=700, height=500, key='data_key')
    

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
    st_folium(geolocation_map_customer, width=700, height=500, key='customer_key')
    st.write(f"Only Display 5000 Customers Dataset")
    
    # Generate the map with customer and seller geolocation data
    geolocation_map_seller = create_geolocation_map_seller(data)
    
    # Display the map using Streamlit Folium integration
    st_folium(geolocation_map_seller, width=700, height=500, key='seller_key')
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
