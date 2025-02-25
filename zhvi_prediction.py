import streamlit as st
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import gdown
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import random
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Function to get coordinates
def get_coordinates(zipcode):
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        location = geolocator.geocode({"postalcode": zipcode, "country": "USA"}, timeout=10)
        if location:
            return location.latitude, location.longitude
    except GeocoderTimedOut:
        time.sleep(1)
        return get_coordinates(zipcode)
    return None, None

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric label {
        color: black !important;  /* Metric label color */
    }
    .stMetric div {
        color: black !important;  /* Metric value color */
    }
    .market-overview {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.title("🏠 Zillow Home Price Prediction")
st.markdown("---")

# Sidebar for user inputs and glossary
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Data loading
    file_id = "1wcabOuayxwGUzj_cd5k5fIboKFBce4yj"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    @st.cache_data
    def load_data():
        output = "zillow_data.csv"
        gdown.download(download_url, output, quiet=False)
        return pd.read_csv(output)

    data = load_data()
    data.rename({"RegionName": "Zipcode"}, axis="columns", inplace=True)

    # Add coordinates if missing
    if "Latitude" not in data.columns or "Longitude" not in data.columns:
        data["Latitude"], data["Longitude"] = zip(*data["Zipcode"].apply(lambda x: get_coordinates(str(x))))
        data.to_csv("zillow_data_with_coords.csv", index=False)
    
    # Sidebar
    st.sidebar.header("Analysis Parameters")
    cities = data["City"].unique()
    selected_city = st.sidebar.selectbox("Select a City:", cities)
    
    # Filter data based on selected city
    city_data = data[data["City"] == selected_city]
    zipcodes = city_data["Zipcode"].unique()
    selected_zipcode = st.sidebar.selectbox("Select a Zipcode:", zipcodes)
    
    # Prediction timeframe
    n_years = st.slider("Prediction Timeframe (Years):", 1, 20, 5)
    
    # Event analysis
    st.subheader("Event Analysis")
    event_name = st.text_input("Hypothetical Event:", placeholder="e.g., New infrastructure project")
    if event_name:
        event_date = st.date_input("Event Date:")
        event_impact = st.slider("Event Impact:", -100, 100, 0)

    # Glossary Section
    st.markdown("---")
    with st.expander("📖 Glossary of Terms"):
        st.markdown("""
            **Market Volatility**: A measure of how much home prices fluctuate over time. Higher volatility indicates greater price changes.
            
            **Return on Investment (ROI)**: The percentage increase in home prices over a specific period. It measures the profitability of an investment.
            
            **Risk Score**: A composite score that evaluates the risk level of investing in a particular market. It considers factors like volatility and ROI.
            
            **Forecast**: A prediction of future home prices based on historical data and trends.
            
            **Event Impact**: The effect of a hypothetical event (e.g., new infrastructure) on home prices. Positive impact increases prices, while negative impact decreases them.
        """)

# Market Overview Section
st.markdown('<div class="market-overview">', unsafe_allow_html=True)
st.header("📊 Market Overview")

# Data processing functions
def melt_data(df):
    melted = pd.melt(df, id_vars=["RegionID", "Zipcode", "City", "State", "Metro", "CountyName", "SizeRank"], var_name="time")
    melted["time"] = pd.to_datetime(melted["time"], infer_datetime_format=True)
    melted = melted.dropna(subset=["value"])
    return melted.groupby("time").aggregate({"value": "mean"}).reset_index()

# Process data
city_data_melted = melt_data(city_data)
zipcode_data_melted = melt_data(data.loc[data["Zipcode"] == selected_zipcode])

# Market Health Metrics
volatility = city_data_melted["value"].std()
roi = (city_data_melted["value"].iloc[-1] - city_data_melted["value"].iloc[0]) / city_data_melted["value"].iloc[0]
risk_score = (volatility * 0.6) + (roi * 0.4)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Market Volatility", f"{volatility:,.2f}")
metric_col2.metric("Return on Investment", f"{roi:.2%}")
metric_col3.metric("Risk Score", f"{risk_score:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# Map visualization
st.header("🗺️ Zip Code Map")
m = folium.Map(location=[city_data["Latitude"].mean(), city_data["Longitude"].mean()], zoom_start=12)

# Add heatmap
heatmap_data = city_data.dropna(subset=["Latitude", "Longitude", "value"])
HeatMap(heatmap_data[["Latitude", "Longitude", "value"]].values, radius=15).add_to(m)

# Add markers
for _, row in heatmap_data.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"Zipcode: {row['Zipcode']}<br>Latest Value: ${row.iloc[-1]:,.2f}",
        tooltip=f"Zipcode: {row['Zipcode']}"
    ).add_to(m)

folium_static(m, width=800)

# Price Analysis Section
st.markdown("---")
st.header("💰 Price Analysis")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader(f"{selected_city} Price Trends")
    fig_city = go.Figure()
    fig_city.add_trace(go.Scatter(x=city_data_melted['time'], y=city_data_melted['value'], 
                                 name='City Average', line_color="blue"))
    fig_city.layout.update(title_text="City-wide Price Trends", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_city, use_container_width=True)

with chart_col2:
    st.subheader(f"Zipcode {selected_zipcode} Price Trends")
    fig_zip = go.Figure()
    fig_zip.add_trace(go.Scatter(x=zipcode_data_melted['time'], y=zipcode_data_melted['value'], 
                                name='Zipcode', line_color="red"))
    fig_zip.layout.update(title_text="Zipcode Price Trends", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_zip, use_container_width=True)

# Forecast Section
st.markdown("---")
st.header("🔮 Price Forecast")

# Prepare forecast data
period = n_years * 365
df_train = zipcode_data_melted[["time", "value"]].rename(columns={"time": "ds", "value": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)

# Add event impact if specified
if event_name:
    event_date = pd.to_datetime(event_date)
    if event_date >= df_train["ds"].min() and event_date <= df_train["ds"].max():
        model.add_regressor(event_name)
        df_train[event_name] = 0
        df_train.loc[df_train["ds"] >= event_date, event_name] = event_impact
        model.fit(df_train)
        future[event_name] = 0
        future.loc[future["ds"] >= event_date, event_name] = event_impact

# Generate and display forecast
forecast = model.predict(future)
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Forecast components
st.subheader("Forecast Components")
fig_components = model.plot_components(forecast)
st.write(fig_components)

# Key Insights
st.markdown("---")
st.header("📈 Key Insights")
growth_rate = (forecast["yhat"].iloc[-1] - forecast["yhat"].iloc[0]) / forecast["yhat"].iloc[0]
col_insights1, col_insights2 = st.columns(2)

with col_insights1:
    st.markdown(f"""
    ### Market Trends
    - Expected growth: **{growth_rate:.2%}** over {n_years} years
    - Market volatility: **{'High' if volatility > 100 else 'Low'}**
    - ROI performance: **{'Above' if roi > 0.1 else 'Below'}** average
    """)

with col_insights2:
    st.markdown(f"""
    ### Risk Assessment
    - Risk level: **{'High' if risk_score > 50 else 'Moderate' if risk_score > 30 else 'Low'}**
    - Market stability: **{'Volatile' if volatility > 100 else 'Stable'}**
    - Investment outlook: **{'Favorable' if growth_rate > 0.1 else 'Moderate' if growth_rate > 0 else 'Cautious'}**
    """)
