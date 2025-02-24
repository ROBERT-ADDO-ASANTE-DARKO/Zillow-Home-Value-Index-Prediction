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

# Page configuration
st.set_page_config(
    page_title="Zillow Home Price Prediction",
    page_icon="🏠",
    layout="wide"
)

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
    .market-overview {
        font-color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.title("🏠 Zillow Home Price Prediction")
st.markdown("---")

# Sidebar for user inputs
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

    # City selection
    cities = data["City"].unique()
    selected_city = st.selectbox("Select a City:", cities)
    
    # Filter data based on selected city
    city_data = data.loc[data["City"] == selected_city]
    
    # Zipcode selection
    zipcodes = city_data["Zipcode"].unique()
    selected_zipcode = st.selectbox("Select a Zipcode:", zipcodes)
    
    # Prediction timeframe
    n_years = st.slider("Prediction Timeframe (Years):", 1, 20, 5)
    
    # Event analysis
    st.subheader("Event Analysis")
    event_name = st.text_input("Hypothetical Event:", placeholder="e.g., New infrastructure project")
    if event_name:
        event_date = st.date_input("Event Date:")
        event_impact = st.slider("Event Impact:", -100, 100, 0)

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
st.markdown("---")
st.header("🗺️ Geographic View")

# City coordinates dictionary (your existing coordinates)
city_coordinates = {
    "New York": (40.7128, -74.0060),
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "Seattle": (47.6062, -122.3321),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Philadelphia": (39.9526, -75.1652),
    "Phoenix": (33.4484, -112.0740),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
    "Austin": (30.2672, -97.7431),
    "Jacksonville": (30.3322, -81.6557),
    "Fort Worth": (32.7555, -97.3331),
    "Columbus": (39.9612, -82.9988),
    "Indianapolis": (39.7684, -86.1581),
    "Charlotte": (35.2271, -80.8431),
    "Denver": (39.7392, -104.9903),
    "Washington": (38.8951, -77.0369),
    "Boston": (42.3601, -71.0589),
    "El Paso": (31.7619, -106.4850),
    "Detroit": (42.3314, -83.0458),
    "Nashville": (36.1627, -86.7816),
    "Baltimore": (39.2904, -76.6122),
    "Oklahoma City": (35.4676, -97.5164),
    "Las Vegas": (36.1699, -115.1398),
    "Louisville": (38.2527, -85.7585),
    "Milwaukee": (43.0389, -87.9065),
    "Albuquerque": (35.0844, -106.6504),
    "Tucson": (32.2226, -110.9747),
    "Fresno": (36.7378, -119.7871),
    "Sacramento": (38.58, -121.49),
    "Kansas City": (39.0997, -94.5786),
    "Mesa": (33.4152, -111.8315),
    "Virginia Beach": (36.8529, -75.9780),
    "Atlanta": (33.7490, -84.3880),
    "Colorado Springs": (38.8339, -104.8214),
    "Omaha": (41.2565, -95.9345),
    "Raleigh": (35.7796, -78.6382),
    "Miami": (25.7617, -80.1918),
    "Cleveland": (41.4993, -81.6944),
    "Tulsa": (36.1540, -95.9928),
    "Oakland": (37.8049, -122.2711),
    "Minneapolis": (44.9778, -93.2650),
    "Wichita": (37.6872, -97.3301),
    "Arlington": (32.7357, -97.1081),
    "Bakersfield": (35.3733, -119.0187),
    "New Orleans": (29.9511, -90.0715),
    "Honolulu": (21.3069, -157.8583),
    "Anaheim": (33.8366, -117.9143),
    "Tampa": (27.9506, -82.4572)
}

# Create map
coordinates = city_coordinates.get(selected_city)
if coordinates:
    m = folium.Map(location=coordinates, zoom_start=11)
    heatmap_data = []
    for _, row in city_data.iterrows():
        lat, lon = city_coordinates.get(row["City"], (None, None))
        if lat is not None and lon is not None:
            latest_value = row.iloc[-1]
            heatmap_data.append([lat, lon, latest_value])
    HeatMap(heatmap_data, radius=15).add_to(m)
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
