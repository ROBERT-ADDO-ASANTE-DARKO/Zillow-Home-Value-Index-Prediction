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

st.title("Zillow Home Prediction App")

# Direct file ID from Google Drive link
file_id = "1wcabOuayxwGUzj_cd5k5fIboKFBce4yj"
download_url = f"https://drive.google.com/uc?id={file_id}"

@st.cache_data
def load_data():
    output = "zillow_data.csv"
    gdown.download(download_url, output, quiet=False)
    return pd.read_csv(output)

data = load_data()

data.rename({"RegionName": "Zipcode"}, axis="columns", inplace=True)

# Define a dictionary with city coordinates
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

# Get list of unique cities
cities = data["City"].unique()
selected_city = st.selectbox("Select a city for prediction:", cities)

# Filter data based on selected city
city_data = data.loc[data["City"] == selected_city]

# Get list of unique zipcodes for the selected city
zipcodes = city_data["Zipcode"].unique()
selected_zipcode = st.selectbox("Select a zipcode for detailed analysis:", zipcodes)

# Filter data based on selected zipcode
zipcode_data = city_data.loc[city_data["Zipcode"] == selected_zipcode]

# Retrieve coordinates for the selected city from the dictionary
coordinates = city_coordinates.get(selected_city, (None, None))

# Create a folium map centered on the selected city
if coordinates[0] is not None and coordinates[1] is not None:
    m = folium.Map(location=coordinates, zoom_start=11)

    # Prepare data for the heatmap
    heatmap_data = []
    for _, row in city_data.iterrows():
        lat, lon = city_coordinates.get(row["City"], (None, None))
        if lat is not None and lon is not None:
            # Use the latest home value as the weight for the heatmap
            latest_value = row.iloc[-1]  # Assuming the last column is the latest value
            heatmap_data.append([lat, lon, latest_value])

    # Add heatmap layer
    HeatMap(heatmap_data, radius=15).add_to(m)

    # Display the map
    st.subheader(f"Heatmap of Home Prices in {selected_city}")
    folium_static(m)
else:
    st.write("Coordinates not available for the selected city.")

n_years = st.slider("Years of prediction:", 1, 20)
period = n_years * 365

def melt_data(df):
    melted = pd.melt(df, id_vars=["RegionID", "Zipcode", "City", "State", "Metro", "CountyName", "SizeRank"], var_name="time")
    melted["time"] = pd.to_datetime(melted["time"], infer_datetime_format=True)
    melted = melted.dropna(subset=["value"])
    return melted.groupby("time").aggregate({"value": "mean"}).reset_index()

data_load_state = st.text("Loading data...")
city_data_melted = melt_data(city_data)
zipcode_data_melted = melt_data(zipcode_data)
data_load_state.text("Loading data...done!")

st.subheader(f"Raw data for {selected_city}")
st.write(city_data_melted.tail())

st.subheader(f"Raw data for Zipcode {selected_zipcode}")
st.write(zipcode_data_melted.tail())

def plot_raw_data(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['value'], name=title, line_color="red"))
    fig.layout.update(title_text=title, width=600, height=600, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(city_data_melted, f"{selected_city} Home Value Index")
plot_raw_data(zipcode_data_melted, f"Zipcode {selected_zipcode} Home Value Index")

# Calculate market health metrics
volatility = city_data_melted["value"].std()
roi = (city_data_melted["value"].iloc[-1] - city_data_melted["value"].iloc[0]) / city_data_melted["value"].iloc[0]
risk_score = (volatility * 0.6) + (roi * 0.4)  # Example formula

# Display metrics
st.subheader("Market Health Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Volatility", f"{volatility:.2f}")
col2.metric("ROI", f"{roi:.2%}")
col3.metric("Risk Score", f"{risk_score:.2f}")

# Forecasting for the selected zipcode
df_train = zipcode_data_melted[["time", "value"]]
df_train = df_train.rename(columns={"time": "ds", "value": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Add event-based regressors
event_name = st.text_input("Add a hypothetical event (e.g., 'New infrastructure project'):")
if event_name:
    event_date = st.date_input("Event date:")
    event_impact = st.slider("Event impact (positive or negative):", -100, 100, 0)

    # Convert event date to datetime
    event_date = pd.to_datetime(event_date)

    # Validate event date range
    if event_date < df_train["ds"].min() or event_date > df_train["ds"].max():
        st.error(f"Event date must be between {df_train['ds'].min()} and {df_train['ds'].max()}.")
    else:
        # Add the event to the Prophet model
        model.add_regressor(event_name)
        df_train[event_name] = 0
        df_train.loc[df_train["ds"] >= event_date, event_name] = event_impact

        # Re-train the model
        model.fit(df_train)
        future[event_name] = 0  # Add the event column to the future DataFrame
        future.loc[future["ds"] >= event_date, event_name] = event_impact

        # Show updated forecast
        forecast = model.predict(future)
        st.subheader(f"Forecast with '{event_name}' Event")
        st.write(forecast.tail(20))

st.subheader(f"Forecast data for Zipcode {selected_zipcode}")
st.write(forecast.tail(20))

st.write("Forecast data")
# Add annotations to the forecast plot
fig1 = plot_plotly(model, forecast, figsize=(600, 600))
fig1.add_annotation(
    x="2025-01-01", y=forecast.loc[forecast["ds"] == "2025-01-01", "yhat"].values[0],
    text="Peak in 2025 due to economic growth",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-40
)
st.plotly_chart(fig1)

# Generate insights
growth_rate = (forecast["yhat"].iloc[-1] - forecast["yhat"].iloc[0]) / forecast["yhat"].iloc[0]
insight_text = f"""
- Home prices in {selected_city} are expected to grow by **{growth_rate:.2%}** over the next {n_years} years.
- The market is currently **{'volatile' if volatility > 100 else 'stable'}**.
- Zipcode {selected_zipcode} has shown **{'above-average' if roi > 0.1 else 'below-average'}** ROI compared to the city average.
"""

# Display insights
st.subheader("Key Insights")
st.markdown(insight_text)

st.write("Forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)
