# Zillow Home Prediction App

## 📌 Project Overview
The **Zillow Home Prediction App** is a web-based application that utilizes time-series forecasting to predict future home values for different cities and zipcodes in the United States. Built using **Streamlit** and **Facebook Prophet**, this app enables users to visualize historical housing prices, analyze market trends, and generate predictions for up to 20 years. The app also provides actionable insights and a glossary of key terminologies to help users make informed real estate investment decisions.

---

## 🚀 Features
### **Core Features**
- **City Selection**: Choose a U.S. city to analyze home value trends.
- **Zipcode-Level Analysis**: Drill down into specific zipcodes within a city for granular insights.
- **Map Visualization**: Displays the selected city’s location on an interactive map with a heatmap overlay showing price trends.
- **Time-Series Analysis**: Converts Zillow data into a structured time-series format for analysis.
- **Forecasting with Prophet**: Predicts future home values based on historical data.
- **Interactive Plots**: Displays historical trends and forecast results with Plotly.

### **Additional Features**
- **Market Health Dashboard**: Provides key metrics such as market volatility, return on investment (ROI), and risk scores.
- **Event-Based Regressors**: Allows users to simulate the impact of hypothetical events (e.g., new infrastructure projects) on home prices.
- **Comparative Analysis**: Compare home price trends across multiple cities or zipcodes.
- **Explainability and Insights**: Includes trend annotations, SHAP values (if applicable), and actionable insights to explain predictions.
- **Glossary of Terminologies**: A collapsible glossary to help users understand key terms like "Market Volatility" and "Risk Score."

---

## 🏗️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (for UI)
- **Backend**: Python
- **Machine Learning**: [Prophet](https://facebook.github.io/prophet/) (for forecasting)
- **Visualization**: [Plotly](https://plotly.com/), [Folium](https://python-visualization.github.io/folium/) (for maps), Pandas
- **Data Storage**: Google Drive (for hosting the dataset)

---

## 📂 Project Structure
```
📁 Zillow_Home_Prediction
│── 📄 zhvi_prediction.py    # Main application script
│── 📄 requirements.txt      # Python dependencies
│── 📄 README.md             # Project documentation
│── 📂 data                  # Folder containing the Zillow dataset
    │── 📄 zillow_data.csv    # Home value index data
```

---

## 🔧 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/ROBERT-ADDO-ASANTE-DARKO/Zillow-Home-Value-Index-Prediction
cd Zillow-Home-Value-Index-Prediction
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
streamlit run zhvi_prediction.py
```

---

## 📊 How It Works
1. **Select a city** from the dropdown list.
2. **Choose a zipcode** within the city for detailed analysis.
3. The app **loads and processes** historical home value data.
4. The **interactive map** shows the city's location with a heatmap overlay for price trends.
5. Users select a **forecasting period (1-20 years)**.
6. The app trains a **Prophet model** and displays future home price predictions.
7. **Interactive graphs** visualize trends, forecast components, and event impacts.
8. The **Market Health Dashboard** provides key metrics like volatility, ROI, and risk scores.
9. The **Glossary of Terminologies** helps users understand key concepts.

---

## 📷 Screenshots

| Home Page | Forecast Results | Market Health Dashboard |
|-----------|------------------|-------------------------|
| ![Screenshot 2025-02-25 001006](https://github.com/user-attachments/assets/1f5ca6fa-c79f-4cd3-84a8-f6fa83de7db5)
 | ![Screenshot 2025-02-05 023106](https://github.com/user-attachments/assets/13dc4588-68aa-47f5-8e94-42aa8f35f33f) | ![Screenshot 2025-02-25 001139](https://github.com/user-attachments/assets/0ab76e61-0378-4e2c-9359-f7dd13ae59a4)
|

---

## 🔍 Model and Dataset
### **Model: Facebook Prophet**
- **Prophet** is an open-source time-series forecasting tool developed by Facebook. It is designed for simplicity and flexibility, making it ideal for predicting trends in home prices.
- The model decomposes time-series data into **trend**, **seasonality**, and **holiday effects** to generate accurate forecasts.
- Users can add **event-based regressors** to simulate the impact of external factors (e.g., new infrastructure projects) on home prices.

### **Dataset: Zillow Home Value Index (ZHVI)**
- The dataset contains historical home value data for various cities and zipcodes across the United States.
- It includes metrics such as **RegionID**, **City**, **State**, **Metro**, **CountyName**, and **SizeRank**.
- The data is processed into a time-series format for analysis and forecasting.

---

## 🌟 Potential Impact
### **Real Estate Investments**
- **Informed Decision-Making**: The app provides actionable insights and forecasts, helping investors identify high-growth areas and assess risks.
- **Event Simulation**: Users can simulate the impact of hypothetical events (e.g., new infrastructure projects) to evaluate potential ROI.
- **Market Health Metrics**: Metrics like volatility, ROI, and risk scores help users assess the stability and profitability of a market.

### **Homebuyers and Sellers**
- **Price Trends**: Homebuyers and sellers can use the app to understand historical price trends and forecast future values.
- **Comparative Analysis**: Users can compare trends across cities or zipcodes to make informed decisions.

### **Urban Planning and Policy**
- **Data-Driven Insights**: Policymakers and urban planners can use the app to analyze housing market trends and plan infrastructure projects.

---

## 🔍 Future Improvements
- **Multi-Model Support**: Incorporate additional machine learning models (e.g., ARIMA, LSTM) for comparison.
- **Real-Time Data Integration**: Connect to Zillow’s API for real-time data updates.
- **Advanced Visualizations**: Add more interactive visualizations like 3D maps and animated trend graphs.
- **Exportable Reports**: Allow users to export forecasts and insights as PDF or CSV files.

---

## 🤝 Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 📬 Contact
For any questions or collaboration opportunities, reach out via **[LinkedIn](https://www.linkedin.com/in/robert-agyekum-addo-3597461b4)**.
