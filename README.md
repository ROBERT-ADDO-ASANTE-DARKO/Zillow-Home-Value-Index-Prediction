# Zillow Home Prediction App

## 📌 Project Overview
The **Zillow Home Prediction App** is a web-based application that utilizes time-series forecasting to predict future home values for different cities in the United States. It is built using **Streamlit** and **Facebook Prophet**, enabling users to visualize historical housing prices and generate predictions for up to 20 years.

## 🚀 Features
- **City Selection**: Choose a U.S. city to analyze home value trends.
- **Map Visualization**: Displays the selected city’s location on an interactive map.
- **Time-Series Analysis**: Converts Zillow data into a structured time-series format.
- **Forecasting with Prophet**: Predicts future home values based on historical data.
- **Interactive Plots**: Displays historical trends and forecast results with Plotly.

## 🏗️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/) (for UI)
- **Backend**: Python
- **Machine Learning**: [Prophet](https://facebook.github.io/prophet/) (for forecasting)
- **Visualization**: [Plotly](https://plotly.com/), Pandas

## 📂 Project Structure
```
📁 Zillow_Home_Prediction
│── 📄 zhvi_prediction.py    # Main application script
│── 📄 requirements.txt      # Python dependencies
│── 📄 README.md             # Project documentation
│── 📂 data                  # Folder containing the Zillow dataset
    │── 📄 zillow_data.csv    # Home value index data
```

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

## 📊 How It Works
1. **Select a city** from the dropdown list.
2. The app **loads and processes** historical home value data.
3. The **interactive map** shows the city's location.
4. Users select a **forecasting period (1-20 years)**.
5. The app trains a **Prophet model** and displays future home price predictions.
6. **Interactive graphs** visualize trends and forecast components.

## 📷 Screenshots
| Home Page | Forecast Results |
|-----------|----------------|
| ![Screenshot 2025-02-05 022945](https://github.com/user-attachments/assets/90f6de11-c03b-4c20-8f4e-97fe2e90d089)
 | ![Forecast]() |

## 🔍 Future Improvements
- Add more **data sources** for improved accuracy.
- Incorporate **machine learning models** beyond Prophet.
- Enhance UI with **advanced visualizations** and filtering options.

## 🤝 Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## 📜 License
This project is open-source and available under the **MIT License**.

## 📬 Contact
For any questions or collaboration opportunities, reach out via **[LinkedIn](https://www.linkedin.com/in/robert-agyekum-addo-3597461b4)**.
