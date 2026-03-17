# LSTM Stock Price Forecaster

A Streamlit-based web application that predicts future stock prices using a Long Short-Term Memory (LSTM) neural network model. The app features a clean, focused "trading dashboard" aesthetic for visualizing historical data and future forecasts.

**📺 Live Demo:** [https://lstm-priceforecaster.streamlit.app/](https://lstm-priceforecaster.streamlit.app/)

## Features
*   **Predictive Modeling**: Utilizes an LSTM model (`stock_price_lstm_model.h5`) trained on historical stock data.
*   **Real-time Data Fetching**: Pulls live historical stock and market data using `yfinance`.
*   **Clean UI Dashboard**: A focused, modern trading interface stripping away clutter to emphasize core metrics and trend visualizations.
*   **Dynamic Visualizations**: Interactive line charts powered by Plotly to compare historical trajectories against LSTM forecasts.

## Tech Stack
*   **Frontend**: Streamlit
*   **Backend Logic**: Python 3.11
*   **Machine Learning**: TensorFlow (Keras), Scikit-Learn (MinMax Scaling)
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly

## Deployment Details (Streamlit Cloud)
The application is configured specifically for Streamlit Cloud deployment:
*   **Environment**: Uses `tensorflow-cpu==2.17.0` to minimize server instance bloat while maintaining Keras 3 compatibility.
*   **Python Version**: Enforced via `runtime.txt` to align with the framework dependencies.
*   **Keras 3 Compatibility**: Includes runtime monkeypatching in `app.py` to seamlessly load older model configurations (stripping unrecognized kwargs like `quantization_config` during deserialization) without requiring model retraining.

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iprashantraj/LSTM_proj.git
   cd LSTM_proj
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
