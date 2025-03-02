# Bitcoin Buy/Sell Signal Predictor

This project is a **Bitcoin Buy/Sell Signal Predictor** using machine learning and technical analysis. The model uses a **Random Forest Classifier** to predict Bitcoin's price movement based on **RSI** (Relative Strength Index) and **SMA** (Simple Moving Average) indicators. The script fetches live Bitcoin price data from the **CoinGecko API**, trains the model, and visualizes buy/sell signals on an interactive graph using **Dash** and **Plotly**.

---

## Features
- **Real-time Bitcoin Price Data**: Fetches live Bitcoin prices from the CoinGecko API.
- **Buy/Sell Signal Prediction**: Uses a Random Forest model to predict Bitcoin price movement.
- **Interactive Graph**: Displays Bitcoin price with buy (green) and sell (red) signals.
- **Data Fetching & Model Training**: Updates data every minute and retrains the model periodically.
- **RSI and SMA Indicators**: Calculates RSI and SMA to generate the features for model training.

---

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/cr0mb/bitcoin-signal-predictor.git
    cd bitcoin-signal-predictor
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Running the Application

To run the app, simply execute the Python script:

```bash
python app.py
