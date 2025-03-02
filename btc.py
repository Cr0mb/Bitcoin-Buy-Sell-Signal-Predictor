import time
import threading
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
import json
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = Dash(__name__, suppress_callback_exceptions=True)

# Global variables
prices_history, dates_history = [], []
buy_signals, sell_signals = [], []
model = None
data_file = "buy_sell_data.json"

def fetch_bitcoin_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1'
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            prices = [entry[1] for entry in data['prices']]
            timestamps = [entry[0] for entry in data['prices']]
            dates = [datetime.fromtimestamp(ts / 1000, timezone.utc) for ts in timestamps]
            logging.info(f"Data fetched successfully (Attempt {attempt + 1})")
            return dates, prices
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            time.sleep(2 ** attempt)
    logging.error("Max retries exceeded. Could not fetch data.")
    return [], []

def save_data_to_json():
    """Save buy/sell signals and prices to a JSON file"""
    data = {
        "prices_history": prices_history,
        "dates_history": [str(date) for date in dates_history],
        "buy_signals": buy_signals,
        "sell_signals": sell_signals
    }
    try:
        with open(data_file, 'w') as f:
            json.dump(data, f)
        logging.info(f"Data saved to {data_file}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

def load_data_from_json():
    """Load buy/sell signals and prices from a JSON file"""
    global prices_history, dates_history, buy_signals, sell_signals
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            prices_history = data.get("prices_history", [])
            dates_history = [datetime.fromisoformat(date) for date in data.get("dates_history", [])]
            buy_signals = data.get("buy_signals", [])
            sell_signals = data.get("sell_signals", [])
        logging.info(f"Data loaded from {data_file}")
    except FileNotFoundError:
        logging.warning(f"{data_file} not found. Starting with empty data.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")

def calculate_rsi(prices, window=14):
    df = pd.DataFrame({'price': prices})
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna().values

def calculate_sma(prices, window=14):
    return pd.Series(prices).rolling(window).mean()  # Return as Series, not ndarray

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = pd.Series(prices).ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = pd.Series(prices).ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices, window=20):
    sma = calculate_sma(prices, window)
    rolling_std = pd.Series(prices).rolling(window).std()
    
    # Align the rolling_std and sma to the full length of prices by padding the beginning
    rolling_std = rolling_std.iloc[window-1:].reset_index(drop=True)
    sma = sma.iloc[window-1:].reset_index(drop=True)

    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    
    return upper_band, lower_band

def generate_features(prices):
    rsi_values = calculate_rsi(prices)
    sma_values = calculate_sma(prices)
    macd, signal = calculate_macd(prices)
    upper_band, lower_band = calculate_bollinger_bands(prices)

    # Ensure all arrays are of the same length
    min_length = min(len(rsi_values), len(sma_values), len(macd), len(signal), len(upper_band), len(lower_band), len(prices) - 15)

    # Slice arrays to ensure they have the same length
    features = np.column_stack((
        rsi_values[:min_length], 
        sma_values[:min_length], 
        macd[:min_length], 
        signal[:min_length], 
        upper_band[:min_length], 
        lower_band[:min_length]
    ))

    labels = (np.array(prices[15:15 + min_length]) > np.array(prices[15 - 1:15 - 1 + min_length])).astype(int)
    
    return features, labels


def train_model(prices):
    features, labels = generate_features(prices)
    if len(features) < 20:
        logging.warning("Not enough data to train model.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    logging.info(f"Model Accuracy (CV): {cv_scores.mean() * 100:.2f}%")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Precision: {precision * 100:.2f}%")
    logging.info(f"Recall: {recall * 100:.2f}%")
    logging.info(f"F1 Score: {f1 * 100:.2f}%")

    return model

def predict_price_movement(model, prices):
    features, _ = generate_features(prices)
    return model.predict(features[-1].reshape(1, -1))[0] if len(features) > 0 else None

def update_data():
    global prices_history, dates_history, buy_signals, sell_signals, model
    retry_delay = 60
    while True:
        dates, prices = fetch_bitcoin_data()
        if not dates or not prices:
            logging.warning("No data available. Retrying...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 3600)
            continue

        prices_history = (prices_history + prices)[-200:]
        dates_history = (dates_history + dates)[-200:]

        if len(prices_history) > 100 and (model is None or len(prices_history) % 50 == 0):
            model = train_model(prices_history)

        if model:
            prediction = predict_price_movement(model, prices_history)
            if len(prices_history) > 14:
                if prediction == 1:
                    buy_signals.append(len(prices_history) - 1)
                else:
                    sell_signals.append(len(prices_history) - 1)

        save_data_to_json()  # Save data after every update
        time.sleep(60)

threading.Thread(target=update_data, daemon=True).start()

# Add the missing Interval component in layout
app.layout = html.Div(
    style={"backgroundColor": "#111", "height": "100vh", "display": "flex", "justifyContent": "center", "alignItems": "center"},
    children=[
        dcc.Graph(
            id='live-graph',
            style={"width": "100vw", "height": "100vh"},
            config={"displayModeBar": False}
        ),
        dcc.Interval(id='interval-component', interval=60 * 1000, n_intervals=0)  # This triggers the callback every minute
    ]
)

@app.callback(Output('live-graph', 'figure'), [Input('interval-component', 'n_intervals')])
def update_graph(_):
    if not dates_history or not prices_history:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_history, y=prices_history, mode='lines', name='Bitcoin Price', line=dict(color='blue')))

    if buy_signals:
        fig.add_trace(go.Scatter(
            x=[dates_history[i] for i in buy_signals], y=[prices_history[i] for i in buy_signals],
            mode='markers', name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))

    if sell_signals:
        fig.add_trace(go.Scatter(
            x=[dates_history[i] for i in sell_signals], y=[prices_history[i] for i in sell_signals],
            mode='markers', name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))

    fig.add_annotation(
        x=dates_history[-1], y=max(prices_history),
        text="",
        showarrow=False,
        font=dict(size=14, color="white"),
        align="left",
        bordercolor="white",
        borderwidth=2,
        bgcolor="black",
        opacity=0.8
    )

    fig.update_layout(
        title='Bitcoin Buy/Sell Signals                                 Made by Cr0mb                       github.com/cr0mb',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
