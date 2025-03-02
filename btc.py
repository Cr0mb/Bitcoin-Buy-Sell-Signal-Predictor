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

data_file = "buy_sell_data.json"
prices_history, dates_history = [], []
buy_signals, sell_signals = [], []
model = None

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
            return dates, prices
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            time.sleep(2 ** attempt)
    return [], []

def save_data_to_json():
    data = {
        "prices_history": prices_history,
        "dates_history": [str(date) for date in dates_history],
        "buy_signals": [str(date) for date in buy_signals],
        "sell_signals": [str(date) for date in sell_signals]
    }
    try:
        with open(data_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"Error saving data: {e}")

def load_data_from_json():
    global prices_history, dates_history, buy_signals, sell_signals
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            prices_history = data.get("prices_history", [])
            dates_history = [datetime.fromisoformat(date) for date in data.get("dates_history", [])]
            buy_signals = [datetime.fromisoformat(date) for date in data.get("buy_signals", [])]
            sell_signals = [datetime.fromisoformat(date) for date in data.get("sell_signals", [])]
    except FileNotFoundError:
        logging.warning(f"{data_file} not found. Starting fresh.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")

def train_model(prices):
    if len(prices) < 50:
        return None
    labels = np.sign(np.diff(prices)).astype(int)
    labels[labels == -1] = 0
    features = np.array(prices[:-1]).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model

def predict_price_movement(model, prices):
    if len(prices) < 2:
        return None
    return model.predict(np.array(prices[-2]).reshape(1, -1))[0]

def update_data():
    global prices_history, dates_history, buy_signals, sell_signals, model
    while True:
        dates, prices = fetch_bitcoin_data()
        if not dates or not prices:
            time.sleep(60)
            continue

        prices_history = (prices_history + prices)[-200:]
        dates_history = (dates_history + dates)[-200:]

        if len(prices_history) > 50 and (model is None or len(prices_history) % 10 == 0):
            model = train_model(prices_history)

        if model:
            prediction = predict_price_movement(model, prices_history)
            if prediction == 1:
                buy_signals.append(dates_history[-1])
            elif prediction == 0:
                sell_signals.append(dates_history[-1])

        save_data_to_json()
        time.sleep(60)

threading.Thread(target=update_data, daemon=True).start()

app.layout = html.Div([
    dcc.Graph(id='live-graph', style={"width": "100vw", "height": "100vh"}),
    dcc.Interval(id='interval-component', interval=60000, n_intervals=0)
])

@app.callback(Output('live-graph', 'figure'), [Input('interval-component', 'n_intervals')])
def update_graph(_):
    if not dates_history or not prices_history:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_history, y=prices_history, mode='lines', name='Bitcoin Price', line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(
        x=[date for date in buy_signals if date in dates_history],
        y=[prices_history[dates_history.index(date)] for date in buy_signals if date in dates_history],
        mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=[date for date in sell_signals if date in dates_history],
        y=[prices_history[dates_history.index(date)] for date in sell_signals if date in dates_history],
        mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)
    ))

    fig.update_layout(title='Bitcoin Buy/Sell Signals', xaxis_title='Time', yaxis_title='Price (USD)', template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
