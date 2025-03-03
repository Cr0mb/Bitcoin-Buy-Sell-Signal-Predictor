import time
import threading
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
import json
from datetime import datetime, timezone
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = Dash(__name__, suppress_callback_exceptions=True)

data_file = "buy_sell_data.json"
metrics_file = "model_metrics.json"
prices_history, dates_history, buy_signals, sell_signals = [], [], [], []
model = None
central_tz = pytz.timezone('US/Central')
BITCOIN_API_URL = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1'
FETCH_INTERVAL = 120


def fetch_bitcoin_data():
    for attempt in range(3):
        try:
            response = requests.get(BITCOIN_API_URL, timeout=5)
            response.raise_for_status()
            data = response.json()
            return [
                datetime.fromtimestamp(entry[0] / 1000, tz=timezone.utc).astimezone(central_tz) 
                for entry in data['prices']
            ], [entry[1] for entry in data['prices']]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    return [], []


def save_data_to_json():
    try:
        with open(data_file, 'w') as f:
            json.dump({
                "prices_history": prices_history,
                "dates_history": [date.isoformat() for date in dates_history],
                "buy_signals": [date.isoformat() for date in buy_signals],
                "sell_signals": [date.isoformat() for date in sell_signals]
            }, f)
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
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning(f"{data_file} not found or invalid. Starting fresh.")


def train_model(prices):
    if len(prices) < 50:
        return None
    labels = np.sign(np.diff(prices)).astype(int)
    labels[labels == -1] = 0
    X_train, X_test, y_train, y_test = train_test_split(np.array(prices[:-1]).reshape(-1, 1), labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logging.info(f"Model trained. Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"Confusion Matrix:\n{cm}")
    
    return model


def save_metrics_to_json(accuracy):
    try:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = []

        metrics.append({"timestamp": datetime.now().isoformat(), "accuracy": accuracy})

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Model accuracy saved to {metrics_file}.")
    except Exception as e:
        logging.error(f"Error saving model metrics: {e}")


def predict_price_movement(model, prices):
    return model.predict(np.array(prices[-2]).reshape(1, -1))[0] if len(prices) >= 2 else None


def update_data():
    global prices_history, dates_history, buy_signals, sell_signals, model
    load_data_from_json()
    while True:
        dates, prices = fetch_bitcoin_data()
        if dates and prices:
            prices_history.extend(prices)
            dates_history.extend(dates)

            if len(prices_history) > 50 and (model is None or len(prices_history) % 10 == 0):
                model = train_model(prices_history)

            if model:
                prediction = predict_price_movement(model, prices_history)
                if prediction == 1:
                    buy_signals.append(dates_history[-1])
                elif prediction == 0:
                    sell_signals.append(dates_history[-1])
            save_data_to_json()
        time.sleep(FETCH_INTERVAL)

threading.Thread(target=update_data, daemon=True).start()

app.layout = html.Div([
    dcc.Graph(id='live-graph', style={"width": "70vw", "height": "100vh"}),
    html.Div([
        dash_table.DataTable(
            id='buy-sell-table',
            columns=[{"name": "Time", "id": "time"}, {"name": "Price (USD)", "id": "price"}, {"name": "Signal", "id": "signal"}],
            data=[],
            style_table={'height': '100%', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center'}
        )
    ], style={"width": "30vw", "height": "100vh", "overflow": "auto", "display": "inline-block"}),
    dcc.Interval(id='interval-component', interval=60000, n_intervals=0)
])

@app.callback(
    [Output('live-graph', 'figure'), Output('buy-sell-table', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph_and_table(_):
    if not dates_history or not prices_history:
        return go.Figure(), []

    price_lookup = dict(zip(dates_history, prices_history))
    buy_points = [(date, price_lookup[date]) for date in buy_signals if date in price_lookup]
    sell_points = [(date, price_lookup[date]) for date in sell_signals if date in price_lookup]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates_history, y=prices_history, mode='lines', name='Bitcoin Price', line=dict(color='blue')))

    if buy_points:
        fig.add_trace(go.Scatter(x=[bp[0] for bp in buy_points], y=[bp[1] for bp in buy_points], 
                                 mode='markers', name='Buy', marker=dict(color='green', size=10)))

    if sell_points:
        fig.add_trace(go.Scatter(x=[sp[0] for sp in sell_points], y=[sp[1] for sp in sell_points], 
                                 mode='markers', name='Sell', marker=dict(color='red', size=10)))

    table_data = [{"time": date.strftime('%Y-%m-%d %I:%M:%S %p'), "price": price, "signal": "Buy"} for date, price in buy_points] + \
                 [{"time": date.strftime('%Y-%m-%d %I:%M:%S %p'), "price": price, "signal": "Sell"} for date, price in sell_points]

    return fig, table_data


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
