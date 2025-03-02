import time
import threading
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
import json
from datetime import datetime, timezone, timedelta
import pytz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = Dash(__name__, suppress_callback_exceptions=True)

data_file = "buy_sell_data.json"
prices_history, dates_history = [], []
buy_signals, sell_signals = [], []
model = None
central_tz = pytz.timezone('US/Central')

def fetch_bitcoin_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1'
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            prices = [entry[1] for entry in data['prices']]
            timestamps = [entry[0] for entry in data['prices']]
            
            dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc).astimezone(central_tz) for ts in timestamps]
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
    
    load_data_from_json()
    
    while True:
        dates, prices = fetch_bitcoin_data()
        if not dates or not prices:
            time.sleep(120)
            continue

        prices_history += prices
        dates_history += dates

        if len(prices_history) > 50 and (model is None or len(prices_history) % 10 == 0):
            model = train_model(prices_history)

        if model:
            prediction = predict_price_movement(model, prices_history)

            if prediction == 1:
                buy_signals.append(dates_history[-1])
            elif prediction == 0:
                sell_signals.append(dates_history[-1])

        save_data_to_json()
        time.sleep(120)

threading.Thread(target=update_data, daemon=True).start()

app.layout = html.Div([
    dcc.Graph(id='live-graph', style={"width": "70vw", "height": "100vh"}),
    html.Div([
        dash_table.DataTable(
            id='buy-sell-table',
            columns=[
                {"name": "Time", "id": "time"},
                {"name": "Price (USD)", "id": "price"},
                {"name": "Signal", "id": "signal"}
            ],
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_history, y=prices_history, mode='lines', name='Bitcoin Price', line=dict(color='blue')))

    price_lookup = {date: price for date, price in zip(dates_history, prices_history)}

    buy_points = [(date, price_lookup[date]) for date in buy_signals if date in price_lookup]
    sell_points = [(date, price_lookup[date]) for date in sell_signals if date in price_lookup]


    if buy_points:
        fig.add_trace(go.Scatter(
            x=[bp[0] for bp in buy_points], 
            y=[bp[1] for bp in buy_points], 
            mode='markers',
            name='Buy', 
            marker=dict(color='green', size=10),
            line=dict(width=0)
        ))

    if sell_points:
        fig.add_trace(go.Scatter(
            x=[sp[0] for sp in sell_points], 
            y=[sp[1] for sp in sell_points], 
            mode='markers',  
            name='Sell', 
            marker=dict(color='red', size=10),
            line=dict(width=0)
        ))


    table_data = [
        {"time": date.strftime('%Y-%m-%d %I:%M:%S %p'), "price": price, "signal": "Buy"} 
        for date, price in buy_points
    ] + [
        {"time": date.strftime('%Y-%m-%d %I:%M:%S %p'), "price": price, "signal": "Sell"} 
        for date, price in sell_points
    ]

    return fig, table_data


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
