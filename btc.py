import time
import threading
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#111", "height": "100vh", "display": "flex", "justifyContent": "center", "alignItems": "center"},
    children=[
        dcc.Graph(
            id='live-graph',
            style={"width": "100vw", "height": "100vh"},
            config={"displayModeBar": False}
        ),
        dcc.Interval(id='interval-component', interval=60 * 1000, n_intervals=0)
    ]
)

prices_history, dates_history = [], []
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
            logging.info(f"Data fetched successfully (Attempt {attempt + 1})")
            return dates, prices
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            time.sleep(2 ** attempt)
    logging.error("Max retries exceeded. Could not fetch data.")
    return [], []

def calculate_rsi(prices, window=14):
    df = pd.DataFrame({'price': prices})
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.dropna().values

def calculate_sma(prices, window=14):
    return pd.Series(prices).rolling(window).mean().dropna().values

def generate_features(prices):
    rsi_values = calculate_rsi(prices)
    sma_values = calculate_sma(prices)
    min_length = min(len(rsi_values), len(sma_values), len(prices) - 15)
    features = np.column_stack((rsi_values[:min_length], sma_values[:min_length]))
    labels = (np.array(prices[15:15 + min_length]) > np.array(prices[15 - 1:15 - 1 + min_length])).astype(int)
    return features, labels

def train_model(prices):
    features, labels = generate_features(prices)
    if len(features) < 20:
        logging.warning("Not enough data to train model.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    return model

def predict_price_movement(model, prices):
    features, _ = generate_features(prices)
    return model.predict(features[-1].reshape(1, -1))[0] if len(features) > 0 else None

def update_data():
    global prices_history, dates_history, model
    while True:
        dates, prices = fetch_bitcoin_data()
        if not dates or not prices:
            logging.warning("No data available. Retrying...")
            time.sleep(60)
            continue

        prices_history = (prices_history + prices)[-200:]
        dates_history = (dates_history + dates)[-200:]

        if len(prices_history) > 100 and (model is None or len(prices_history) % 50 == 0):
            model = train_model(prices_history)

        time.sleep(60)

threading.Thread(target=update_data, daemon=True).start()

@app.callback(Output('live-graph', 'figure'), [Input('interval-component', 'n_intervals')])
def update_graph(_):
    if not dates_history or not prices_history:
        return go.Figure()

    buy_signals, sell_signals = [], []
    if model:
        prediction = predict_price_movement(model, prices_history)
        if len(prices_history) > 14:
            if prediction == 1:
                buy_signals.append(len(prices_history) - 1)
            else:
                sell_signals.append(len(prices_history) - 1)

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
        text="Legend:\n🟢 Green = Buy\n🔴 Red = Sell",
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
