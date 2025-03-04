    import dash
    from dash import dcc, html
    import plotly.graph_objs as go
    import pandas as pd
    import requests
    import dash_bootstrap_components as dbc
    from dash.dash_table import DataTable
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import numpy as np

    def fetch_bitcoin_data():
        url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
        params = {'vs_currency': 'usd', 'days': '1'}
        response = requests.get(url, params=params)
        data = response.json()
        prices = data['prices']
        return pd.DataFrame(prices, columns=['timestamp', 'price'])

    def format_timestamps(data):
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    signal_data = pd.DataFrame(columns=['timestamp', 'price', 'signal'])

    app.layout = html.Div([
        html.H1("Bitcoin Buy/Sell Signal Predictor", style={'textAlign': 'center', 'marginTop': '20px', 'color': 'white'}),
        
        dcc.Dropdown(
            id='day-filter',
            options=[
                {'label': 'Last 24 hours', 'value': '24h'},
                {'label': 'Today', 'value': 'today'}
            ],
            value='24h',
            style={'width': '50%', 'margin': 'auto', 'marginTop': '20px', 'color': 'black', 'backgroundColor': '#f8f9fa'}
        ),
        
        html.Div([
            html.Div([
                dcc.Graph(
                    id='price-graph',
                    style={'height': '500px', 'width': '100%'},
                    config={'displayModeBar': True, 'scrollZoom': True, 'responsive': True}
                )
            ], style={'width': '70%', 'display': 'inline-block'}),

            html.Div([
                DataTable(id='signals-table', style_table={'height': '400px', 'overflowY': 'auto'},
                          style_cell={'textAlign': 'center', 'padding': '5px', 'color': 'white', 'backgroundColor': '#343a40'}),
            ], style={'width': '30%', 'display': 'inline-block', 'paddingLeft': '20px'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '20px'}),

        html.Div(id='model-status', style={'textAlign': 'center', 'marginTop': '20px', 'color': 'white'}),
        html.Div(id='metrics-display', style={'textAlign': 'center', 'marginTop': '20px', 'color': 'white'}),
    ])

    @app.callback(
        [
            dash.dependencies.Output('price-graph', 'figure'),
            dash.dependencies.Output('model-status', 'children'),
            dash.dependencies.Output('signals-table', 'data'),
            dash.dependencies.Output('metrics-display', 'children')
        ],
        [
            dash.dependencies.Input('day-filter', 'value')
        ]
    )
    def update_graph(day_filter):
        data = fetch_bitcoin_data()

        data = format_timestamps(data)

        print("Fetching new data...")

        buy_signal = data['price'] > data['price'].shift(1)
        sell_signal = data['price'] < data['price'].shift(1)

        data['signal'] = np.nan
        data.loc[buy_signal, 'signal'] = 'Buy'
        data.loc[sell_signal, 'signal'] = 'Sell'

        actual_signals = data['signal'].shift(-1).fillna('Sell')

        accuracy = accuracy_score(actual_signals[1:], data['signal'][1:])
        f1 = f1_score(actual_signals[1:], data['signal'][1:], pos_label='Buy')
        conf_matrix = confusion_matrix(actual_signals[1:], data['signal'][1:], labels=['Buy', 'Sell'])

        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        trace_price = go.Scatter(
            x=data['timestamp'],
            y=data['price'],
            mode='lines',
            name='Bitcoin Price',
            line=dict(color='royalblue', width=2)
        )

        trace_buy = go.Scatter(
            x=data[buy_signal]['timestamp'],
            y=data[buy_signal]['price'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='lime', size=12),
            name='Buy Signal'
        )

        trace_sell = go.Scatter(
            x=data[sell_signal]['timestamp'],
            y=data[sell_signal]['price'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12),
            name='Sell Signal'
        )

        figure = {
            'data': [trace_price, trace_buy, trace_sell],
            'layout': go.Layout(
                title='Bitcoin Price with Buy/Sell Signals',
                xaxis=dict(title='Time', showgrid=False),
                yaxis=dict(title='Price (USD)', showgrid=True, zeroline=False),
                showlegend=True,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(family='Arial', size=14, color='white')
            )
        }

        table_data = data[['timestamp', 'price', 'signal']].to_dict('records')

        metrics_display = (
            f"Accuracy: {accuracy:.2f}<br>"
            f"F1-Score: {f1:.2f}<br>"
            f"Confusion Matrix: <br>"
            f"[[{conf_matrix[0][0]}, {conf_matrix[0][1]}],<br>"
            f"[{conf_matrix[1][0]}, {conf_matrix[1][1]}]]"
        )

        return figure, "Model is ready!", table_data, metrics_display


    if __name__ == '__main__':
        app.run_server(debug=True)
