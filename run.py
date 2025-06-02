from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import threading
import time
import websocket
import json
import queue

app = Flask(__name__)
CORS(app)

# Global storage for real-time data
market_data = {
    'timestamps': [],
    'prices': [],
    'volumes': [],
    'rsi': [],
    'macd': [],
    'bollinger_upper': [],
    'bollinger_middle': [],
    'bollinger_lower': [],
    'stochastic': []
}

# Deriv API config
DERIV_APP_ID = 1089
DERIV_WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"
SYMBOL = 'frxUSDJPY'  # Default symbol, can be parameterized
API_TOKEN = ''  # Set your token here or via environment

# Helper: Calculate indicators

def calculate_indicators(prices):
    prices = pd.Series(prices)
    rsi = prices.rolling(14).apply(lambda x: (100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0).sum() or 1))))))
    macd = prices.ewm(span=12).mean() - prices.ewm(span=26).mean()
    bollinger_middle = prices.rolling(20).mean()
    bollinger_std = prices.rolling(20).std()
    bollinger_upper = bollinger_middle + 2 * bollinger_std
    bollinger_lower = bollinger_middle - 2 * bollinger_std
    stochastic = 100 * (prices - prices.rolling(14).min()) / (prices.rolling(14).max() - prices.rolling(14).min() + 1e-9)
    return {
        'rsi': rsi.fillna(50).tolist(),
        'macd': macd.fillna(0).tolist(),
        'bollinger_upper': bollinger_upper.fillna(prices).tolist(),
        'bollinger_middle': bollinger_middle.fillna(prices).tolist(),
        'bollinger_lower': bollinger_lower.fillna(prices).tolist(),
        'stochastic': stochastic.fillna(50).tolist()
    }

# WebSocket thread to fetch real data from Deriv

ws_thread = None
ws_app = None

# Enhanced WebSocket Handlers
def on_open_for_deriv(ws):
    """Called when the WebSocket connection is established."""
    if API_TOKEN:
        print(f"[Deriv WS] Connection opened. Attempting to authorize with token.")
        ws.send(json.dumps({"authorize": API_TOKEN}))
    else:
        print(f"[Deriv WS] Connection opened. No API token provided. Subscribing to public ticks for {SYMBOL}.")
        # For public ticks, we usually don't need to authorize first.
        ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

def on_message_for_deriv(ws, message):
    """Called when a message is received from the WebSocket."""
    global market_data
    data = json.loads(message)
    msg_type = data.get('msg_type')

    print(f"[Deriv WS] Received message: {json.dumps(data, indent=2)}") # Log entire message

    if msg_type == 'authorize':
        if data.get('error'):
            print(f"[Deriv WS] Authorization failed: {data['error']['message']}")
            # Optionally, you could close ws here or set an error state
            # ws.close()
        else:
            print(f"[Deriv WS] Authorization successful for user: {data.get('authorize', {}).get('loginid')}")
            # Now that we are authorized, subscribe to ticks
            print(f"[Deriv WS] Subscribing to ticks for {SYMBOL}.")
            ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
    elif msg_type == 'tick':
        tick = data.get('tick')
        if not tick:
            print(f"[Deriv WS] Received tick message without tick data: {data}")
            return
            
        epoch = tick.get('epoch')
        price = tick.get('quote')

        if epoch is None or price is None:
            print(f"[Deriv WS] Tick data missing epoch or quote: {tick}")
            return

        ts = datetime.fromtimestamp(epoch, timezone.utc) # Use timezone.utc
        print(f"[Deriv WS] Tick received: Symbol={tick.get('symbol')}, Price={price}, Time={ts}")
        
        # Append new data
        market_data['timestamps'].append(ts.isoformat())
        market_data['prices'].append(price)
        # Keep only last 100 points
        for k in ['timestamps', 'prices']:
            if len(market_data[k]) > 100:
                market_data[k] = market_data[k][-100:]
        # Calculate indicators
        indicators = calculate_indicators(market_data['prices'])
        for key_indicator in indicators: # Renamed 'key' to avoid conflict if any
            market_data[key_indicator] = indicators[key_indicator][-100:]
        
        # Simulate volume if not present in tick data (Deriv ticks usually don't have volume)
        if 'volumes' not in market_data or len(market_data['volumes']) < len(market_data['prices']):
            market_data['volumes'].append(np.random.randint(1000, 5000)) # Adjusted volume range
        if len(market_data['volumes']) > 100:
            market_data['volumes'] = market_data['volumes'][-100:]
            
    elif msg_type == 'proposal_open_contract': # Example: if you were trading
        if data.get('error'):
            print(f"[Deriv WS] Proposal error: {data['error']['message']}")
        else:
            print(f"[Deriv WS] Proposal successful: {data.get('proposal_open_contract')}")
            
    elif data.get('error'):
        print(f"[Deriv WS] Received error: {data['error']['message']} (Code: {data['error'].get('code')})")
        # ws.close() # Consider closing on critical errors
        
    # else:
        # print(f"[Deriv WS] Received unhandled message type: {msg_type}")

def start_deriv_ws(token=None):
    global ws_thread, ws_app, API_TOKEN
    if token:
        API_TOKEN = token # Update global token if a new one is provided
    
    if ws_app and ws_app.sock and ws_app.sock.connected:
        print("[Deriv WS] WebSocket is already running. Closing existing connection to restart.")
        try:
            ws_app.keep_running = False # Signal to stop
            ws_app.close()
            if ws_thread and ws_thread.is_alive():
                 ws_thread.join(timeout=2) # Wait for thread to finish
        except Exception as e:
            print(f"[Deriv WS] Error closing existing WebSocket: {e}")
    
    print(f"[Deriv WS] Starting WebSocket connection to {DERIV_WS_URL} for symbol {SYMBOL}.")
    
    def run_ws():
        global ws_app
        ws_app = websocket.WebSocketApp(
            DERIV_WS_URL,
            on_open=on_open_for_deriv,
            on_message=on_message_for_deriv,
            on_error=lambda ws, err: print(f"[Deriv WS] Error: {err}"),
            on_close=lambda ws, status_code, msg: print(f"[Deriv WS] Connection closed. Status: {status_code}, Msg: {msg}")
        )
        # Set keep_running to True before starting
        ws_app.keep_running = True
        ws_app.run_forever(ping_interval=20, ping_timeout=10) # Added ping for keep-alive
        print("[Deriv WS] WebSocket run_forever loop has exited.")

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    print("[Deriv WS] WebSocket thread started.")

@app.route('/api/market_data')
def get_market_data():
    return jsonify(market_data)

auth_result_queue = queue.Queue()

def check_deriv_token(token):
    """Check if the Deriv API token is valid by authorizing via WebSocket."""
    result = {'success': False, 'error': None}
    done = threading.Event()
    def on_open(ws):
        ws.send(json.dumps({"authorize": token}))
    def on_message(ws, message):
        data = json.loads(message)
        if data.get('msg_type') == 'authorize':
            result['success'] = True
            ws.close()
            done.set()
        elif data.get('error'):
            result['error'] = data['error']['message']
            ws.close()
            done.set()
    ws = websocket.WebSocketApp(
        DERIV_WS_URL,
        on_open=on_open,
        on_message=on_message
    )
    thread = threading.Thread(target=ws.run_forever)
    thread.start()
    done.wait(timeout=10)  # Wait for up to 10 seconds
    return result

@app.route('/api/connect', methods=['POST'])
def connect_deriv_api():
    data = request.get_json()
    token = data.get('token', '')
    # Check token validity first
    check = check_deriv_token(token)
    if check['success']:
        start_deriv_ws(token)
        return jsonify({'status': 'started', 'token': token})
    else:
        return jsonify({'status': 'failed', 'error': check['error'] or 'Invalid token'}), 401

@app.route('/api/connect', methods=['GET'])
def connect_get():
    return jsonify({'status': 'ok', 'message': 'Use POST to connect with your token.'})

@app.route('/')
def home():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

# Start WebSocket thread on server start (default, no token)
start_deriv_ws()

if __name__ == '__main__':
    app.run(debug=True)
