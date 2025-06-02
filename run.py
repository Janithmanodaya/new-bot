from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import threading
import time
import websocket
import json
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Thread lock for market_data
market_data_lock = threading.Lock()

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
current_tick_subscription_id = None # Store the ID of the active tick subscription

# Helper: Calculate indicators

def calculate_indicators(prices_list: list[float]):
    """
    Calculates various technical indicators based on a list of prices.
    """
    if not prices_list:
        logging.warning("[Indicators] Price list is empty. Cannot calculate indicators.")
        # Return empty lists of the correct structure if prices_list is empty
        return {
            'rsi': [], 'macd': [], 'bollinger_upper': [],
            'bollinger_middle': [], 'bollinger_lower': [], 'stochastic': []
        }

    prices_series = pd.Series(prices_list, dtype=float)
    logging.debug(f"[Indicators] Calculating indicators for a series of {len(prices_series)} prices.")
    logging.debug(f"[Indicators] Prices head: {prices_series.head().tolist()}, Prices tail: {prices_series.tail().tolist()}")

    # RSI
    delta = prices_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss
    # Handle division by zero for RS: if loss is 0, RS is effectively infinite (or NaN if gain is also 0)
    # An RS of infinity means RSI approaches 100. If gain is also 0 (no change), RSI is undefined, 50 is a neutral default.
    rs[loss == 0] = np.inf
    rs[(gain == 0) & (loss == 0)] = np.nan # Or some other suitable value for RSI to be 50 later

    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50) # Fill initial NaNs and specific (gain=0, loss=0) NaNs with 50
    logging.debug(f"[Indicators] RSI calculated. Head: {rsi.head().tolist()}, Tail: {rsi.tail().tolist()}")

    # MACD
    # Using adjust=False for EWMA is common in financial calculations.
    # This calculates the MACD line. Signal line and histogram are not part of this.
    logging.debug("[Indicators] MACD: Calculating MACD line (not signal line or histogram).")
    macd_line = prices_series.ewm(span=12, adjust=False).mean() - prices_series.ewm(span=26, adjust=False).mean()
    macd_line = macd_line.fillna(0) # Fill initial NaNs with 0
    logging.debug(f"[Indicators] MACD line calculated. Head: {macd_line.head().tolist()}, Tail: {macd_line.tail().tolist()}")

    # Bollinger Bands
    # Using ddof=0 for population standard deviation, common in some platforms.
    bollinger_middle = prices_series.rolling(window=20).mean()
    bollinger_std = prices_series.rolling(window=20).std(ddof=0)
    bollinger_upper = bollinger_middle + (2 * bollinger_std)
    bollinger_lower = bollinger_middle - (2 * bollinger_std)

    # fillna(prices_series) for initial values makes the bands follow the price line.
    # This might not be visually ideal for early data points but maintains current behavior.
    # Alternatives could be bfill/ffill or np.nan.
    logging.debug("[Indicators] Bollinger Bands: Using fillna(prices_series) for initial band values.")
    bollinger_upper = bollinger_upper.fillna(prices_series)
    bollinger_middle = bollinger_middle.fillna(prices_series)
    bollinger_lower = bollinger_lower.fillna(prices_series)
    logging.debug(f"[Indicators] Bollinger Bands calculated. Middle Head: {bollinger_middle.head().tolist()}, Upper Tail: {bollinger_upper.tail().tolist()}")

    # Stochastic Oscillator (%K)
    low_14 = prices_series.rolling(window=14).min()
    high_14 = prices_series.rolling(window=14).max()
    # Adding a small epsilon to prevent division by zero if high_14 == low_14
    stochastic_k = 100 * (prices_series - low_14) / (high_14 - low_14 + 1e-9)
    stochastic_k = stochastic_k.fillna(50) # Fill initial NaNs with 50 (neutral)
    logging.debug(f"[Indicators] Stochastic Oscillator (%K) calculated. Head: {stochastic_k.head().tolist()}, Tail: {stochastic_k.tail().tolist()}")

    # Ensure all returned lists are of the same length as the input prices_series
    # Pandas rolling/ewm operations with default settings should preserve index and length, filling leading values with NaN.
    return {
        'rsi': rsi.tolist(),
        'macd': macd_line.tolist(),
        'bollinger_upper': bollinger_upper.tolist(),
        'bollinger_middle': bollinger_middle.tolist(),
        'bollinger_lower': bollinger_lower.tolist(),
        'stochastic': stochastic_k.tolist()
    }

# WebSocket thread to fetch real data from Deriv

ws_thread = None
ws_app = None

# Enhanced WebSocket Handlers
def on_open_for_deriv(ws):
    """Called when the WebSocket connection is established."""
    logging.info(f"WebSocket connection opened. Attempting to subscribe to ticks for SYMBOL: {SYMBOL} or authorize.")
    if API_TOKEN:
        logging.info(f"[Deriv WS] Connection opened. API_TOKEN is present. Attempting to authorize.")
        ws.send(json.dumps({"authorize": API_TOKEN}))
    else:
        logging.info(f"[Deriv WS] Connection opened. No API_TOKEN provided. Subscribing to public ticks for {SYMBOL}.")
        ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

def on_message_for_deriv(ws, message):
    """Called when a message is received from the WebSocket."""
    global market_data, SYMBOL, current_tick_subscription_id # Added SYMBOL and subscription ID
    data = json.loads(message)
    msg_type = data.get('msg_type')

    # Basic logging for all messages, can be made more verbose if needed
    # logging.info(f"[Deriv WS] Received message: {json.dumps(data, indent=2)}")

    if msg_type == 'authorize':
        auth_response = data.get('authorize', {})
        if data.get('error'):
            logging.error(f"[Deriv WS] Authorization failed: {data['error']['message']}")
            # Do not subscribe to ticks if authorization fails
        else:
            loginid = auth_response.get('loginid')
            logging.info(f"[Deriv WS] Authorization successful for user: {loginid}")
            # Consider updating SYMBOL if provided in auth_response, though not typical
            # current_symbol_in_auth = auth_response.get('symbol') # Fictional field for example
            # if current_symbol_in_auth and current_symbol_in_auth != SYMBOL:
            #     logging.info(f"[Deriv WS] Symbol updated from {SYMBOL} to {current_symbol_in_auth} based on authorization response.")
            #     SYMBOL = current_symbol_in_auth
            logging.info(f"[Deriv WS] Subscribing to ticks for {SYMBOL} after successful authorization.")
            ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

    elif msg_type == 'subscribe': # Handle subscription confirmation
        logging.info(f"Full 'subscribe' response: {message}")
        if data.get('error'):
            logging.error(f"Error in 'subscribe' response: {data['error']}")
        else:
            subscription = data.get('subscription')
            if subscription and subscription.get('id'):
                current_tick_subscription_id = subscription['id']
                # This specific logging for ID is still useful alongside full message
                logging.info(f"[Deriv WS] Successfully subscribed with ID: {current_tick_subscription_id}")
            else:
                logging.warning(f"[Deriv WS] Received subscription confirmation, but no ID found in data: {data}")

    elif msg_type == 'forget':
        logging.info(f"Full 'forget' response: {message}")

    elif msg_type == 'tick':
        tick = data.get('tick')
        if not tick: # Check if 'tick' object itself is missing
            logging.error(f"[Deriv WS] Received message with msg_type 'tick' but no 'tick' object: {data}")
            return

        epoch = tick.get('epoch')
        price = tick.get('quote')
        tick_symbol = tick.get('symbol') # Get symbol from tick data

        if epoch is None:
            logging.error(f"[Deriv WS] Tick data missing 'epoch': {tick}")
            return
        if price is None:
            logging.error(f"[Deriv WS] Tick data missing 'quote': {tick}")
            return
        if tick_symbol is None: # Also good to check for symbol in tick
            logging.warning(f"[Deriv WS] Tick data missing 'symbol': {tick}") # Warning as we might still process if SYMBOL matches

        # Ensure timestamp conversion is robust
        try:
            ts = datetime.fromtimestamp(epoch, timezone.utc)
        except Exception as e:
            logging.error(f"[Deriv WS] Error converting epoch '{epoch}' to datetime: {e}")
            return

        logging.info(f"[Deriv WS] Processing tick: Symbol={tick_symbol}, Price={price}, Timestamp={ts.isoformat()}")
        
        # Optionally, ensure the tick received is for the subscribed SYMBOL, if critical
        # if tick_symbol != SYMBOL:
        #     logging.warning(f"[Deriv WS] Received tick for symbol {tick_symbol}, but subscribed to {SYMBOL}. Ignoring.")
        #     return

        with market_data_lock:
            try:
                market_data['timestamps'].append(ts.isoformat())
                market_data['prices'].append(price) # price is already a float/numeric
                for k in ['timestamps', 'prices']:
                    if len(market_data[k]) > 100:
                        market_data[k] = market_data[k][-100:]

                current_prices = market_data['prices']
                logging.debug(f"[Deriv WS] Passing {len(current_prices)} prices to calculate_indicators.")
                try:
                    indicators = calculate_indicators(current_prices)
                    for key_indicator in indicators:
                        # Ensure calculated indicators list is also sliced if shorter than 100 for some reason
                        # (though calculate_indicators should return full length)
                        market_data[key_indicator] = indicators[key_indicator][-100:]
                except Exception as e_calc:
                    logging.error(f"[Deriv WS] Error calling calculate_indicators: {e_calc}", exc_info=True)
                    # Decide how to handle: skip update for indicators, or clear them, or use last known good
                    # For now, indicators might become stale or mismatched if error occurs.

                if 'volumes' not in market_data or len(market_data['volumes']) < len(market_data['prices']):
                    market_data['volumes'].append(np.random.randint(1000, 5000))
                if len(market_data['volumes']) > 100:
                    market_data['volumes'] = market_data['volumes'][-100:]
                logging.info("[Deriv WS] Market data updated successfully.")
            except Exception as e:
                logging.error(f"[Deriv WS] Error processing tick and updating market_data: {e}")
            finally:
                pass # Lock is released automatically by with statement context exit
            
    elif msg_type == 'proposal_open_contract':
        if data.get('error'):
            logging.error(f"[Deriv WS] Proposal error: {data['error']['message']}")
        else:
            logging.info(f"[Deriv WS] Proposal successful: {data.get('proposal_open_contract')}")
            
    elif data.get('error'): # This is a general error catcher for messages that have an 'error' field
        logging.error(f"Deriv API Error received: {data['error']}. Full message: {message}")
        
    # else:
        # logging.info(f"[Deriv WS] Received unhandled message type: {msg_type}. Full message: {message}")

def on_error_for_deriv(ws, error):
    """Called when a WebSocket error occurs."""
    logging.error(f"[Deriv WS] Error: {error}")

def on_close_for_deriv(ws, close_status_code, close_msg):
    """Called when the WebSocket connection is closed."""
    logging.warning(f"WebSocket connection closed. Status: {close_status_code}, Message: {close_msg}. Current SYMBOL: {SYMBOL}")

def start_deriv_ws(token=None):
    global ws_thread, ws_app, API_TOKEN, market_data
    if token:
        API_TOKEN = token
    
    if ws_app and ws_app.sock and ws_app.sock.connected:
        logging.info("[Deriv WS] WebSocket is already running. Closing existing connection to restart.")
        try:
            ws_app.keep_running = False
            ws_app.close() # Ensure close is called
            if ws_thread and ws_thread.is_alive():
                 ws_thread.join(timeout=5) # Increased timeout slightly
        except Exception as e:
            logging.error(f"[Deriv WS] Error closing existing WebSocket: {e}")

    logging.info(f"[Deriv WS] Starting WebSocket connection to {DERIV_WS_URL} for symbol {SYMBOL}.")
    
    # Clear market data on restart
    with market_data_lock:
        market_data = {
            'timestamps': [], 'prices': [], 'volumes': [], 'rsi': [], 'macd': [],
            'bollinger_upper': [], 'bollinger_middle': [], 'bollinger_lower': [], 'stochastic': []
        }

    global current_tick_subscription_id
    current_tick_subscription_id = None # Reset subscription ID for new connection

    def run_ws():
        global ws_app
        ws_app = websocket.WebSocketApp(
            DERIV_WS_URL,
            on_open=on_open_for_deriv,
            on_message=on_message_for_deriv,
            on_error=on_error_for_deriv, # Use new handler
            on_close=on_close_for_deriv  # Use new handler
        )
        # Set keep_running to True before starting
        ws_app.keep_running = True
        ws_app.run_forever(ping_interval=20, ping_timeout=10) # Added ping for keep-alive
        logging.info("[Deriv WS] WebSocket run_forever loop has exited.")

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    logging.info("[Deriv WS] WebSocket thread started.")

@app.route('/api/market_data')
def get_market_data():
    with market_data_lock:
        try:
            # Return a copy to avoid issues if data is modified while serializing
            data_copy = market_data.copy()
            return jsonify(data_copy)
        finally:
            pass # Lock is released automatically

auth_result_queue = queue.Queue()

def check_deriv_token(token):
    """Check if the Deriv API token is valid by authorizing via WebSocket."""
    logging.info(f"[Token Check] Starting token validation for token: {'********' if token else 'No token'}")
    result = {'success': False, 'error': None}
    done = threading.Event()

    def on_open_check(ws_check):
        logging.debug("[Token Check] WebSocket opened for token check.")
        ws_check.send(json.dumps({"authorize": token}))

    def on_message_check(ws_check, message_text):
        data = json.loads(message_text)
        logging.debug(f"[Token Check] Received message: {data}")
        if data.get('msg_type') == 'authorize':
            if not data.get('error'):
                result['success'] = True
                logging.info("[Token Check] Token validation successful.")
            else:
                result['error'] = data['error']['message']
                logging.error(f"Token validation error: {data['error']}") # Enhanced logging
            ws_check.close()
            done.set()
        elif data.get('error'): # General error not specific to authorize msg_type
            result['error'] = data['error']['message']
            logging.error(f"[Token Check] Received error during token check: {result['error']}")
            ws_check.close()
            done.set()

    def on_error_check(ws_check, error_text):
        logging.error(f"[Token Check] WebSocket error during token check: {error_text}")
        result['error'] = str(error_text) # Ensure error is captured
        # ws_check.close() # ws_check might be None or in a bad state.
        done.set() # Ensure we don't hang forever

    def on_close_check(ws_check, status, msg):
        logging.debug(f"[Token Check] WebSocket closed for token check. Status: {status}, Msg: {msg}")
        if not done.is_set(): # If not already set by on_message or on_error
            logging.warning("[Token Check] WebSocket closed before authorization confirmation.")
            if not result['error'] and not result['success']: # Avoid overwriting specific error
                 result['error'] = "Connection closed before authorization response."
            done.set()


    ws_check_app = websocket.WebSocketApp(
        DERIV_WS_URL,
        on_open=on_open_check,
        on_message=on_message_check,
        on_error=on_error_check,
        on_close=on_close_check
    )

    thread = threading.Thread(target=ws_check_app.run_forever, daemon=True)
    thread.start()

    if not done.wait(timeout=10): # Wait for up to 10 seconds
        logging.error("[Token Check] Token validation timed out.")
        result['error'] = "Token validation timed out."
        try:
            ws_check_app.close() # Attempt to close timed-out socket
        except Exception as e:
            logging.error(f"[Token Check] Error closing timed-out WebSocket: {e}")

    logging.info(f"[Token Check] Finished token validation. Success: {result['success']}")
    return result

@app.route('/api/connect', methods=['POST'])
def connect_deriv_api():
    data = request.get_json()
    token = data.get('token', '')
    logging.info(f"[/api/connect] Connection attempt with token: {'********' if token else 'No token'}")
    # Check token validity first
    check = check_deriv_token(token)
    if check['success']:
        logging.info(f"[/api/connect] Token validated successfully. Starting Deriv WebSocket for token.")
        start_deriv_ws(token)
        return jsonify({'status': 'started', 'token': token})
    else:
        error_msg = check['error'] or 'Invalid token'
        logging.error(f"[/api/connect] Token validation failed: {error_msg}")
        return jsonify({'status': 'failed', 'error': error_msg}), 401

@app.route('/api/connect', methods=['GET'])
def connect_get():
    return jsonify({'status': 'ok', 'message': 'Use POST to connect with your token.'})

@app.route('/api/set_symbol', methods=['POST'])
def set_symbol():
    global SYMBOL, market_data, ws_app, API_TOKEN, current_tick_subscription_id
    data = request.get_json()
    new_symbol = data.get('symbol')

    if not new_symbol or not isinstance(new_symbol, str):
        logging.error(f"[/api/set_symbol] Invalid symbol provided: {new_symbol}")
        return jsonify({'status': 'error', 'message': 'Invalid symbol format'}), 400

    logging.info(f"[/api/set_symbol] Received request to change symbol from {SYMBOL} to {new_symbol}")
    SYMBOL = new_symbol

    # Reset market_data
    with market_data_lock:
        logging.info(f"[/api/set_symbol] Clearing market data for new symbol {SYMBOL}.")
        for key in market_data:
            market_data[key].clear()
        # Re-initialize if necessary, or ensure calculate_indicators handles empty lists
        market_data['timestamps'] = []
        market_data['prices'] = []
        market_data['volumes'] = []
        # Indicators will be repopulated by calculate_indicators
        logging.info(f"[/api/set_symbol] Market data cleared.")

    # Handle WebSocket Re-subscription
    if ws_app and ws_app.sock and ws_app.sock.connected:
        # No explicit "forget" needed here. start_deriv_ws will close the old connection,
        # which implicitly ends old subscriptions. The new connection will subscribe to the new SYMBOL.
        logging.info(f"[/api/set_symbol] WebSocket is connected. Restarting connection for new symbol: {SYMBOL}.")
        logging.info(f"Calling start_deriv_ws to restart WebSocket for new symbol: {SYMBOL}")
        start_deriv_ws(API_TOKEN) # Pass current token to maintain auth if present
    else:
        logging.info(f"[/api/set_symbol] WebSocket not connected. New symbol {SYMBOL} will be used on next connection.")

    return jsonify({'status': 'symbol_updated', 'new_symbol': SYMBOL})

@app.route('/')
def home():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

# Start WebSocket thread on server start (default, no token)
start_deriv_ws()

if __name__ == '__main__':
    app.run(debug=True)
