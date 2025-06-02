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
import copy # For deepcopy

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

app = Flask(__name__)
CORS(app)

# Thread lock for market_data
market_data_lock = threading.Lock()

current_chart_type = 'tick'  # Default: 'tick' or 'ohlcv'
current_granularity_seconds = 60 # Default: e.g., 60 for 1-minute candles
TIMEFRAME_TO_SECONDS = {
    '1M': 60, '5M': 300, '15M': 900, '1H': 3600, '4H': 14400, '1D': 86400, '1W': 604800
}

DEFAULT_MARKET_DATA = {
    'timestamps': [], 'prices': [], 'volumes': [], # For tick chart and general use
    'ohlcv_candles': [], # For OHLCV chart: list of {'time': ms, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
    'rsi': [], 'macd': [], 'bollinger_upper': [], 'bollinger_middle': [],
    'bollinger_lower': [], 'stochastic': [], 'cci_20': [],
    'sma_10': [], 'sma_20': [], 'sma_30': [], 'sma_50': [], 'sma_100': [], 'sma_200': [],
    'ema_10': [], 'ema_20': [], 'ema_30': [], 'ema_50': [], 'ema_100': [], 'ema_200': [],
    # Signals
    'rsi_signal': 'N/A', 'stochastic_signal': 'N/A', 'macd_signal': 'N/A',
    'cci_20_signal': 'N/A', 'price_ema_50_signal': 'N/A',
    'market_sentiment_text': 'N/A',
    # Daily data
    'high_24h': 'N/A', 'low_24h': 'N/A', 'volume_24h': 'N/A',
    # Prediction states
    'rsi_prediction_state': 'N/A',
    'stochastic_prediction_state': 'N/A',
    'macd_prediction_state': 'N/A',
    'cci_20_prediction_state': 'N/A',
}

# Global storage for real-time data
market_data = copy.deepcopy(DEFAULT_MARKET_DATA)

# Deriv API config
DERIV_APP_ID = 1089
DERIV_WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

DERIV_SYMBOL_MAPPING = {
    "USD/JPY": "frxUSDJPY",
    "EUR/USD": "frxEURUSD",
    "GBP/USD": "frxGBPUSD",
    "AUD/USD": "frxAUDUSD",
    "USD/CAD": "frxUSDCAD",
    "BTC/USD": "cryBTCUSD",
    "ETH/USD": "cryETHUSD"
}

SYMBOL = "frxUSDJPY"
API_TOKEN = ''
current_tick_subscription_id = None

# Helper: Calculate indicators
def calculate_indicators(prices_list: list[float]):
    logging.debug(f"calculate_indicators received prices_list length: {len(prices_list)}, last 5: {prices_list[-5:] if len(prices_list) >= 5 else prices_list}")
    if not prices_list:
        logging.warning("[Indicators] Price list is empty. Cannot calculate indicators.")
        empty_results = {}
        for k, v_default in DEFAULT_MARKET_DATA.items():
            if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']:
                if isinstance(v_default, list): empty_results[k] = []
                elif isinstance(v_default, str): empty_results[k] = 'N/A'
        return empty_results

    prices_series = pd.Series(prices_list, dtype=float)
    results = {}

    results['rsi'] = prices_series.rolling(14).apply(lambda x: (100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0).sum() or 1))))), raw=True).fillna(50).tolist()
    logging.debug(f"calculate_indicators computed RSI, length: {len(results['rsi'])}, last 5: {results['rsi'][-5:] if len(results['rsi']) >= 5 else results['rsi']}")

    macd_line_series = prices_series.ewm(span=12, adjust=False).mean() - prices_series.ewm(span=26, adjust=False).mean()
    results['macd'] = macd_line_series.fillna(0).tolist()
    logging.debug(f"calculate_indicators computed MACD, length: {len(results['macd'])}, last 5: {results['macd'][-5:] if len(results['macd']) >= 5 else results['macd']}")

    bollinger_middle = prices_series.rolling(window=20).mean()
    bollinger_std = prices_series.rolling(window=20).std(ddof=0)
    results['bollinger_upper'] = (bollinger_middle + (2 * bollinger_std)).fillna(prices_series).tolist()
    results['bollinger_middle'] = bollinger_middle.fillna(prices_series).tolist()
    results['bollinger_lower'] = (bollinger_middle - (2 * bollinger_std)).fillna(prices_series).tolist()

    low_14 = prices_series.rolling(14).min()
    high_14 = prices_series.rolling(14).max()
    stochastic_k = 100 * (prices_series - low_14) / (high_14 - low_14 + 1e-9)
    results['stochastic'] = stochastic_k.fillna(50).tolist()

    for N in [10, 20, 30, 50, 100, 200]:
        results[f'sma_{N}'] = prices_series.rolling(window=N).mean().fillna(0).tolist()
    for N in [10, 20, 30, 50, 100, 200]:
        results[f'ema_{N}'] = prices_series.ewm(span=N, adjust=False).mean().fillna(0).tolist()

    cci_period = 20
    tp = prices_series
    sma_tp = tp.rolling(window=cci_period).mean()
    mean_dev = (tp - sma_tp).abs().rolling(window=cci_period).mean()
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    results['cci_20'] = cci.replace([np.inf, -np.inf], 0).fillna(0).tolist()

    if not prices_series.empty and len(prices_list) > 0:
        latest_price = prices_series.iloc[-1]
        latest_rsi = results['rsi'][-1] if results['rsi'] else 50
        results['rsi_signal'] = 'Buy' if latest_rsi < 30 else ('Sell' if latest_rsi > 70 else 'Neutral')
        results['rsi_prediction_state'] = 'Oversold' if latest_rsi < 30 else ('Overbought' if latest_rsi > 70 else 'Neutral')
        latest_stochastic = results['stochastic'][-1] if results['stochastic'] else 50
        results['stochastic_signal'] = 'Buy' if latest_stochastic < 20 else ('Sell' if latest_stochastic > 80 else 'Neutral')
        results['stochastic_prediction_state'] = 'Oversold' if latest_stochastic < 20 else ('Overbought' if latest_stochastic > 80 else 'Neutral')
        latest_macd = results['macd'][-1] if results['macd'] else 0
        results['macd_signal'] = 'Buy' if latest_macd > 0 else ('Sell' if latest_macd < 0 else 'Neutral')
        results['macd_prediction_state'] = 'Bullish' if latest_macd > 0 else ('Bearish' if latest_macd < 0 else 'Neutral')
        latest_cci = results['cci_20'][-1] if results['cci_20'] else 0
        results['cci_signal'] = 'Buy' if latest_cci < -100 else ('Sell' if latest_cci > 100 else 'Neutral')
        results['cci_20_prediction_state'] = 'Oversold' if latest_cci < -100 else ('Overbought' if latest_cci > 100 else 'Neutral')
        latest_ema_50 = results['ema_50'][-1] if (results.get('ema_50') and len(results['ema_50']) > 0) else latest_price
        results['price_ema_50_signal'] = 'Buy' if latest_price > latest_ema_50 else ('Sell' if latest_price < latest_ema_50 else 'Neutral')
    else:
        for sig_key in ['rsi_signal', 'stochastic_signal', 'macd_signal', 'cci_signal', 'price_ema_50_signal',
                        'rsi_prediction_state', 'stochastic_prediction_state', 'macd_prediction_state', 'cci_20_prediction_state',
                        'market_sentiment_text']:
            results[sig_key] = 'N/A'

    signal_keys = ['rsi_signal', 'stochastic_signal', 'macd_signal', 'cci_20_signal', 'price_ema_50_signal']
    active_signals = [results.get(key) for key in signal_keys if results.get(key) and results.get(key) not in ['N/A', 'Neutral']]
    buy_count = active_signals.count('Buy')
    sell_count = active_signals.count('Sell')
    results['market_sentiment_text'] = 'Bullish' if buy_count > sell_count else ('Bearish' if sell_count > buy_count else 'Neutral')
    logging.debug(f"Prediction States: RSI: {results.get('rsi_prediction_state')}, Stochastic: {results.get('stochastic_prediction_state')}, MACD: {results.get('macd_prediction_state')}, CCI: {results.get('cci_20_prediction_state')}")
    return results

ws_thread = None
ws_app = None

def request_ohlcv_data(ws_app_instance, symbol_to_fetch, granularity_s, candle_count=100):
    if ws_app_instance and ws_app_instance.sock and ws_app_instance.sock.connected:
        req_id_ohlcv = f"ohlcv_{symbol_to_fetch}_{granularity_s}_{int(time.time())}"
        request_payload = {
            "ticks_history": symbol_to_fetch,
            "style": "candles",
            "granularity": int(granularity_s),
            "end": "latest",
            "count": int(candle_count),
            "adjust_start_time": 1, # Ensure candles align to typical market open times
            "req_id": req_id_ohlcv
        }
        logging.debug(f"Sending ticks_history request for OHLCV (req_id: {req_id_ohlcv}): {json.dumps(request_payload)}")
        ws_app_instance.send(json.dumps(request_payload))
        return req_id_ohlcv
    else:
        logging.warning(f"WebSocket not connected. Cannot request OHLCV data for {symbol_to_fetch}.")
        return None

def request_daily_data(ws_app_instance, symbol_to_fetch):
    if ws_app_instance and ws_app_instance.sock and ws_app_instance.sock.connected:
        req_id_daily = f"daily_{symbol_to_fetch}_{int(time.time())}"
        logging.info(f"Requesting daily summary data (req_id: {req_id_daily}) for {symbol_to_fetch}...")
        ws_app_instance.send(json.dumps({
            "ticks_history": symbol_to_fetch,
            "style": "candles",
            "granularity": 86400,
            "end": "latest",
            "count": 1,
            "req_id": req_id_daily
        }))
    else:
        logging.warning(f"WebSocket not connected. Cannot request daily data for {symbol_to_fetch}.")

def on_open_for_deriv(ws):
    logging.info(f"WebSocket connection opened. SYMBOL: {SYMBOL}, Granularity: {current_granularity_seconds}s")
    request_daily_data(ws, SYMBOL)
    request_ohlcv_data(ws, SYMBOL, current_granularity_seconds)
    if API_TOKEN:
        ws.send(json.dumps({"authorize": API_TOKEN}))
    else:
        ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

def on_message_for_deriv(ws, message):
    global market_data, SYMBOL, current_tick_subscription_id, current_chart_type, current_granularity_seconds
    data = json.loads(message)
    msg_type = data.get('msg_type')
    echo_req = data.get('echo_req', {})
    req_id_received = echo_req.get('req_id')

    if msg_type == 'authorize':
        if data.get('error'):
            logging.error(f"Authorization failed: {data['error']['message']}")
        else:
            logging.info(f"Authorization successful for user: {data.get('authorize', {}).get('loginid')}")
            ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
            request_daily_data(ws, SYMBOL)
            request_ohlcv_data(ws, SYMBOL, current_granularity_seconds)

    elif msg_type == 'subscribe':
        logging.info(f"Full 'subscribe' response: {message[:200]}")
        if data.get('error'):
            logging.error(f"Error in 'subscribe' response: {data['error']}")
        else:
            sub = data.get('subscription')
            if sub and sub.get('id'): current_tick_subscription_id = sub['id']
            logging.info(f"Successfully subscribed to ticks. ID: {current_tick_subscription_id}")

    elif msg_type == 'forget':
        logging.info(f"Full 'forget' response: {message[:200]}")

    elif msg_type == 'tick':
        tick = data.get('tick')
        if not tick or tick.get('epoch') is None or tick.get('quote') is None:
            logging.error(f"Invalid tick data: {data}")
            return

        ts = datetime.fromtimestamp(tick['epoch'], timezone.utc)
        # logging.info(f"Processing tick: Symbol={tick.get('symbol')}, Price={tick['quote']}, Time={ts.isoformat()}") # Too verbose for DEBUG

        with market_data_lock:
            if current_chart_type == 'tick':
                market_data['timestamps'].append(ts.isoformat())
                market_data['prices'].append(tick['quote'])
                for k_list in ['timestamps', 'prices']:
                    if len(market_data[k_list]) > 100: market_data[k_list] = market_data[k_list][-100:]

            prices_for_indicators = []
            if current_chart_type == 'ohlcv':
                if market_data.get('ohlcv_candles') and len(market_data['ohlcv_candles']) > 0:
                    prices_for_indicators = [c['close'] for c in market_data['ohlcv_candles']]
                    logging.debug(f"Using OHLCV close prices for indicators. Count: {len(prices_for_indicators)}. Last 5: {prices_for_indicators[-5:] if prices_for_indicators else []}")
                else:
                    current_tick_prices = market_data['prices']
                    prices_for_indicators = current_tick_prices[-(min(50, len(current_tick_prices))):]
                    logging.debug(f"OHLCV chart active but no ohlcv_candles. Falling back to recent {len(prices_for_indicators)} ticks for indicators.")
            else:
                current_tick_prices = market_data['prices']
                prices_for_indicators = current_tick_prices[-(min(50, len(current_tick_prices))):]
                logging.debug(f"Tick chart active. Using recent {len(prices_for_indicators)} ticks for indicators. Last 5: {prices_for_indicators[-5:] if prices_for_indicators else []}")

            updated_data = {}
            if not prices_for_indicators:
                logging.warning("No price data available for indicator calculation. Skipping.")
            else:
                updated_data = calculate_indicators(prices_for_indicators)

            for key, value in updated_data.items():
                if key in market_data:
                    if isinstance(value, list):
                        market_data[key] = value
                    elif isinstance(value, str):
                        market_data[key] = value

            if 'volumes' in market_data:
                 market_data['volumes'].append(np.random.randint(1000,5000))
                 if len(market_data['volumes']) > 100: market_data['volumes'] = market_data['volumes'][-100:]
            # logging.info("[Deriv WS] Market data (including indicators) updated successfully based on new tick.") # Can be too verbose

    elif msg_type == 'candles':
        # echo_req and req_id_received are already available from the top of the function
        logging.debug(f"Full raw 'candles' response for req_id {req_id_received}: {message}")
        raw_candles_list = data.get('candles', [])
        logging.debug(f"Received {len(raw_candles_list)} raw candle objects for req_id {req_id_received}.")
        
        if req_id_received and req_id_received.startswith("ohlcv_"):
            parsed_ohlcv_candles = []
            if raw_candles_list: # Check if raw_candles_list is not empty
                for candle in raw_candles_list:
                    if isinstance(candle, dict):
                        candle_time_ms = int(candle.get('epoch', 0)) * 1000
                        parsed_ohlcv_candles.append({
                            'time': candle_time_ms,
                            'open': float(candle.get('open', 0)),
                            'high': float(candle.get('high', 0)),
                            'low': float(candle.get('low', 0)),
                            'close': float(candle.get('close', 0)),
                            'volume': float(candle.get('volume', candle.get('vol', 0)))
                        })
                parsed_ohlcv_candles.sort(key=lambda x: x['time'])
                logging.debug(f"Parsed {len(parsed_ohlcv_candles)} candles for OHLCV req_id {req_id_received}. First 1-2 parsed: {parsed_ohlcv_candles[:2] if parsed_ohlcv_candles else []}")
                with market_data_lock:
                    market_data['ohlcv_candles'] = parsed_ohlcv_candles
                logging.info(f"Updated market_data['ohlcv_candles'] with {len(parsed_ohlcv_candles)} candles for main chart (req_id: {req_id_received}).")
                if current_chart_type == 'ohlcv':
                    logging.info("OHLCV data received, recalculating indicators using these candles.")
                    prices_for_recalc = [c['close'] for c in parsed_ohlcv_candles]
                    if prices_for_recalc:
                        updated_data_from_ohlcv = calculate_indicators(prices_for_recalc)
                        with market_data_lock:
                            for key, value in updated_data_from_ohlcv.items():
                                if key in market_data:
                                    if isinstance(value, list): market_data[key] = value
                                    elif isinstance(value, str): market_data[key] = value
                        logging.info("Indicators recalculated and updated using new OHLCV data.")
            else:
                logging.warning(f"No candle data in 'ohlcv_' response for req_id: {req_id_received}")

        elif req_id_received and req_id_received.startswith("daily_"):
            if raw_candles_list: # Check if raw_candles_list is not empty
                candle = raw_candles_list[0]
                if isinstance(candle, dict):
                    with market_data_lock:
                        market_data['high_24h'] = float(candle.get('high')) if candle.get('high') is not None else 'N/A'
                        market_data['low_24h'] = float(candle.get('low')) if candle.get('low') is not None else 'N/A'
                        market_data['volume_24h'] = float(candle.get('volume', candle.get('vol'))) if candle.get('volume', candle.get('vol')) is not None else 'N/A'
                    logging.info(f"Updated 24h data: High={market_data['high_24h']}, Low={market_data['low_24h']}, Volume={market_data['volume_24h']}")
                else: logging.warning("Daily candle data item not in dict format.")
            else: logging.warning(f"No candle data in 'daily_' response for req_id: {req_id_received}")
        else:
            logging.warning(f"Received 'candles' message with unrecognized req_id: {req_id_received}. Full message: {message[:300]}")

    elif data.get('error'):
        logging.error(f"Deriv API Error received: {data['error']}. Full message: {message}")

def on_error_for_deriv(ws, error):
    logging.error(f"[Deriv WS] Error: {error}")

def on_close_for_deriv(ws, close_status_code, close_msg):
    logging.warning(f"WebSocket connection closed. Status: {close_status_code}, Message: {close_msg}. Current SYMBOL: {SYMBOL}")

def start_deriv_ws(token=None):
    global ws_thread, ws_app, API_TOKEN, market_data, current_tick_subscription_id
    if token: API_TOKEN = token
    
    if ws_app and ws_app.sock and ws_app.sock.connected:
        logging.info("Existing WebSocket connection found. Closing it before restarting.")
        try:
            ws_app.keep_running = False
            ws_app.close()
            if ws_thread and ws_thread.is_alive(): ws_thread.join(timeout=5)
        except Exception as e: logging.error(f"Error closing existing WebSocket: {e}")
    
    logging.info(f"Starting WebSocket connection for SYMBOL: {SYMBOL}.")
    with market_data_lock:
        market_data.clear()
        market_data.update(copy.deepcopy(DEFAULT_MARKET_DATA))
        logging.info(f"Market data cleared and re-initialized for {SYMBOL}.")

    current_tick_subscription_id = None

    def run_ws():
        global ws_app
        ws_app = websocket.WebSocketApp(
            DERIV_WS_URL,
            on_open=on_open_for_deriv,
            on_message=on_message_for_deriv,
            on_error=on_error_for_deriv,
            on_close=on_close_for_deriv
        )
        ws_app.keep_running = True
        ws_app.run_forever(ping_interval=20, ping_timeout=10)
        logging.info("WebSocket run_forever loop has exited.")

    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    logging.info("WebSocket thread started.")

@app.route('/api/market_data')
def get_market_data_endpoint():
    global current_chart_type, current_granularity_seconds, TIMEFRAME_TO_SECONDS

    with market_data_lock:
        data_to_send = copy.deepcopy(market_data)

    data_to_send['current_chart_type'] = current_chart_type
    data_to_send['current_granularity_seconds'] = current_granularity_seconds

    current_timeframe_str = 'N/A'
    for tf_str, tf_secs in TIMEFRAME_TO_SECONDS.items():
        if tf_secs == current_granularity_seconds:
            current_timeframe_str = tf_str
            break
    data_to_send['current_timeframe_string'] = current_timeframe_str

    logging.debug(f"Sending market_data with chart_type: {current_chart_type}, granularity: {current_granularity_seconds}s, timeframe_str: {current_timeframe_str}")
    return jsonify(data_to_send)

@app.route('/api/set_chart_settings', methods=['POST'])
def set_chart_settings_endpoint():
    global current_chart_type, current_granularity_seconds, ws_app, SYMBOL, market_data_lock, market_data

    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    new_chart_type = data.get('chart_type', current_chart_type)
    new_timeframe_str = data.get('timeframe_str', None)
    made_change = False

    if new_chart_type != current_chart_type:
        if new_chart_type in ['tick', 'ohlcv']:
            current_chart_type = new_chart_type
            made_change = True
            logging.info(f"Chart type changed to: {current_chart_type}")
        else:
            logging.warning(f"Invalid chart_type received: {new_chart_type}")

    if new_timeframe_str:
        granularity = TIMEFRAME_TO_SECONDS.get(new_timeframe_str.upper())
        if granularity:
            if granularity != current_granularity_seconds:
                current_granularity_seconds = granularity
                made_change = True
                logging.info(f"Chart granularity changed to: {current_granularity_seconds}s (from {new_timeframe_str})")
        else:
            logging.warning(f"Invalid timeframe_str received: {new_timeframe_str}")

    if made_change:
        logging.info(f"Chart settings changed. ChartType: {current_chart_type}, Granularity: {current_granularity_seconds}s. Requesting relevant data if needed.")
        if ws_app and ws_app.sock and ws_app.sock.connected:
            if current_chart_type == 'ohlcv':
                logging.info(f"Attempting to fetch/refresh OHLCV data due to settings change: Symbol={SYMBOL}, Granularity={current_granularity_seconds}s, ChartType={current_chart_type}")
                with market_data_lock:
                    if 'ohlcv_candles' in market_data: market_data['ohlcv_candles'].clear()
                    if 'prices' in market_data: market_data['prices'].clear()
                    if 'timestamps' in market_data: market_data['timestamps'].clear()
                request_ohlcv_data(ws_app, SYMBOL, current_granularity_seconds)
            # If chart type changed to 'tick', existing live ticks will populate 'prices' and 'timestamps'.
            # Indicators will be recalculated on the next tick based on the new 'current_chart_type'.
        else:
            logging.warning("WebSocket not connected. Cannot immediately fetch new data on settings change.")

    return jsonify({
        'status': 'success',
        'current_chart_type': current_chart_type,
        'current_granularity_seconds': current_granularity_seconds
    })

@app.route('/api/connect', methods=['POST'])
def connect_deriv_api():
    data = request.get_json()
    token = data.get('token', '')
    logging.info(f"[/api/connect] Connection attempt with token: {'********' if token else 'No token'}")
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
    global SYMBOL, market_data, ws_app, API_TOKEN, current_tick_subscription_id, current_granularity_seconds
    data = request.get_json()
    new_symbol_value = data.get('symbol')

    if not new_symbol_value or not isinstance(new_symbol_value, str):
        logging.error(f"[/api/set_symbol] Invalid symbol value provided: {new_symbol_value}")
        return jsonify({'status': 'error', 'message': 'Invalid symbol format'}), 400

    actual_deriv_symbol = DERIV_SYMBOL_MAPPING.get(new_symbol_value, new_symbol_value)
    logging.info(f"Symbol received from frontend: '{new_symbol_value}', Mapped to Deriv symbol: '{actual_deriv_symbol}'")

    if actual_deriv_symbol == new_symbol_value and not any(new_symbol_value.startswith(p) for p in ['frx', 'cry', 'R_']):
        logging.warning(f"Symbol '{new_symbol_value}' was not found in DERIV_SYMBOL_MAPPING and might not be a valid Deriv API symbol format.")

    logging.info(f"[/api/set_symbol] Attempting to change global SYMBOL from '{SYMBOL}' to '{actual_deriv_symbol.strip()}'")
    SYMBOL = actual_deriv_symbol.strip()

    if not (ws_app and ws_app.sock and ws_app.sock.connected):
        with market_data_lock:
            logging.info(f"[/api/set_symbol] WebSocket not connected. Clearing market data for new symbol {SYMBOL} locally.")
            market_data.clear()
            market_data.update(copy.deepcopy(DEFAULT_MARKET_DATA))
            logging.info(f"[/api/set_symbol] Market data cleared and re-initialized locally.")

    if ws_app and ws_app.sock and ws_app.sock.connected:
        logging.info(f"[/api/set_symbol] WebSocket is connected. Restarting connection for new symbol: {SYMBOL}.")
        start_deriv_ws(API_TOKEN)
    else:
        logging.info(f"[/api/set_symbol] WebSocket not connected. New symbol {SYMBOL} will be used on next connection.")

    return jsonify({'status': 'symbol_updated', 'new_symbol': SYMBOL})

@app.route('/')
def home():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

def check_deriv_token(token):
    logging.info(f"[Token Check] Starting token validation for token: {'********' if token else 'No token'}")
    result = {'success': False, 'error': None}
    done_event = threading.Event()

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
                logging.error(f"Token validation error: {data['error']}")
            ws_check.close()
            done_event.set()
        elif data.get('error'):
            result['error'] = data['error']['message']
            logging.error(f"[Token Check] Received error during token check: {result['error']}")
            ws_check.close()
            done_event.set()

    def on_error_check(ws_check, error_text):
        logging.error(f"[Token Check] WebSocket error during token check: {error_text}")
        result['error'] = str(error_text)
        done_event.set()

    def on_close_check(ws_check, status, msg):
        logging.debug(f"[Token Check] WebSocket closed for token check. Status: {status}, Msg: {msg}")
        if not done_event.is_set():
            logging.warning("[Token Check] WebSocket closed before authorization confirmation.")
            if not result['error'] and not result['success']:
                 result['error'] = "Connection closed before authorization response."
            done_event.set()

    ws_check_app = websocket.WebSocketApp(
        DERIV_WS_URL,
        on_open=on_open_check,
        on_message=on_message_check,
        on_error=on_error_check,
        on_close=on_close_check
    )

    thread = threading.Thread(target=ws_check_app.run_forever, daemon=True)
    thread.start()

    if not done_event.wait(timeout=10):
        logging.error("[Token Check] Token validation timed out.")
        result['error'] = "Token validation timed out."
        try:
            ws_check_app.close()
        except Exception as e:
            logging.error(f"[Token Check] Error closing timed-out WebSocket: {e}")

    logging.info(f"[Token Check] Finished token validation. Success: {result['success']}")
    return result

start_deriv_ws()

if __name__ == '__main__':
    app.run(debug=True)
