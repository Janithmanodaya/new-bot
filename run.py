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

# Global variables for request tracking
next_req_id = 1
pending_api_requests = {} # Stores req_id: {'type': request_type_string, 'event': threading.Event(), 'data': None, 'error': None}

# Thread lock for market_data and pending_api_requests
shared_data_lock = threading.Lock()
market_data_lock = threading.Lock() # Retaining for specific market_data operations if needed, but shared_data_lock can cover pending_api_requests
api_response_events = {} # req_id: threading.Event()
api_response_data = {}   # req_id: data_from_ws

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
    'atr_14': [], # Average True Range
    # Signals
    'rsi_signal': 'N/A', 'stochastic_signal': 'N/A', 'macd_signal': 'N/A',
    'cci_20_signal': 'N/A', 'price_ema_50_signal': 'N/A',
    'market_sentiment_text': 'N/A',
    # Daily data
    'high_24h': 'N/A', 'low_24h': 'N/A', 'volume_24h': 'N/A',
    # Prediction states & Volatility
    'rsi_prediction_state': 'N/A',
    'stochastic_prediction_state': 'N/A',
    'macd_prediction_state': 'N/A',
    'cci_20_prediction_state': 'N/A',
    'atr_volatility_state': 'N/A', # New Volatility State
}

# Global storage for real-time data
market_data = copy.deepcopy(DEFAULT_MARKET_DATA)

# ATR Calculation specific constants
ATR_PERIOD = 14

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
    "ETH/USD": "cryETHUSD",
    "Gold/USD": "XAUUSD",      # New
    "Volatility 75 Index": "R_75" # New
}

SYMBOL = "frxUSDJPY" 
API_TOKEN = ''  
current_tick_subscription_id = None 

# Helper: Calculate indicators
def calculate_indicators(data_input):
    """
    Calculates various technical indicators.

    Args:
        data_input: Can be one of:
            - list[float]: A list of close prices (typically for tick-based calculations).
            - list[dict]: A list of OHLCV candles, where each dict has
                          {'time', 'open', 'high', 'low', 'close', 'volume'}.
                          Required for ATR calculation.
    Returns:
        dict: A dictionary containing lists of calculated indicator values.
    """

    is_ohlcv_data = False
    prices_list = [] # Close prices
    high_prices = []
    low_prices = []
    close_prices_for_atr = [] # Separate for clarity if needed by ATR logic

    if not data_input:
        logging.warning("[Indicators] Data input is empty. Cannot calculate indicators.")
        # Return default empty structure for all indicators
        empty_results = {}
        for k, v_default in DEFAULT_MARKET_DATA.items():
            if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']:
                if isinstance(v_default, list): empty_results[k] = []
                elif isinstance(v_default, str): empty_results[k] = 'N/A'
        return empty_results

    if isinstance(data_input[0], dict): # OHLCV data
        is_ohlcv_data = True
        try:
            prices_list = [d['close'] for d in data_input]
            high_prices = [d['high'] for d in data_input]
            low_prices = [d['low'] for d in data_input]
            close_prices_for_atr = prices_list # Used for previous close in TR calc
            logging.debug(f"calculate_indicators received OHLCV data. Count: {len(data_input)}")
        except KeyError as e:
            logging.error(f"[Indicators] OHLCV data missing key: {e}. Cannot calculate all indicators.")
            # Fallback to just close prices if possible, or return empty for ATR.
            # For simplicity, if structure is wrong, we might not proceed with ATR.
            is_ohlcv_data = False # Revert, as we don't have full HLC
            if not prices_list: # If 'close' also failed
                 return {k: [] if isinstance(DEFAULT_MARKET_DATA[k], list) else 'N/A' for k in DEFAULT_MARKET_DATA if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']}
    elif isinstance(data_input[0], float): # List of close prices
        prices_list = data_input
        logging.debug(f"calculate_indicators received list of close prices. Count: {len(prices_list)}")
    else:
        logging.error(f"[Indicators] Unknown data_input type: {type(data_input[0])}. Cannot calculate indicators.")
        return {k: [] if isinstance(DEFAULT_MARKET_DATA[k], list) else 'N/A' for k in DEFAULT_MARKET_DATA if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']}

    if not prices_list: # Should have been caught by initial check, but as a safeguard
        logging.warning("[Indicators] Price list (derived) is empty. Cannot calculate indicators.")
        return {k: [] if isinstance(DEFAULT_MARKET_DATA[k], list) else 'N/A' for k in DEFAULT_MARKET_DATA if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']}

    prices_series = pd.Series(prices_list, dtype=float)
    results = {} 

    # Standard Indicators (using prices_series from close prices)
    results['rsi'] = prices_series.rolling(14).apply(
        lambda x: (100 - (100 / (1 + (pd.Series(x).diff().clip(lower=0).sum() / (abs(pd.Series(x).diff().clip(upper=0).sum()) or 1e-9) ) ) ) ), 
        raw=True
    ).fillna(50).tolist()
    logging.debug(f"calculate_indicators computed RSI, length: {len(results['rsi'])}, last 5: {results['rsi'][-5:] if len(results['rsi']) >= 5 else results['rsi']}")
    
    macd_line_series = prices_series.ewm(span=12, adjust=False).mean() - prices_series.ewm(span=26, adjust=False).mean()
    results['macd'] = macd_line_series.fillna(0).tolist()
    logging.debug(f"calculate_indicators computed MACD, length: {len(results['macd'])}, last 5: {results['macd'][-5:] if len(results['macd']) >= 5 else results['macd']}")

    bollinger_middle = prices_series.rolling(window=20).mean()
    bollinger_std = prices_series.rolling(window=20).std(ddof=0) # Population standard deviation
    results['bollinger_upper'] = (bollinger_middle + (2 * bollinger_std)).fillna(prices_series).tolist()
    results['bollinger_middle'] = bollinger_middle.fillna(prices_series).tolist()
    results['bollinger_lower'] = (bollinger_middle - (2 * bollinger_std)).fillna(prices_series).tolist()
    
    low_14_close = prices_series.rolling(14).min() # Based on close prices
    high_14_close = prices_series.rolling(14).max() # Based on close prices
    stochastic_k = 100 * (prices_series - low_14_close) / (high_14_close - low_14_close + 1e-9)
    results['stochastic'] = stochastic_k.fillna(50).tolist()

    for N in [10, 20, 30, 50, 100, 200]:
        results[f'sma_{N}'] = prices_series.rolling(window=N).mean().fillna(prices_series).tolist() # Fill with current price if NaN start
    for N in [10, 20, 30, 50, 100, 200]:
        results[f'ema_{N}'] = prices_series.ewm(span=N, adjust=False).mean().fillna(prices_series).tolist() # Fill with current price

    cci_period = 20
    # Typical Price (TP) = (High + Low + Close) / 3. If not OHLCV, use Close as TP.
    if is_ohlcv_data and len(high_prices) == len(low_prices) == len(prices_list):
        tp_series = pd.Series((np.array(high_prices) + np.array(low_prices) + np.array(prices_list)) / 3, dtype=float)
    else:
        tp_series = prices_series # Use close price if H/L not available for TP

    sma_tp = tp_series.rolling(window=cci_period).mean()
    mean_dev = (tp_series - sma_tp).abs().rolling(window=cci_period).mean()
    cci = (tp_series - sma_tp) / (0.015 * mean_dev + 1e-9) # Added epsilon to denominator
    results['cci_20'] = cci.replace([np.inf, -np.inf], 0).fillna(0).tolist()

    # ATR Calculation
    if is_ohlcv_data and len(high_prices) >= ATR_PERIOD and len(low_prices) >= ATR_PERIOD and len(close_prices_for_atr) >= ATR_PERIOD:
        logging.debug(f"Calculating ATR_{ATR_PERIOD} with OHLCV data. Count: {len(high_prices)}")
        high_s = pd.Series(high_prices, dtype=float)
        low_s = pd.Series(low_prices, dtype=float)
        close_s = pd.Series(close_prices_for_atr, dtype=float)

        tr = pd.DataFrame()
        tr['h_l'] = high_s - low_s
        tr['h_pc'] = abs(high_s - close_s.shift(1))
        tr['l_pc'] = abs(low_s - close_s.shift(1))

        true_range = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        true_range = true_range.fillna(0) # Fill NaN for the first TR value if any (e.g. from shift)

        # Wilder's Smoothing for ATR
        # atr = true_range.ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
        # More explicit Wilder's smoothing:
        atr = pd.Series(np.nan, index=true_range.index)
        atr.iloc[ATR_PERIOD-1] = true_range.iloc[:ATR_PERIOD].mean() # Initial ATR is simple average of first N TRs
        for i in range(ATR_PERIOD, len(true_range)):
            atr.iloc[i] = (atr.iloc[i-1] * (ATR_PERIOD - 1) + true_range.iloc[i]) / ATR_PERIOD

        results['atr_14'] = atr.fillna(0).tolist() # Fill initial NaNs with 0 or a suitable value
        logging.debug(f"ATR_{ATR_PERIOD} calculated. Length: {len(results['atr_14'])}, Last 5: {results['atr_14'][-5:] if results['atr_14'] else 'N/A'}")
    else:
        results['atr_14'] = []
        if not is_ohlcv_data:
            logging.debug("ATR calculation skipped: Not OHLCV data.")
        else:
            logging.warning(f"ATR calculation skipped: Insufficient OHLCV data length. Need {ATR_PERIOD}, got {len(high_prices)} H, {len(low_prices)} L, {len(close_prices_for_atr)} C.")

    results['atr_volatility_state'] = 'N/A' # Default
    if results.get('atr_14') and len(results['atr_14']) >= ATR_PERIOD: # Need enough ATR values to form SMA of ATR
        atr_series = pd.Series(results['atr_14'])
        sma_atr_5 = atr_series.rolling(window=5).mean() # 5-period SMA of ATR
        if not sma_atr_5.empty and not pd.isna(sma_atr_5.iloc[-1]) and not pd.isna(atr_series.iloc[-1]):
            latest_atr = atr_series.iloc[-1]
            latest_sma_atr = sma_atr_5.iloc[-1]

            if latest_atr > latest_sma_atr * 1.5: # Thresholds are examples
                results['atr_volatility_state'] = 'High'
            elif latest_atr < latest_sma_atr * 0.7:
                results['atr_volatility_state'] = 'Low'
            else:
                results['atr_volatility_state'] = 'Normal'
            logging.debug(f"ATR Volatility State: {results['atr_volatility_state']} (ATR: {latest_atr:.4f}, SMA_ATR(5): {latest_sma_atr:.4f})")
        else:
            logging.debug(f"ATR Volatility State: N/A (SMA of ATR not available or latest ATR is NaN. SMA_ATR_5 length: {len(sma_atr_5)}, last value: {sma_atr_5.iloc[-1] if not sma_atr_5.empty else 'empty'})")
            results['atr_volatility_state'] = 'N/A' # Explicitly N/A if SMA_ATR cannot be determined
    else:
        logging.debug("ATR Volatility State: N/A (ATR data insufficient or not calculated).")


    # Signal Generation (remains based on close prices and derived indicators)
    if not prices_series.empty: # prices_list was non-empty
        latest_price = prices_series.iloc[-1]

        # RSI
        latest_rsi = results['rsi'][-1] if results.get('rsi') and results['rsi'] else 50
        results['rsi_signal'] = 'Buy' if latest_rsi < 30 else ('Sell' if latest_rsi > 70 else 'Neutral')
        results['rsi_prediction_state'] = 'Oversold' if latest_rsi < 30 else ('Overbought' if latest_rsi > 70 else 'Neutral')

        # Stochastic
        latest_stochastic = results['stochastic'][-1] if results.get('stochastic') and results['stochastic'] else 50
        results['stochastic_signal'] = 'Buy' if latest_stochastic < 20 else ('Sell' if latest_stochastic > 80 else 'Neutral')
        results['stochastic_prediction_state'] = 'Oversold' if latest_stochastic < 20 else ('Overbought' if latest_stochastic > 80 else 'Neutral')

        # MACD
        latest_macd = results['macd'][-1] if results.get('macd') and results['macd'] else 0
        # Signal could also use MACD line vs Signal line cross, but this is simpler: MACD > 0 is bullish.
        results['macd_signal'] = 'Buy' if latest_macd > 0 else ('Sell' if latest_macd < 0 else 'Neutral')
        results['macd_prediction_state'] = 'Bullish' if latest_macd > 0 else ('Bearish' if latest_macd < 0 else 'Neutral')

        # CCI
        latest_cci = results['cci_20'][-1] if results.get('cci_20') and results['cci_20'] else 0
        results['cci_signal'] = 'Buy' if latest_cci < -100 else ('Sell' if latest_cci > 100 else 'Neutral')
        results['cci_20_prediction_state'] = 'Oversold' if latest_cci < -100 else ('Overbought' if latest_cci > 100 else 'Neutral')

        # Price vs EMA (e.g., EMA50)
        latest_ema_50 = results['ema_50'][-1] if (results.get('ema_50') and results['ema_50']) else latest_price
        results['price_ema_50_signal'] = 'Buy' if latest_price > latest_ema_50 else ('Sell' if latest_price < latest_ema_50 else 'Neutral')

    else: # prices_series is empty
        # Set all signals and states to N/A if no price data
        for sig_key in ['rsi_signal', 'stochastic_signal', 'macd_signal', 'cci_signal', 'price_ema_50_signal', 
                        'rsi_prediction_state', 'stochastic_prediction_state', 'macd_prediction_state',
                        'cci_20_prediction_state', 'market_sentiment_text', 'atr_volatility_state']:
            results[sig_key] = 'N/A'
    
    # Market Sentiment (based on count of Buy/Sell signals from primary indicators)
    signal_keys = ['rsi_signal', 'stochastic_signal', 'macd_signal', 'cci_20_signal', 'price_ema_50_signal']
    active_signals = [results.get(key) for key in signal_keys if results.get(key) and results.get(key) not in ['N/A', 'Neutral']] # Ensure key exists
    buy_count = active_signals.count('Buy')
    sell_count = active_signals.count('Sell')
    if buy_count == 0 and sell_count == 0 and not prices_series.empty: # If all signals are Neutral or N/A but there is price data
        results['market_sentiment_text'] = 'Neutral'
    elif buy_count > sell_count:
        results['market_sentiment_text'] = 'Bullish'
    elif sell_count > buy_count:
        results['market_sentiment_text'] = 'Bearish'
    else: # buy_count == sell_count (and not both zero, or prices_series is empty)
        results['market_sentiment_text'] = 'Neutral' if not prices_series.empty else 'N/A'

    logging.debug(f"Prediction States: RSI: {results.get('rsi_prediction_state')}, Stochastic: {results.get('stochastic_prediction_state')}, MACD: {results.get('macd_prediction_state')}, CCI: {results.get('cci_20_prediction_state')}, ATR Volatility: {results.get('atr_volatility_state')}")
    return results

ws_thread = None
ws_app = None

def request_ohlcv_data(ws_app_instance, symbol_to_fetch, granularity_s, candle_count=100):
    global next_req_id, pending_api_requests
    global next_req_id, pending_api_requests, api_response_events, shared_data_lock
    if ws_app_instance and ws_app_instance.sock and ws_app_instance.sock.connected:
        with shared_data_lock:
            current_req_id = next_req_id
            next_req_id += 1
            # No event needed here as ohlcv_chart_data updates global market_data directly
            pending_api_requests[current_req_id] = {'type': 'ohlcv_chart_data'}
        
        request_payload = {
            "ticks_history": symbol_to_fetch,
            "style": "candles",
            "granularity": int(granularity_s),
            "end": "latest",
            "count": int(candle_count),
            "adjust_start_time": 1, # Ensure candles align to typical market open times
            "req_id": current_req_id # Use integer req_id
        }
        logging.debug(f"Sending ticks_history request for OHLCV (req_id: {current_req_id}, type: ohlcv_chart_data): {json.dumps(request_payload)}")
        ws_app_instance.send(json.dumps(request_payload))
        return current_req_id
    else:
        logging.warning(f"WebSocket not connected. Cannot request OHLCV data for {symbol_to_fetch}.")
        return None

def request_daily_data(ws_app_instance, symbol_to_fetch):
    global next_req_id, pending_api_requests, shared_data_lock
    if ws_app_instance and ws_app_instance.sock and ws_app_instance.sock.connected:
        with shared_data_lock:
            current_req_id = next_req_id
            next_req_id += 1
            # No event needed here as daily_summary_data updates global market_data directly
            pending_api_requests[current_req_id] = {'type': 'daily_summary_data'}
        
        request_payload = {
            "ticks_history": symbol_to_fetch,
            "style": "candles",
            "granularity": 86400, 
            "end": "latest",    
            "count": 1,         
            "req_id": current_req_id # Use integer req_id
        }
        logging.info(f"Requesting daily OHLCV data (req_id: {current_req_id}, type: daily_summary_data) for {symbol_to_fetch}... Payload: {json.dumps(request_payload)}")
        ws_app_instance.send(json.dumps(request_payload))
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
    global market_data, SYMBOL, current_tick_subscription_id, current_chart_type, current_granularity_seconds, pending_api_requests
    global market_data, SYMBOL, current_tick_subscription_id, current_chart_type, current_granularity_seconds
    global pending_api_requests, api_response_events, api_response_data, shared_data_lock
    
    data = json.loads(message)
    logging.debug(f"Raw WS message received: {message[:500]}") # Log more of the message

    req_id_of_message = data.get('echo_req', {}).get('req_id')
    request_details = None

    if req_id_of_message is not None:
        with shared_data_lock:
            request_details = pending_api_requests.get(req_id_of_message)

    if data.get('error'):
        error_details = data['error']
        failed_req_type_str = 'Unknown Request'
        event_to_set = None
        if request_details:
            failed_req_type_str = request_details.get('type', 'Unknown type in details')
            # Store error for requests that expect a response via event
            if 'event' in request_details:
                request_details['error'] = error_details
                event_to_set = request_details['event']

            with shared_data_lock: # Ensure thread-safe removal
                 if req_id_of_message in pending_api_requests:
                    pending_api_requests.pop(req_id_of_message)

        logging.error(f"Deriv API Error for req_id {req_id_of_message} (type: {failed_req_type_str}): {error_details}. Full message: {message}")

        if event_to_set: # Must be outside lock to prevent deadlock if event waiter tries to acquire lock
            event_to_set.set()
        return

    msg_type = data.get('msg_type')

    if msg_type == 'authorize':
        if data.get('error'):
            logging.error(f"Authorization failed: {data['error']['message']}")
            # Potentially signal to connect_deriv_api or check_deriv_token if req_id was used for authorize
            if req_id_of_message:
                 with shared_data_lock:
                    request_details = pending_api_requests.get(req_id_of_message)
                    if request_details and 'event' in request_details:
                        request_details['error'] = data['error']
                        request_details['event'].set()
                    if req_id_of_message in pending_api_requests: # remove it
                        pending_api_requests.pop(req_id_of_message)

        else:
            logging.info(f"Authorization successful for user: {data.get('authorize', {}).get('loginid')}")
            # If this authorize was part of a specific flow (like initial connect or token check), signal it
            if req_id_of_message:
                with shared_data_lock:
                    request_details = pending_api_requests.get(req_id_of_message)
                    if request_details and 'event' in request_details:
                        request_details['data'] = data.get('authorize')
                        request_details['event'].set()
                    if req_id_of_message in pending_api_requests: # remove it
                        pending_api_requests.pop(req_id_of_message)
            else: # General authorization (e.g. on open)
                ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
                request_daily_data(ws, SYMBOL)
                request_ohlcv_data(ws, SYMBOL, current_granularity_seconds)

    elif msg_type == 'subscribe':
        if data.get('error'):
             logging.error(f"Error in 'subscribe' response: {data['error']}")
        else:
            sub = data.get('subscription')
            if sub and sub.get('id'): current_tick_subscription_id = sub['id']
            logging.info(f"Successfully subscribed to ticks. ID: {current_tick_subscription_id}. Symbol: {data.get('echo_req',{}).get('ticks')}")
    
    elif msg_type == 'forget':
        logging.info(f"Full 'forget' response: {message[:200]}. Req ID: {req_id_of_message}")
        # If forget was initiated by a req_id that has an event
        if req_id_of_message:
            with shared_data_lock:
                request_details = pending_api_requests.get(req_id_of_message)
                if request_details and 'event' in request_details:
                    request_details['data'] = data.get('forget') # 1 for success
                    request_details['event'].set()
                if req_id_of_message in pending_api_requests:
                    pending_api_requests.pop(req_id_of_message)


    elif msg_type == 'tick':
        tick = data.get('tick')
        if not tick or tick.get('epoch') is None or tick.get('quote') is None:
            logging.error(f"Invalid tick data: {data}")
            return
        
        ts = datetime.fromtimestamp(tick['epoch'], timezone.utc)
        
        with market_data_lock:
            if current_chart_type == 'tick': 
                market_data['timestamps'].append(ts.isoformat())
                market_data['prices'].append(tick['quote'])
                for k_list in ['timestamps', 'prices']: 
                    if len(market_data[k_list]) > 100: market_data[k_list] = market_data[k_list][-100:]
            
            prices_for_indicators = []
            if current_chart_type == 'ohlcv':
                # When chart is OHLCV, indicators should be based on the OHLCV data itself
                ohlcv_data_for_indicators = market_data.get('ohlcv_candles')
                if ohlcv_data_for_indicators and len(ohlcv_data_for_indicators) > 0:
                    logging.debug(f"Tick received; chart type is OHLCV. Using {len(ohlcv_data_for_indicators)} ohlcv_candles for indicators.")
                    indicator_input_data = ohlcv_data_for_indicators
                else: # Fallback if ohlcv_candles is empty, use recent ticks' close prices
                    current_tick_prices = market_data['prices']
                    # Ensure we have enough data for at least some basic indicators if possible
                    # Using a slice that might be too short for some indicators, but calculate_indicators handles it.
                    indicator_input_data = current_tick_prices[-(min(100, len(current_tick_prices))):]
                    logging.debug(f"Tick received; chart type OHLCV, but no candles. Using last {len(indicator_input_data)} tick prices for indicators.")
            else: # current_chart_type == 'tick'
                # When chart is Tick, indicators are based on recent tick prices (close prices)
                current_tick_prices = market_data['prices']
                indicator_input_data = current_tick_prices[-(min(100, len(current_tick_prices))):]
                # logging.debug(f"Tick received; chart type is Tick. Using last {len(indicator_input_data)} tick prices for indicators.")

            updated_data = {} 
            if not indicator_input_data:
                logging.warning("No data (tick or ohlcv) available for indicator calculation on new tick. Skipping.")
            else:
                # logging.debug(f"Calling calculate_indicators with data of type: {type(indicator_input_data[0] if indicator_input_data else None)}, length: {len(indicator_input_data)}")
                updated_data = calculate_indicators(indicator_input_data)
            
            for key, value in updated_data.items():
                if key in market_data: 
                    if isinstance(value, list):
                        market_data[key] = value 
                    elif isinstance(value, str):
                        market_data[key] = value 
            
            if 'volumes' in market_data:
                 market_data['volumes'].append(np.random.randint(1000,5000))
                 if len(market_data['volumes']) > 100: market_data['volumes'] = market_data['volumes'][-100:]

    elif msg_type == 'candles':
        raw_candles_list = data.get('candles', [])
        request_info = None
        request_type_str = 'unknown_candles'

        if req_id_of_message is not None:
            with shared_data_lock:
                # .pop() here as candle data is processed immediately into market_data, not via event for Flask
                request_info = pending_api_requests.pop(req_id_of_message, None)
            if request_info:
                request_type_str = request_info.get('type', 'unknown_candles_type_in_details')
        
        logging.info(f"Received 'candles' data (req_id: {req_id_of_message}, type: {request_type_str}): {str(message)[:300]}...")
        logging.debug(f"Received {len(raw_candles_list)} raw candle objects for req_id {req_id_of_message} (type: {request_type_str}).")

        if request_type_str == 'ohlcv_chart_data':
            parsed_ohlcv_candles = []
            if raw_candles_list:
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
                logging.debug(f"Parsed {len(parsed_ohlcv_candles)} candles for OHLCV (req_id: {req_id_of_message}, type: {request_type_str}). First 1-2: {parsed_ohlcv_candles[:2] if parsed_ohlcv_candles else []}")
                with market_data_lock:
                    market_data['ohlcv_candles'] = parsed_ohlcv_candles
                logging.info(f"Updated market_data['ohlcv_candles'] with {len(parsed_ohlcv_candles)} candles (req_id: {req_id_of_message}, type: {request_type_str}).")
                if current_chart_type == 'ohlcv': # Or always recalculate if new candles arrive, regardless of chart type
                    logging.info(f"OHLCV data received (req_id: {req_id_of_message}, type: {request_type_str}), recalculating all indicators using these candles.")
                    # prices_for_recalc = [c['close'] for c in parsed_ohlcv_candles] # Old way
                    if parsed_ohlcv_candles: # parsed_ohlcv_candles is the list of dicts
                        updated_data_from_ohlcv = calculate_indicators(parsed_ohlcv_candles)
                        with market_data_lock:
                            for key, value in updated_data_from_ohlcv.items():
                                if key in market_data: # Ensure key exists in DEFAULT_MARKET_DATA
                                    if isinstance(value, list): market_data[key] = value
                                    elif isinstance(value, str): market_data[key] = value
                        logging.info(f"All indicators recalculated using new OHLCV data (req_id: {req_id_of_message}, type: {request_type_str}). ATR: {'Yes' if 'atr_14' in updated_data_from_ohlcv and updated_data_from_ohlcv['atr_14'] else 'No/Empty'}")
            else:
                logging.warning(f"No candle data in 'ohlcv_chart_data' response (req_id: {req_id_of_message}).")

        elif request_type_str == 'daily_summary_data':
            if raw_candles_list:
                candle = raw_candles_list[0]
                if isinstance(candle, dict):
                    with market_data_lock:
                        market_data['high_24h'] = float(candle.get('high')) if candle.get('high') is not None else 'N/A'
                        market_data['low_24h'] = float(candle.get('low')) if candle.get('low') is not None else 'N/A'
                        market_data['volume_24h'] = float(candle.get('volume', candle.get('vol'))) if candle.get('volume', candle.get('vol')) is not None else 'N/A'
                    logging.info(f"Updated 24h data from daily_summary_data (req_id: {req_id_of_message}): High={market_data['high_24h']}, Low={market_data['low_24h']}, Volume={market_data['volume_24h']}")
                else: 
                    logging.warning(f"Daily candle data item not in dict format (req_id: {req_id_of_message}, type: {request_type_str}).")
            else: 
                logging.warning(f"No candle data in 'daily_summary_data' response (req_id: {req_id_of_message}).")
        else:
            logging.warning(f"Received 'candles' message with unexpected or untracked req_id: {req_id_of_message} (type: {request_type_str}). Message: {str(message)[:300]}")

    # Handle responses for requests that use events (balance, portfolio, proposal, buy)
    elif req_id_of_message and request_details and 'event' in request_details:
        logging.info(f"Processing event-based response for req_id: {req_id_of_message}, type: {request_details['type']}, msg_type: {msg_type}")
        request_details['data'] = data.get(msg_type, data) # Store the relevant part of the message or the whole message
        request_details['event'].set() # Signal the waiting Flask endpoint
        with shared_data_lock: # Remove after processing
            if req_id_of_message in pending_api_requests:
                 pending_api_requests.pop(req_id_of_message)

    # Fallback for unhandled messages with req_id but no event (should be rare now)
    elif req_id_of_message and request_details:
        logging.warning(f"Message with req_id {req_id_of_message} (type: {request_details.get('type')}) was not handled by specific logic and had no event. Msg Type: {msg_type}. Data: {str(data)[:200]}")
        with shared_data_lock: # Remove to prevent buildup
            if req_id_of_message in pending_api_requests:
                 pending_api_requests.pop(req_id_of_message)

    # Fallback for messages without req_id and not handled by msg_type
    elif not req_id_of_message and msg_type not in ['tick', 'subscribe', 'authorize', 'forget', 'candles']:
         logging.warning(f"Unhandled message type '{msg_type}' without req_id: {str(data)[:200]}")


def on_error_for_deriv(ws, error):
    # This handles WebSocket connection level errors, not API logical errors.
    # API logical errors (e.g. bad request params) are in on_message.
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
    global next_req_id, pending_api_requests, shared_data_lock
    logging.info(f"[Token Check] Starting token validation for token: {'********' if token else 'No token'}")

    # Use a temporary WebSocket connection for this check
    temp_ws = websocket.create_connection(DERIV_WS_URL, timeout=10)
    if not temp_ws or not temp_ws.connected:
        logging.error("[Token Check] Could not create WebSocket connection for token check.")
        return {'success': False, 'error': "Failed to connect to Deriv WebSocket."}

    validation_result = {'success': False, 'error': None}

    try:
        # Deriv API requires a unique req_id for each request for proper tracking,
        # even for a simple authorize check on a temporary connection.
        with shared_data_lock: # Protect next_req_id
            check_req_id = next_req_id
            next_req_id += 1
            # No event needed here as we are doing a blocking recv

        authorize_payload = {"authorize": token, "req_id": check_req_id}
        logging.debug(f"[Token Check] Sending authorize request: {json.dumps(authorize_payload)}")
        temp_ws.send(json.dumps(authorize_payload))

        # Wait for the response
        response_str = temp_ws.recv() # Blocking receive
        if not response_str:
            validation_result['error'] = "No response received from Deriv API for token check."
            logging.error(f"[Token Check] {validation_result['error']}")
        else:
            response_data = json.loads(response_str)
            logging.debug(f"[Token Check] Received response: {response_data}")

            if response_data.get('echo_req', {}).get('req_id') != check_req_id:
                validation_result['error'] = "Mismatched req_id in token check response."
                logging.error(f"[Token Check] {validation_result['error']}")
            elif response_data.get('error'):
                validation_result['error'] = response_data['error'].get('message', 'Unknown authorization error.')
                logging.error(f"[Token Check] Token validation error: {validation_result['error']}")
            elif response_data.get('msg_type') == 'authorize' and response_data.get('authorize'):
                validation_result['success'] = True
                logging.info("[Token Check] Token validation successful.")
            else:
                validation_result['error'] = "Unexpected response format during token validation."
                logging.warning(f"[Token Check] {validation_result['error']} - Response: {response_data}")

    except websocket.WebSocketTimeoutException:
        validation_result['error'] = "WebSocket timeout during token validation."
        logging.error(f"[Token Check] {validation_result['error']}")
    except websocket.WebSocketConnectionClosedException:
        validation_result['error'] = "WebSocket connection closed unexpectedly during token validation."
        logging.error(f"[Token Check] {validation_result['error']}")
    except json.JSONDecodeError:
        validation_result['error'] = "Failed to decode JSON response from Deriv API."
        logging.error(f"[Token Check] {validation_result['error']}")
    except Exception as e:
        validation_result['error'] = f"An unexpected error occurred during token validation: {str(e)}"
        logging.exception(f"[Token Check] Unexpected error: {e}")
    finally:
        if temp_ws and temp_ws.connected:
            temp_ws.close()
            logging.debug("[Token Check] Temporary WebSocket closed.")

    logging.info(f"[Token Check] Finished token validation. Success: {validation_result['success']}")
    return validation_result

# --- Helper function to send requests and wait for responses ---
def send_ws_request_and_wait(payload_type: str, payload: dict, timeout_seconds=10):
    global next_req_id, pending_api_requests, ws_app, shared_data_lock

    if not (ws_app and ws_app.sock and ws_app.sock.connected):
        logging.error(f"WebSocket not connected. Cannot send {payload_type} request.")
        return None, "WebSocket not connected."

    if not API_TOKEN: # Most of these requests require auth
        logging.error(f"API TOKEN not set. Cannot send {payload_type} request.")
        return None, "API token not set."

    response_event = threading.Event()
    request_entry = {'type': payload_type, 'event': response_event, 'data': None, 'error': None}

    with shared_data_lock:
        current_req_id = next_req_id
        next_req_id += 1
        payload['req_id'] = current_req_id
        pending_api_requests[current_req_id] = request_entry

    logging.info(f"Sending {payload_type} request (req_id: {current_req_id}): {json.dumps(payload)}")
    ws_app.send(json.dumps(payload))

    if response_event.wait(timeout=timeout_seconds):
        # Event was set
        with shared_data_lock: # Ensure atomicity of accessing and clearing the entry
            # The entry might have been removed by on_message if error occurred there
            # or if it was processed there. This re-check is to ensure we get the data if available.
            final_request_details = pending_api_requests.pop(current_req_id, request_entry)

        if final_request_details.get('error'):
            logging.error(f"{payload_type} request (req_id: {current_req_id}) failed: {final_request_details['error']}")
            return None, final_request_details['error']
        elif final_request_details.get('data') is not None:
            logging.info(f"{payload_type} request (req_id: {current_req_id}) successful.")
            return final_request_details['data'], None
        else: # Should not happen if event was set without error/data
            logging.warning(f"{payload_type} request (req_id: {current_req_id}) event set but no data/error found.")
            return None, "Response event set but no data or error recorded."
            
    else: # Timeout
        logging.warning(f"{payload_type} request (req_id: {current_req_id}) timed out after {timeout_seconds}s.")
        with shared_data_lock: # Clean up on timeout
            pending_api_requests.pop(current_req_id, None)
        return None, f"{payload_type} request timed out."

# --- Account and Trading Endpoints ---
@app.route('/api/account/balance', methods=['GET'])
def get_account_balance():
    if not API_TOKEN:
        return jsonify({'status': 'error', 'message': 'API token not configured or user not connected.'}), 401
    
    balance_payload = {"balance": 1} # "subscribe": 1 can be added if continuous updates are needed
    
    data, error = send_ws_request_and_wait('account_balance', balance_payload)

    if error:
        return jsonify({'status': 'error', 'message': str(error)}), 500
    if data and data.get('balance'):
        return jsonify({'status': 'success', 'balance': data['balance']})
    else:
        logging.error(f"Unexpected data format for balance: {data}")
        return jsonify({'status': 'error', 'message': 'Unexpected data format from Deriv API for balance.'}), 500

@app.route('/api/account/open_positions', methods=['GET'])
def get_open_positions():
    if not API_TOKEN:
        return jsonify({'status': 'error', 'message': 'API token not configured or user not connected.'}), 401

    portfolio_payload = {"portfolio": 1}
    data, error = send_ws_request_and_wait('account_portfolio', portfolio_payload)

    if error:
        return jsonify({'status': 'error', 'message': str(error)}), 500
    if data and data.get('portfolio'):
        return jsonify({'status': 'success', 'portfolio': data['portfolio']})
    else:
        logging.error(f"Unexpected data format for portfolio: {data}")
        return jsonify({'status': 'error', 'message': 'Unexpected data format from Deriv API for portfolio.'}), 500

@app.route('/api/trade/execute', methods=['POST'])
def execute_trade():
    if not API_TOKEN:
        return jsonify({'status': 'error', 'message': 'API token not configured or user not connected.'}), 401

    trade_params = request.get_json()
    if not trade_params:
        return jsonify({'status': 'error', 'message': 'No trade parameters provided.'}), 400

    symbol = trade_params.get('symbol')
    trade_type = trade_params.get('trade_type', '').upper() # Expect 'BUY' or 'SELL' (CALL/PUT)
    amount = trade_params.get('amount') # Stake

    if not all([symbol, trade_type, amount]):
        return jsonify({'status': 'error', 'message': 'Missing parameters: symbol, trade_type, or amount.'}), 400

    try:
        amount = float(amount) # Or int depending on Deriv requirements for stake
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid amount format.'}), 400

    if trade_type not in ["BUY"]: # For now only BUY (CALL for rise)
         return jsonify({'status': 'error', 'message': f"Trade type '{trade_type}' not yet supported."}), 400

    # 1. Get Proposal
    # For simple market buy, duration might be 1 tick or a short duration if it's a contract like Rise/Fall
    # This example assumes a "CALL" (expecting price to rise) contract for a short duration (e.g., 1 minute or few ticks)
    # Adjust 'duration_unit' and 'duration' as needed by the specific contract type on Deriv.
    # For some assets, market orders might not need proposal if it's direct asset purchase.
    # This structure is for contracts that need proposal then buy.
    proposal_payload = {
        "proposal": 1,
        "contract_type": "CALL", # For a "buy" expecting price to rise
        "symbol": symbol,
        "amount": amount, # Stake
        "currency": "USD", # Or get from account settings
        "duration": 60, # Example: 60 seconds
        "duration_unit": "s",
        "barrier": "+0" # Example for simple rise, adjust if needed
    }

    logging.info(f"Requesting trade proposal for {symbol}, Amount: {amount}, Type: CALL")
    proposal_data, proposal_error = send_ws_request_and_wait('trade_proposal', proposal_payload, timeout_seconds=10)

    if proposal_error:
        logging.error(f"Trade proposal failed: {proposal_error}")
        return jsonify({'status': 'error', 'message': f"Trade proposal failed: {proposal_error}"}), 500

    if not proposal_data or 'proposal' not in proposal_data or 'id' not in proposal_data['proposal']:
        logging.error(f"Invalid proposal data received: {proposal_data}")
        return jsonify({'status': 'error', 'message': 'Invalid proposal data received from Deriv API.'}), 500

    proposal_id = proposal_data['proposal']['id']
    proposed_price = proposal_data['proposal'].get('ask_price') # Or spot, depending on contract

    logging.info(f"Proposal successful. ID: {proposal_id}, Price: {proposed_price}. Proceeding to buy.")

    # 2. Execute Buy
    buy_payload = {
        "buy": proposal_id,
        "price": proposed_price # Or a sufficiently high price for market orders, Deriv usually uses the price from proposal
    }

    buy_data, buy_error = send_ws_request_and_wait('trade_buy', buy_payload, timeout_seconds=15)

    if buy_error:
        logging.error(f"Trade execution (buy) failed: {buy_error}")
        return jsonify({'status': 'error', 'message': f"Trade execution failed: {buy_error}"}), 500

    if buy_data and buy_data.get('buy'):
        logging.info(f"Trade executed successfully: {buy_data['buy']}")
        return jsonify({'status': 'success', 'trade_confirmation': buy_data['buy']})
    else:
        logging.error(f"Invalid buy confirmation data: {buy_data}")
        return jsonify({'status': 'error', 'message': 'Invalid trade confirmation received.'}), 500


start_deriv_ws()

if __name__ == '__main__':
    app.run(debug=True)
