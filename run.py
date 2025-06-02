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

DEFAULT_MARKET_DATA = {
    'timestamps': [], 'prices': [], 'volumes': [],
    'rsi': [], 'macd': [], 'bollinger_upper': [], 'bollinger_middle': [],
    'bollinger_lower': [], 'stochastic': [], 'cci_20': [],
    'sma_10': [], 'sma_20': [], 'sma_30': [], 'sma_50': [], 'sma_100': [], 'sma_200': [],
    'ema_10': [], 'ema_20': [], 'ema_30': [], 'ema_50': [], 'ema_100': [], 'ema_200': [],
    # Signals
    'rsi_signal': 'N/A', 'stochastic_signal': 'N/A', 'macd_signal': 'N/A',
    'cci_20_signal': 'N/A', 'price_ema_50_signal': 'N/A',
    'market_sentiment_text': 'N/A', # Added in previous step, ensure it's here
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
    "BTC/USD": "cryBTCUSD", # Assumption
    "ETH/USD": "cryETHUSD"  # Assumption
}

SYMBOL = "frxUSDJPY"  # Default symbol, MUST be in Deriv API format
API_TOKEN = ''  # Set your token here or via environment
current_tick_subscription_id = None # Store the ID of the active tick subscription

# Helper: Calculate indicators

def calculate_indicators(prices_list: list[float]):
    """
    Calculates various technical indicators based on a list of prices.
    """
    logging.debug(f"calculate_indicators received prices_list length: {len(prices_list)}, last 5: {prices_list[-5:] if len(prices_list) >= 5 else prices_list}")
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
    rsi_series = rsi.fillna(50) # Fill initial NaNs and specific (gain=0, loss=0) NaNs with 50
    # logging.debug(f"[Indicators] RSI calculated. Head: {rsi_series.head().tolist()}, Tail: {rsi_series.tail().tolist()}") # Original detailed log
    logging.debug(f"calculate_indicators computed RSI, length: {len(rsi_series)}, last 5: {rsi_series.tolist()[-5:] if len(rsi_series) >= 5 else rsi_series.tolist()}")


    # MACD
    # Using adjust=False for EWMA is common in financial calculations.
    # This calculates the MACD line. Signal line and histogram are not part of this.
    logging.debug("[Indicators] MACD: Calculating MACD line (not signal line or histogram).")
    macd_line_series = prices_series.ewm(span=12, adjust=False).mean() - prices_series.ewm(span=26, adjust=False).mean()
    macd_line_series = macd_line_series.fillna(0) # Fill initial NaNs with 0
    # logging.debug(f"[Indicators] MACD line calculated. Head: {macd_line_series.head().tolist()}, Tail: {macd_line_series.tail().tolist()}") # Original detailed log
    logging.debug(f"calculate_indicators computed MACD, length: {len(macd_line_series)}, last 5: {macd_line_series.tolist()[-5:] if len(macd_line_series) >= 5 else macd_line_series.tolist()}")

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


    results = {
        'rsi': rsi_series.tolist(),
        'macd': macd_line_series.tolist(),
        'bollinger_upper': bollinger_upper.tolist(),
        'bollinger_middle': bollinger_middle.tolist(),
        'bollinger_lower': bollinger_lower.tolist(),
        'stochastic': stochastic_k.tolist()
    }

    # SMA Calculations
    sma_periods = [10, 20, 30, 50, 100, 200]
    for N in sma_periods:
        sma_N = prices_series.rolling(window=N).mean().fillna(0) # Using fillna(0) for now
        results[f'sma_{N}'] = sma_N.tolist()
    logging.debug(f"calculate_indicators computed SMAs. Last SMA_200 value: {results.get('sma_200', [])[-1:]}")

    # EMA Calculations
    ema_periods = [10, 20, 30, 50, 100, 200]
    for N in ema_periods:
        ema_N = prices_series.ewm(span=N, adjust=False).mean().fillna(0) # Using fillna(0) for now
        results[f'ema_{N}'] = ema_N.tolist()
    logging.debug(f"calculate_indicators computed EMAs. Last EMA_200 value: {results.get('ema_200', [])[-1:]}")

    # CCI (Commodity Channel Index) Calculation (20-period)
    cci_period = 20
    tp = prices_series # Using close price as Typical Price
    sma_tp = tp.rolling(window=cci_period).mean()
    # Calculate Mean Deviation (MD)
    # For MD, ensure we are calculating the mean of absolute differences from sma_tp over the lookback period.
    # Pandas rolling apply can be used here, but direct rolling mean of absolute differences is more straightforward if definition aligns.
    # The common definition is Sum(Abs(TP - SMA_TP)) / N for the MD part of the constant.
    # Let's use pandas rolling mean on the absolute difference series.
    mean_dev = (tp - sma_tp).abs().rolling(window=cci_period).mean()

    # Constant 0.015 is standard
    # Handle division by zero for md: if md is 0, cci is effectively infinite or undefined.
    # Replace inf with a large number or 0, or NaN. Using 0 for now.
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    cci = cci.replace([np.inf, -np.inf], 0).fillna(0) # Replace inf/(-inf) with 0 and then fill NaNs with 0

    results['cci_20'] = cci.tolist()
    logging.debug(f"calculate_indicators computed CCI_20, length: {len(results['cci_20'])}, last 5: {results['cci_20'][-5:] if len(results['cci_20']) >= 5 else results['cci_20']}")

    # Signal Generation
    # Ensure there's enough data to get latest values. prices_series is used for latest_price.
    if not prices_series.empty:
        latest_price = prices_series.iloc[-1]

        # RSI Signal
        if results['rsi'] and len(results['rsi']) > 0:
            latest_rsi = results['rsi'][-1]
            if latest_rsi < 30: results['rsi_signal'] = 'Buy'
            elif latest_rsi > 70: results['rsi_signal'] = 'Sell'
            else: results['rsi_signal'] = 'Neutral'
        else:
            results['rsi_signal'] = 'N/A'

        # Stochastic Signal
        if results['stochastic'] and len(results['stochastic']) > 0:
            latest_stochastic = results['stochastic'][-1]
            if latest_stochastic < 20: results['stochastic_signal'] = 'Buy'
            elif latest_stochastic > 80: results['stochastic_signal'] = 'Sell'
            else: results['stochastic_signal'] = 'Neutral'
        else:
            results['stochastic_signal'] = 'N/A'

        # MACD Signal (MACD line vs. Zero)
        if results['macd'] and len(results['macd']) > 0:
            latest_macd = results['macd'][-1]
            if latest_macd > 0: results['macd_signal'] = 'Buy'
            elif latest_macd < 0: results['macd_signal'] = 'Sell'
            else: results['macd_signal'] = 'Neutral'
        else:
            results['macd_signal'] = 'N/A'

        # CCI Signal
        if results['cci_20'] and len(results['cci_20']) > 0:
            latest_cci = results['cci_20'][-1]
            if latest_cci < -100: results['cci_signal'] = 'Buy'
            elif latest_cci > 100: results['cci_signal'] = 'Sell'
            else: results['cci_signal'] = 'Neutral'
        else:
            results['cci_signal'] = 'N/A'

        # Price vs. EMA(50) Signal
        if results['ema_50'] and len(results['ema_50']) > 0:
            latest_ema_50 = results['ema_50'][-1]
            if latest_price > latest_ema_50: results['price_ema_50_signal'] = 'Buy'
            elif latest_price < latest_ema_50: results['price_ema_50_signal'] = 'Sell'
            else: results['price_ema_50_signal'] = 'Neutral'
        else:
            results['price_ema_50_signal'] = 'N/A'
    else:
        # Default signals if no price data
        results['rsi_signal'] = 'N/A'
        results['stochastic_signal'] = 'N/A'
        results['macd_signal'] = 'N/A'
        results['cci_signal'] = 'N/A'
        results['price_ema_50_signal'] = 'N/A'

    logging.debug(f"Generated signals: RSI: {results.get('rsi_signal')}, MACD: {results.get('macd_signal')}, Price/EMA50: {results.get('price_ema_50_signal')}")

    # Market Sentiment Logic
    signal_keys = ['rsi_signal', 'stochastic_signal', 'macd_signal', 'cci_20_signal', 'price_ema_50_signal']
    active_signals = [results.get(key) for key in signal_keys if results.get(key) and results.get(key) not in ['N/A', 'Neutral']]

    buy_count = active_signals.count('Buy')
    sell_count = active_signals.count('Sell')

    if buy_count > sell_count:
        results['market_sentiment_text'] = 'Bullish'
    elif sell_count > buy_count:
        results['market_sentiment_text'] = 'Bearish'
    else:
        results['market_sentiment_text'] = 'Neutral'

    logging.debug(f"Sentiment calculation: Buy signals: {buy_count}, Sell signals: {sell_count}, Overall Sentiment: {results['market_sentiment_text']}")

    # Prediction States Generation
    if not prices_series.empty:
        # RSI Prediction State
        if results['rsi'] and len(results['rsi']) > 0:
            latest_rsi = results['rsi'][-1]
            if latest_rsi < 30: results['rsi_prediction_state'] = 'Oversold'
            elif latest_rsi > 70: results['rsi_prediction_state'] = 'Overbought'
            else: results['rsi_prediction_state'] = 'Neutral'
        else:
            results['rsi_prediction_state'] = 'N/A'

        # Stochastic Prediction State
        if results['stochastic'] and len(results['stochastic']) > 0:
            latest_stochastic = results['stochastic'][-1]
            if latest_stochastic < 20: results['stochastic_prediction_state'] = 'Oversold'
            elif latest_stochastic > 80: results['stochastic_prediction_state'] = 'Overbought'
            else: results['stochastic_prediction_state'] = 'Neutral'
        else:
            results['stochastic_prediction_state'] = 'N/A'

        # MACD Prediction State
        if results['macd'] and len(results['macd']) > 0:
            latest_macd = results['macd'][-1]
            if latest_macd > 0: results['macd_prediction_state'] = 'Bullish'
            elif latest_macd < 0: results['macd_prediction_state'] = 'Bearish'
            else: results['macd_prediction_state'] = 'Neutral'
        else:
            results['macd_prediction_state'] = 'N/A'

        # CCI Prediction State
        if results['cci_20'] and len(results['cci_20']) > 0:
            latest_cci = results['cci_20'][-1]
            if latest_cci < -100: results['cci_20_prediction_state'] = 'Oversold'
            elif latest_cci > 100: results['cci_20_prediction_state'] = 'Overbought'
            else: results['cci_20_prediction_state'] = 'Neutral'
        else:
            results['cci_20_prediction_state'] = 'N/A'
    else:
        results['rsi_prediction_state'] = 'N/A'
        results['stochastic_prediction_state'] = 'N/A'
        results['macd_prediction_state'] = 'N/A'
        results['cci_20_prediction_state'] = 'N/A'

    logging.debug(f"Prediction States: RSI: {results.get('rsi_prediction_state')}, Stochastic: {results.get('stochastic_prediction_state')}, MACD: {results.get('macd_prediction_state')}, CCI: {results.get('cci_20_prediction_state')}")

    # Ensure all returned lists are of the same length as the input prices_series
    # Pandas rolling/ewm operations with default settings should preserve index and length, filling leading values with NaN (before our fillna(0)).
    return results

# WebSocket thread to fetch real data from Deriv

ws_thread = None
ws_app = None

def request_daily_data(ws_app_instance, symbol_to_fetch):
    if ws_app_instance and ws_app_instance.sock and ws_app_instance.sock.connected:
        logging.info(f"Requesting daily OHLCV data for {symbol_to_fetch}...")
        # Using a unique req_id is best practice
        req_id_daily = f"daily_{symbol_to_fetch}_{int(time.time())}"
        ws_app_instance.send(json.dumps({
            "ticks_history": symbol_to_fetch,
            "style": "candles",
            "granularity": 86400, # Daily candles (24 hours * 60 minutes * 60 seconds)
            "end": "latest",    # Get data up to the latest available tick
            "count": 1,         # We only need the most recent daily candle data
            "req_id": req_id_daily
        }))
    else:
        logging.warning(f"WebSocket not connected. Cannot request daily data for {symbol_to_fetch}.")

# Enhanced WebSocket Handlers
def on_open_for_deriv(ws):
    """Called when the WebSocket connection is established."""
    logging.info(f"WebSocket connection opened. Attempting to subscribe to ticks for SYMBOL: {SYMBOL} or authorize.")
    request_daily_data(ws, SYMBOL) # Request daily data on new connection
    if API_TOKEN:
        logging.info(f"[Deriv WS] API_TOKEN is present. Attempting to authorize.")
        ws.send(json.dumps({"authorize": API_TOKEN}))
    else:
        logging.info(f"[Deriv WS] No API_TOKEN provided. Subscribing to public ticks for {SYMBOL}.")
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
            # Also request daily data after successful authorization
            request_daily_data(ws, SYMBOL)

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

                current_prices = market_data['prices'][:] # Use a copy for calculation
                logging.debug(f"Before indicator calculation for SYMBOL: {SYMBOL} - market_data['prices'] length: {len(market_data['prices'])}, last 5 prices: {market_data['prices'][-5:] if len(market_data['prices']) >= 5 else market_data['prices']}")

                updated_indicators = {} # Define to ensure it's available in case of exception
                try:
                    updated_indicators = calculate_indicators(current_prices) # Renamed for clarity
                    logging.debug(f"After indicator calculation for SYMBOL: {SYMBOL} - calculated RSI length: {len(updated_indicators.get('rsi', []))}, last 5 RSI: {updated_indicators.get('rsi', [])[-5:] if len(updated_indicators.get('rsi', [])) >= 5 else updated_indicators.get('rsi', [])}")
                    logging.debug(f"After indicator calculation for SYMBOL: {SYMBOL} - calculated MACD length: {len(updated_indicators.get('macd', []))}, last 5 MACD: {updated_indicators.get('macd', [])[-5:] if len(updated_indicators.get('macd', [])) >= 5 else updated_indicators.get('macd', [])}")

                    for key, value in updated_indicators.items():
                        if isinstance(value, list):
                            # This is an indicator series (e.g., rsi, sma_10)
                            market_data[key] = value[-100:] # Store last 100 points
                            logging.debug(f"Updating market_data list '{key}'. Length: {len(market_data[key])}. Last 5: {market_data[key][-5:] if len(market_data[key]) >= 5 else market_data[key]}")
                        elif isinstance(value, str):
                            # This is a signal string (e.g., rsi_signal, macd_signal)
                            market_data[key] = value # Store the single string value
                            logging.debug(f"Updating market_data signal '{key}' to: {value}")
                        else:
                            logging.warning(f"Unexpected data type for key '{key}' in updated_indicators: {type(value)}")
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

    elif data.get('msg_type') == 'candles':
        # Ideally, match data.get('echo_req', {}).get('req_id') here
        # with the req_id_daily sent, if we were using a robust req_id system.
        logging.info(f"Received 'candles' data: {message[:250]}...") # Log snippet
        candles = data.get('candles', [])
        if candles:
            latest_candle = candles[0] # Deriv returns latest candle first with count:1
            # Candle format: {epoch, open, high, low, close, volume} (volume is optional)
            if isinstance(latest_candle, dict):
                high_24h = latest_candle.get('high')
                low_24h = latest_candle.get('low')
                # Deriv 'volume' in candles might be actual trade volume or tick count, depending on instrument.
                # For synthetic indices it's usually tick count. For Forex, it might be absent or different.
                # For this dashboard, we'll call it 'volume_24h' generically.
                volume_24h = latest_candle.get('volume', latest_candle.get('vol')) # Try 'volume' then 'vol'

                with market_data_lock:
                    market_data['high_24h'] = float(high_24h) if high_24h is not None else 'N/A'
                    market_data['low_24h'] = float(low_24h) if low_24h is not None else 'N/A'
                    market_data['volume_24h'] = float(volume_24h) if volume_24h is not None else 'N/A'
                logging.info(f"Updated 24h data: High={market_data['high_24h']}, Low={market_data['low_24h']}, Volume={market_data['volume_24h']}")
            else:
                logging.warning(f"Received candle data is not in expected dict format: {latest_candle}")
        else:
            logging.warning("Received 'candles' message but no candle data found.")
        # No explicit 'forget' is needed for one-time requests like ticks_history with count=1.

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
        market_data.clear()
        market_data.update(copy.deepcopy(DEFAULT_MARKET_DATA))
        logging.info(f"[Deriv WS] Market data cleared and re-initialized for symbol {SYMBOL}.")

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
    new_symbol_value = data.get('symbol') # User-friendly symbol, e.g., "USD/JPY"

    if not new_symbol_value or not isinstance(new_symbol_value, str):
        logging.error(f"[/api/set_symbol] Invalid symbol value provided: {new_symbol_value}")
        return jsonify({'status': 'error', 'message': 'Invalid symbol format'}), 400

    actual_deriv_symbol = DERIV_SYMBOL_MAPPING.get(new_symbol_value, new_symbol_value)
    logging.info(f"Symbol received from frontend: '{new_symbol_value}', Mapped to Deriv symbol: '{actual_deriv_symbol}'")

    if actual_deriv_symbol == new_symbol_value and not any(new_symbol_value.startswith(p) for p in ['frx', 'cry', 'R_']):
        logging.warning(f"Symbol '{new_symbol_value}' was not found in DERIV_SYMBOL_MAPPING and might not be a valid Deriv API symbol format.")

    # SYMBOL global should always store the Deriv API format
    logging.info(f"[/api/set_symbol] Attempting to change global SYMBOL from '{SYMBOL}' to '{actual_deriv_symbol.strip()}'")
    SYMBOL = actual_deriv_symbol.strip()


    # Reset market_data is now primarily handled by start_deriv_ws.
    # If ws is not connected, we still need to clear/reset it here for consistency
    # if the symbol changes but connection isn't immediate.
    if not (ws_app and ws_app.sock and ws_app.sock.connected):
        with market_data_lock:
            logging.info(f"[/api/set_symbol] WebSocket not connected. Clearing market data for new symbol {SYMBOL} locally.")
            market_data.clear()
            market_data.update(copy.deepcopy(DEFAULT_MARKET_DATA))
            logging.info(f"[/api/set_symbol] Market data cleared and re-initialized locally.")

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
