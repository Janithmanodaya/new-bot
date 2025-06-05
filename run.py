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
import os
import uuid
import sqlite3 # Added
from database import get_db_connection, create_tables # Added
# json is already imported
# datetime, timezone are already imported
import asyncio # Added for websockets
from asyncio import Semaphore # Added for rate limiting
import websockets # Added for websockets
# threading is already imported
from concurrent.futures import ThreadPoolExecutor # Added for async indicators

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

app = Flask(__name__)
CORS(app)

# Ensure database tables are created on application startup
create_tables()
logging.info("Database tables checked/created on startup.")

# --- Strategy Management Globals and Functions ---
custom_strategies = {} # In-memory storage for strategies: {strategy_id: strategy_object}
strategies_lock = threading.Lock() # Lock for accessing custom_strategies

def load_strategies_from_db():
    global custom_strategies
    logging.info("Loading strategies from database...")
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch all non-deleted strategies
    cursor.execute("SELECT * FROM strategies WHERE is_deleted = FALSE")
    strategies_rows = cursor.fetchall()

    new_strategies_cache = {}
    for strategy_row in strategies_rows:
        strategy_id = strategy_row['strategy_id']
        logging.debug(f"Loading strategy ID: {strategy_id}")

        # Fetch the latest version for this strategy
        # Using created_at from strategy_versions for latest, as version_id might not always reflect true latest if old versions are imported later.
        # Or, if version_id is guaranteed to be incremental for new versions, it's simpler. Assuming version_id is reliable for latest.
        cursor.execute("""
            SELECT * FROM strategy_versions
            WHERE strategy_id = ?
            ORDER BY version_id DESC
            LIMIT 1
        """, (strategy_id,))
        latest_version_row = cursor.fetchone()

        if latest_version_row:
            try:
                conditions_group = json.loads(latest_version_row['conditions_group'])
                actions = json.loads(latest_version_row['actions'])

                reconstructed_strategy = {
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_row['strategy_name'], # From strategies table
                    "description": strategy_row['description'], # From strategies table
                    "conditions_group": conditions_group, # Parsed from version
                    "actions": actions, # Parsed from version
                    "is_active": bool(strategy_row['is_active']),
                    "created_at": strategy_row['created_at'], # Original creation from strategies table
                    "updated_at": strategy_row['updated_at'], # Last update from strategies table
                    # "version_id": latest_version_row['version_id'], # Optional: if frontend needs to know
                    # "version_notes": latest_version_row['version_notes'] # Optional
                }
                new_strategies_cache[strategy_id] = reconstructed_strategy
                logging.debug(f"Successfully loaded and reconstructed strategy ID: {strategy_id} with version ID: {latest_version_row['version_id']}")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON for strategy ID {strategy_id}, version ID {latest_version_row['version_id']}: {e}")
            except Exception as e:
                logging.error(f"Error processing strategy ID {strategy_id}, version ID {latest_version_row['version_id']}: {e}")
        else:
            logging.warning(f"No versions found for strategy ID: {strategy_id} in strategy_versions table. Skipping this strategy.")

    conn.close()
    with strategies_lock:
        custom_strategies = new_strategies_cache
    logging.info(f"Loaded {len(custom_strategies)} strategies from database into memory.")

def save_strategy_to_db(strategy_data, is_update=False):
    """
    Saves a strategy to the database. Creates a new strategy record and a version record,
    or updates an existing strategy record and creates a new version record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    strategy_id = strategy_data['strategy_id']
    strategy_name = strategy_data['strategy_name']
    description = strategy_data.get('description', '')
    # is_active is handled by specific endpoints or defaults in table

    current_time_utc_iso = datetime.now(timezone.utc).isoformat()

    try:
        if not is_update:
            # New Strategy: Insert into strategies table
            logging.info(f"Creating new strategy in DB with ID: {strategy_id}")
            cursor.execute("""
                INSERT INTO strategies (strategy_id, strategy_name, description, is_active, created_at, updated_at, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (strategy_id, strategy_name, description, strategy_data.get('is_active', True), current_time_utc_iso, current_time_utc_iso, False))
        else:
            # Existing Strategy: Update strategies table
            logging.info(f"Updating existing strategy in DB with ID: {strategy_id}")
            # is_active is usually updated by enable/disable endpoints, but can be part of general update too
            is_active_val = strategy_data.get('is_active', None) # Get from payload if present
            if is_active_val is not None:
                 cursor.execute("""
                    UPDATE strategies SET strategy_name = ?, description = ?, updated_at = ?, is_active = ?
                    WHERE strategy_id = ?
                """, (strategy_name, description, current_time_utc_iso, bool(is_active_val), strategy_id))
            else: # Don't update is_active if not in payload
                 cursor.execute("""
                    UPDATE strategies SET strategy_name = ?, description = ?, updated_at = ?
                    WHERE strategy_id = ?
                """, (strategy_name, description, current_time_utc_iso, strategy_id))


        # Always insert a new version into strategy_versions
        logging.info(f"Creating new version for strategy ID: {strategy_id}")
        conditions_group_json = json.dumps(strategy_data['conditions_group'])
        actions_json = json.dumps(strategy_data.get('actions', []))
        version_notes = strategy_data.get('version_notes', None) # Optional

        cursor.execute("""
            INSERT INTO strategy_versions (strategy_id, strategy_name, description, conditions_group, actions, version_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (strategy_id, strategy_name, description, conditions_group_json, actions_json, version_notes, current_time_utc_iso))

        conn.commit()
        logging.info(f"Strategy ID: {strategy_id} (is_update={is_update}) and its new version saved to DB successfully.")
        return True
    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Database error saving strategy ID {strategy_id}: {e}")
        return False
    except Exception as e:
        conn.rollback()
        logging.error(f"Unexpected error saving strategy ID {strategy_id}: {e}")
        return False
    finally:
        conn.close()

# Load strategies at startup
load_strategies_from_db()
# --- End Strategy Management ---


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
    '1M': 60, '5M': 300, '10M': 600, '15M': 900, '30M': 1800, '1H': 3600, '4H': 14400, '1D': 86400, '1W': 604800
}

MAX_OHLCV_CANDLES = 500 # Max number of historical + aggregated candles to keep
in_progress_ohlcv_candle = None # Holds the currently forming OHLCV candle from ticks

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
    "Gold/USD": "XAUUSD",
    "Volatility 10 Index": "R_10",
    "Volatility 25 Index": "R_25",
    "Volatility 50 Index": "R_50",
    "Volatility 75 Index": "R_75",
    "Volatility 100 Index": "R_100",
    "Jump 10 Index": "JD10",
    "Jump 25 Index": "JD25",
    "Jump 50 Index": "JD50",
    "Jump 75 Index": "JD75",
    "Jump 100 Index": "JD100",
    "Boom 500 Index": "BOOM500",
    "Crash 500 Index": "CRASH500",
    "Step Index": "STPRNG"
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

    if not data_input or (isinstance(data_input, list) and not data_input): # Check if data_input itself is empty or an empty list
        logging.warning("[Indicators] Data input is empty list or None. Cannot calculate indicators.")
        empty_results = {}
        for k, v_default in DEFAULT_MARKET_DATA.items():
            if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']:
                if isinstance(v_default, list): empty_results[k] = []
                elif isinstance(v_default, str): empty_results[k] = 'N/A'
        return empty_results
    
    # Determine data type (list of floats or list of dicts)
    # This check needs to be careful if data_input is a list that could be empty after filtering
    if isinstance(data_input[0], dict): # OHLCV data
        is_ohlcv_data = True
        try:
            # Ensure all candles have the required keys. Filter out malformed ones if necessary.
            # For now, assume if it's dict, it's mostly correct.
            prices_list = [d['close'] for d in data_input if isinstance(d, dict) and 'close' in d]
            if not prices_list: # If all items were malformed
                 logging.error("[Indicators] OHLCV data provided, but no valid 'close' prices found.")
                 raise KeyError("No valid 'close' in OHLCV data")

            high_prices = [d['high'] for d in data_input if isinstance(d, dict) and 'high' in d]
            low_prices = [d['low'] for d in data_input if isinstance(d, dict) and 'low' in d]
            # Check if H/L were also validly extracted matching prices_list length
            if not (len(prices_list) == len(high_prices) == len(low_prices)):
                logging.warning("[Indicators] Mismatch in HLC data lengths from OHLCV input. ATR/CCI might be affected.")
                # Decide if is_ohlcv_data should be False if H/L are not complete. For ATR, it will fail later if lengths differ.
            
            close_prices_for_atr = prices_list 
            logging.debug(f"calculate_indicators received OHLCV data. Count: {len(data_input)}, Valid close prices: {len(prices_list)}")
        except KeyError as e:
            logging.error(f"[Indicators] OHLCV data missing key: {e}. Cannot calculate all indicators reliably.")
            is_ohlcv_data = False 
            if not prices_list: # If 'close' also failed during extraction
                 return {k: [] if isinstance(DEFAULT_MARKET_DATA[k], list) else 'N/A' for k in DEFAULT_MARKET_DATA if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']}
    elif isinstance(data_input[0], float): # List of close prices
        prices_list = data_input
        logging.debug(f"calculate_indicators received list of close prices. Count: {len(prices_list)}")
    else:
        logging.error(f"[Indicators] Unknown data_input type: {type(data_input[0] if data_input else 'None')}. Cannot calculate indicators.")
        return {k: [] if isinstance(DEFAULT_MARKET_DATA[k], list) else 'N/A' for k in DEFAULT_MARKET_DATA if k not in ['timestamps', 'prices', 'volumes', 'ohlcv_candles', 'high_24h', 'low_24h', 'volume_24h']}

    if not prices_list: 
        logging.warning("[Indicators] Price list (derived after type check) is empty. Cannot calculate indicators.")
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

# --- WebSocket (Deriv API) Refactored with asyncio and websockets ---
ws_app = None # Will hold the websockets client connection object
ws_thread = None # Thread for running the asyncio event loop
asyncio_loop = None # The asyncio event loop
current_ws_task = None # The current asyncio.Task for connect_and_listen
executor = ThreadPoolExecutor(max_workers=4) # Added for async indicators
indicator_cache = {} # Added for caching indicator results
indicator_cache_lock = threading.Lock() # Added for indicator_cache access
deriv_api_request_semaphore = Semaphore(5) # Added for rate limiting sensitive requests

# Exponential backoff parameters for WebSocket reconnection
INITIAL_RECONNECT_DELAY = 1.0  # seconds
MAX_RECONNECT_DELAY = 60.0   # seconds
RECONNECT_FACTOR = 2.0
current_reconnect_delay = INITIAL_RECONNECT_DELAY # Tracks current delay, reset on successful connect

async def request_ohlcv_data_async(ws, symbol_to_fetch, granularity_s, candle_count=100):
    global next_req_id, pending_api_requests, shared_data_lock
    if ws is None or not ws.open:
        logging.warning(f"WebSocket not connected or open. Cannot request OHLCV data for {symbol_to_fetch}.")
        return None

    with shared_data_lock:
        current_req_id = next_req_id
        next_req_id += 1
        pending_api_requests[current_req_id] = {'type': 'ohlcv_chart_data'}

    request_payload = {
        "ticks_history": symbol_to_fetch,
        "style": "candles",
        "granularity": int(granularity_s),
        "end": "latest",
        "count": int(candle_count),
        "adjust_start_time": 1,
        "req_id": current_req_id
    }
    logging.debug(f"Sending async ticks_history request for OHLCV (req_id: {current_req_id}): {json.dumps(request_payload)}")
    await ws.send(json.dumps(request_payload))
    return current_req_id

async def request_daily_data_async(ws, symbol_to_fetch):
    global next_req_id, pending_api_requests, shared_data_lock
    if ws is None or not ws.open:
        logging.warning(f"WebSocket not connected or open. Cannot request daily data for {symbol_to_fetch}.")
        return
        
    with shared_data_lock:
        current_req_id = next_req_id
        next_req_id += 1
        pending_api_requests[current_req_id] = {'type': 'daily_summary_data'}

    request_payload = {
        "ticks_history": symbol_to_fetch,
        "style": "candles",
        "granularity": 86400,
        "end": "latest",
        "count": 1,
        "req_id": current_req_id
    }
    logging.info(f"Requesting async daily OHLCV data (req_id: {current_req_id}) for {symbol_to_fetch}... Payload: {json.dumps(request_payload)}")
    await ws.send(json.dumps(request_payload))

async def on_open_for_deriv(ws):
    logging.info(f"Async WebSocket connection opened. SYMBOL: {SYMBOL}, Granularity: {current_granularity_seconds}s")
    await request_daily_data_async(ws, SYMBOL)
    await request_ohlcv_data_async(ws, SYMBOL, current_granularity_seconds)
    if API_TOKEN:
        await ws.send(json.dumps({"authorize": API_TOKEN}))
    else:
        await ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))

async def on_message_for_deriv(ws, message_str):
    global market_data, SYMBOL, current_tick_subscription_id, current_chart_type, current_granularity_seconds, in_progress_ohlcv_candle
    global pending_api_requests, api_response_events, api_response_data, shared_data_lock, market_data_lock
    
    # logging.debug(f"Async Raw WS message received: {message_str[:500]}")
    data = json.loads(message_str)

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
        tick_data = data.get('tick')
        if not tick_data or tick_data.get('epoch') is None or tick_data.get('quote') is None:
            logging.error(f"Invalid tick data: {data}")
            return
        
        tick_time_dt = datetime.fromtimestamp(tick_data['epoch'], timezone.utc)
        tick_price = float(tick_data['quote'])
        # tick_volume = float(tick_data.get('volume', 1.0)) # If volume per tick is available

        with market_data_lock:
            # Always update latest tick prices for non-OHLCV indicators if needed, or for current price display
            market_data['timestamps'].append(tick_time_dt.isoformat())
            market_data['prices'].append(tick_price)
            # Cap timestamps and prices for tick chart (e.g., last 200-300 points for performance)
            # This cap should be different from MAX_OHLCV_CANDLES
            max_tick_points = 300 
            if len(market_data['timestamps']) > max_tick_points:
                market_data['timestamps'] = market_data['timestamps'][-max_tick_points:]
                market_data['prices'] = market_data['prices'][-max_tick_points:]
            
            if 'volumes' in market_data: # This is dummy volume for tick chart
                 market_data['volumes'].append(np.random.randint(1000,5000))
                 if len(market_data['volumes']) > max_tick_points: market_data['volumes'] = market_data['volumes'][-max_tick_points:]

            indicator_input_data_for_calc = [] # This will be passed to calculate_indicators

            if current_chart_type == 'ohlcv':
                granularity_td = timedelta(seconds=current_granularity_seconds)
                # Calculate the start of the interval for the current tick
                interval_start_epoch = int(tick_data['epoch'] / current_granularity_seconds) * current_granularity_seconds
                interval_start_time_dt = datetime.fromtimestamp(interval_start_epoch, timezone.utc)
                interval_start_time_ms = int(interval_start_time_dt.timestamp() * 1000)

                if in_progress_ohlcv_candle is None or interval_start_time_ms > in_progress_ohlcv_candle['time']:
                    # New candle interval begins
                    if in_progress_ohlcv_candle is not None: # Finalize previous candle
                        logging.info(f"Finalizing candle for {in_progress_ohlcv_candle['time']} due to new interval starting at {interval_start_time_ms}.")
                        market_data['ohlcv_candles'].append(in_progress_ohlcv_candle)
                        # Sort and cap (though appending should keep it sorted if historical data was sorted)
                        market_data['ohlcv_candles'].sort(key=lambda c: c['time'])
                        if len(market_data['ohlcv_candles']) > MAX_OHLCV_CANDLES:
                            market_data['ohlcv_candles'] = market_data['ohlcv_candles'][-MAX_OHLCV_CANDLES:]
                    
                    logging.info(f"Starting new OHLCV candle for interval: {interval_start_time_dt} (granularity: {current_granularity_seconds}s)")
                    in_progress_ohlcv_candle = {
                        'time': interval_start_time_ms, 'open': tick_price, 'high': tick_price,
                        'low': tick_price, 'close': tick_price, 'volume': 1 # Using 1 as placeholder volume per tick
                    }
                elif interval_start_time_ms == in_progress_ohlcv_candle['time']:
                    # Tick is within the current candle interval
                    in_progress_ohlcv_candle['high'] = max(in_progress_ohlcv_candle['high'], tick_price)
                    in_progress_ohlcv_candle['low'] = min(in_progress_ohlcv_candle['low'], tick_price)
                    in_progress_ohlcv_candle['close'] = tick_price
                    in_progress_ohlcv_candle['volume'] += 1 
                    # logging.debug(f"Updating in-progress candle for {interval_start_time_dt}: C={tick_price}, H={in_progress_ohlcv_candle['high']}, L={in_progress_ohlcv_candle['low']}")
                
                # For indicators, use historical candles + the current in-progress one
                temp_ohlcv_list = list(market_data['ohlcv_candles'])
                if in_progress_ohlcv_candle:
                    temp_ohlcv_list.append(in_progress_ohlcv_candle) # Add current forming candle
                indicator_input_data_for_calc = temp_ohlcv_list
            
            else: # current_chart_type == 'tick'
                # Use recent tick prices for indicators
                indicator_input_data_for_calc = market_data['prices'][-max_tick_points:] # Use the capped tick prices

            # Calculate indicators based on the determined input data (now asynchronously)
            if not indicator_input_data_for_calc:
                logging.warning("No data (tick or ohlcv) available for async indicator calculation on new tick. Skipping.")
            else:
                loop = asyncio.get_running_loop()
                try:
                    # Offload the potentially blocking calculation to the thread pool
                    logging.debug(f"Offloading indicator calculation for {len(indicator_input_data_for_calc)} data points.")
                    updated_data = await loop.run_in_executor(
                        executor, # The global ThreadPoolExecutor instance
                        calculate_indicators, # The synchronous function
                        copy.deepcopy(indicator_input_data_for_calc) # Pass a copy to ensure thread safety of input
                    )
                    logging.debug("Async indicator calculation complete. Updating market data and cache.")

                    # Update market_data (this part should be quick)
                    with market_data_lock:
                        for key, value in updated_data.items():
                            if key in market_data: # Ensure key exists in DEFAULT_MARKET_DATA structure
                                if isinstance(value, list): market_data[key] = value
                                elif isinstance(value, str): market_data[key] = value
                    logging.debug("Market data updated with new async indicators from tick.")

                    # Update indicator_cache
                    cache_key_symbol = SYMBOL # Assuming SYMBOL is globally current for this connection
                    cache_key_granularity = 'tick' # For tick data
                    cache_key = f"{cache_key_symbol}_{cache_key_granularity}"
                    last_processed_ts = market_data['timestamps'][-1] if market_data['timestamps'] else None

                    with indicator_cache_lock:
                        indicator_cache[cache_key] = {
                            'indicators': copy.deepcopy(updated_data), # Store a copy
                            'last_processed_timestamp': last_processed_ts,
                            'updated_at': datetime.now(timezone.utc).isoformat()
                        }
                    logging.debug(f"Indicator cache updated for key: {cache_key}")

                except Exception as e:
                    logging.error(f"Error during async indicator calculation or update from tick: {e}", exc_info=True)

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
                        loop = asyncio.get_running_loop()
                        try:
                            logging.debug(f"Offloading indicator calculation for {len(parsed_ohlcv_candles)} new OHLCV candles.")
                            # Pass a deepcopy of parsed_ohlcv_candles to ensure thread safety for the input data
                            updated_data_from_ohlcv = await loop.run_in_executor(
                                executor,
                                calculate_indicators,
                                copy.deepcopy(parsed_ohlcv_candles)
                            )
                            logging.debug("Async indicator calculation for OHLCV candles complete. Updating market data and cache.")

                            with market_data_lock:
                                for key, value in updated_data_from_ohlcv.items():
                                    if key in market_data: # Ensure key exists in DEFAULT_MARKET_DATA
                                        if isinstance(value, list): market_data[key] = value
                                        elif isinstance(value, str): market_data[key] = value
                            logging.info(f"Market data updated with new async indicators from OHLCV (req_id: {req_id_of_message}, type: {request_type_str}).")

                            # Update indicator_cache for OHLCV
                            cache_key_symbol = SYMBOL # Assuming SYMBOL is current
                            cache_key_granularity = current_granularity_seconds # current_granularity_seconds from global
                            cache_key = f"{cache_key_symbol}_{cache_key_granularity}"
                            last_candle_timestamp = parsed_ohlcv_candles[-1]['time'] if parsed_ohlcv_candles else None

                            with indicator_cache_lock:
                                indicator_cache[cache_key] = {
                                    'indicators': copy.deepcopy(updated_data_from_ohlcv),
                                    'last_processed_timestamp': last_candle_timestamp, # Store ms timestamp
                                    'updated_at': datetime.now(timezone.utc).isoformat()
                                }
                            logging.debug(f"Indicator cache updated for OHLCV key: {cache_key}")

                        except Exception as e:
                             logging.error(f"Error during async indicator calculation or update from OHLCV candles: {e}", exc_info=True)
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
    # API logical errors are in on_message. This handles connection level errors for the new library.
    logging.error(f"[Async Deriv WS] Error: {error}")
    # Potentially try to reconnect or signal closure. For now, just log.
    # If this is called, recv_loop will likely exit.

async def on_close_for_deriv(ws, code, reason):
    logging.warning(f"Async WebSocket connection closed. Code: {code}, Reason: {reason}. Current SYMBOL: {SYMBOL}")
    # Reset ws_app to None to indicate connection is closed
    global ws_app
    if ws_app == ws: # Ensure it's the current connection closing
        ws_app = None

async def recv_loop(ws):
    """Continuously receive messages and handle them."""
    try:
        async for message_str in ws:
            await on_message_for_deriv(ws, message_str)
    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"Async WebSocket connection closed OK (recv_loop).")
        await on_close_for_deriv(ws, ws.close_code, "ConnectionClosedOK")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"Async WebSocket connection closed with error (recv_loop): {e}")
        await on_close_for_deriv(ws, e.code, e.reason)
        # Consider calling on_error_for_deriv as well, or instead, depending on desired handling
        # await on_error_for_deriv(ws, e)
    except Exception as e:
        logging.exception(f"Async WebSocket unexpected error in recv_loop: {e}")
        # This is a more general error, might need specific handling or calling on_error_for_deriv
        await on_error_for_deriv(ws, e)
        if ws.open: # If still open despite error, try to close gracefully
            await ws.close(code=1011) # Internal Error
        await on_close_for_deriv(ws, 1011, str(e))


def run_asyncio_loop_in_thread():
    global asyncio_loop
    asyncio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(asyncio_loop)
    logging.info("Asyncio event loop started in dedicated thread.")
    asyncio_loop.run_forever()
    logging.info("Asyncio event loop in dedicated thread has stopped.")


async def _connect_and_listen_deriv():
    """Manages the WebSocket connection lifecycle, including reconnections with exponential backoff."""
    global ws_app, current_reconnect_delay, current_tick_subscription_id, market_data, in_progress_ohlcv_candle

    while True: # Outer loop for reconnections
        active_connection_ws = None # Temporary ws variable for the current connection attempt
        try:
            logging.info(f"Attempting to connect to Deriv WebSocket: {DERIV_WS_URL} (Delay: {current_reconnect_delay:.2f}s)")
            async with websockets.connect(
                DERIV_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=10 # Timeout for initial connection handshake
            ) as ws:
                active_connection_ws = ws # Assign to temp var for this scope
                logging.info("Successfully connected to Deriv WebSocket.")
                current_reconnect_delay = INITIAL_RECONNECT_DELAY # Reset delay on successful connection
                ws_app = active_connection_ws # Set global ws_app ONLY after successful connection

                await on_open_for_deriv(active_connection_ws)
                await recv_loop(active_connection_ws) # This will block until connection is closed

        except websockets.exceptions.InvalidURI:
            logging.error(f"Invalid Deriv WebSocket URI: {DERIV_WS_URL}. Halting reconnection attempts.")
            ws_app = None # Ensure global ws_app is None
            break # Critical error, stop trying to reconnect
        except (websockets.exceptions.WebSocketException, ConnectionRefusedError, asyncio.TimeoutError) as e:
            # Specific exceptions related to connection attempts or established connections.
            # WebSocketException is broad and covers ConnectionClosed, ProtocolError etc.
            # ConnectionRefusedError for when server actively refuses.
            # asyncio.TimeoutError for open_timeout from websockets.connect.
            logging.warning(f"WebSocket connection attempt failed or connection lost: {e} (Type: {type(e)})")
            # Note: recv_loop calls on_close_for_deriv for established connections that drop.
            # If the connection was never established (e.g. ConnectionRefusedError, TimeoutError on connect),
            # on_open_for_deriv and recv_loop wouldn't have run.
        except Exception as e: # Catch any other unexpected errors
            logging.error(f"Unexpected error in WebSocket connect/listen loop: {e}", exc_info=True)
        finally:
            # This block executes after each connection attempt, whether successful then closed, or failed.
            if ws_app == active_connection_ws: # If global ws_app was set to this connection
                ws_app = None # Nullify global ws_app as this connection instance is now closed/failed.
                logging.info("Global ws_app cleared due to connection closure/failure.")

            # If recv_loop was entered, it should have called on_close_for_deriv upon exit.
            # If the connection failed before recv_loop (e.g., during websockets.connect),
            # on_close_for_deriv for *this specific connection instance* might not be meaningful
            # as it was never fully "opened" in our application's view.
            # However, we always proceed to backoff and retry unless it's a critical error like InvalidURI.

            logging.info(f"Waiting {current_reconnect_delay:.2f} seconds before next reconnection attempt.")
            await asyncio.sleep(current_reconnect_delay)
            current_reconnect_delay = min(current_reconnect_delay * RECONNECT_FACTOR, MAX_RECONNECT_DELAY)
        # End of try-except-finally for a single connection attempt
    # End of while True loop (reconnection loop)
    logging.critical("_connect_and_listen_deriv loop has exited. This should only happen on critical unrecoverable errors.")


def start_deriv_ws(token=None):
    global API_TOKEN, market_data, current_tick_subscription_id, in_progress_ohlcv_candle, asyncio_loop, current_ws_task, ws_app, current_reconnect_delay
    
    if token:
        API_TOKEN = token
        logging.info(f"API Token set for Deriv WS connection.")

    if asyncio_loop is None:
        logging.error("Asyncio loop not available. Cannot start WebSocket.")
        return

    # If there's an existing WebSocket task, cancel it
    if current_ws_task and not current_ws_task.done():
        logging.info("Existing WebSocket task found. Attempting to cancel and close connection.")
        current_ws_task.cancel()
        # Waiting for task cancellation can be done here if needed, but it might block.
        # For now, let's assume cancellation is enough and new task will take over.
        # Alternatively, could `await asyncio.gather(current_ws_task, return_exceptions=True)` but from sync context.
        # For simplicity, just cancel. The old task's finally block should clean up ws_app.
    
    # Reset market data and related state (as done previously)
    logging.info(f"Resetting market data and state for new/restarted WebSocket connection for SYMBOL: {SYMBOL}.")
    with market_data_lock:
        market_data.clear()
        market_data.update(copy.deepcopy(DEFAULT_MARKET_DATA))
        in_progress_ohlcv_candle = None
    current_tick_subscription_id = None
    ws_app = None # Explicitly clear global ws_app before new connection attempt

    logging.info(f"Creating new asyncio task for Deriv WebSocket connection for SYMBOL: {SYMBOL}.")
    # Schedule the _connect_and_listen_deriv coroutine to run on the asyncio_loop
    current_ws_task = asyncio.run_coroutine_threadsafe(_connect_and_listen_deriv(), asyncio_loop).result(timeout=0.1) # Check if task creation is successful
    # Note: .result(timeout=0.1) on run_coroutine_threadsafe is just to quickly check if scheduling itself failed,
    # not for the completion of _connect_and_listen_deriv().
    # The task runs in the background.

    logging.info("Deriv WebSocket connection process initiated via asyncio task.")


@app.route('/api/market_data')
def get_market_data_endpoint():
    global current_chart_type, current_granularity_seconds, TIMEFRAME_TO_SECONDS, market_data, in_progress_ohlcv_candle, market_data_lock

    with market_data_lock: 
        # Create a deepcopy to avoid issues if market_data is modified while this runs
        data_to_send = copy.deepcopy(market_data)
        
        # If OHLCV chart is active and there's an in-progress candle, add it to the list for the frontend
        if current_chart_type == 'ohlcv' and in_progress_ohlcv_candle:
            # Ensure ohlcv_candles is a list before trying to append
            if not isinstance(data_to_send.get('ohlcv_candles'), list):
                data_to_send['ohlcv_candles'] = [] # Initialize if not already a list
            
            # Append a copy of the in-progress candle so modifications don't affect this snapshot
            data_to_send['ohlcv_candles'].append(copy.deepcopy(in_progress_ohlcv_candle))
            # Optional: Re-sort if appending could mess order, though it should be the latest
            # data_to_send['ohlcv_candles'].sort(key=lambda c: c['time'])
            # Optional: Re-cap if adding the in-progress candle could exceed a display limit (different from storage MAX_OHLCV_CANDLES)
            # For now, assume frontend can handle one extra candle for display.

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
    global current_chart_type, current_granularity_seconds, ws_app, SYMBOL, market_data_lock, market_data, in_progress_ohlcv_candle, indicator_cache_lock, indicator_cache

    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    new_chart_type = data.get('chart_type', current_chart_type)
    new_timeframe_str = data.get('timeframe_str', None)
    
    chart_type_changed = False
    granularity_changed = False

    if new_chart_type != current_chart_type:
        if new_chart_type in ['tick', 'ohlcv']:
            with market_data_lock: # Lock for modifying current_chart_type and in_progress_ohlcv_candle
                current_chart_type = new_chart_type
                if current_chart_type != 'ohlcv': # Reset if switching away from OHLCV
                    in_progress_ohlcv_candle = None
                    logging.info("Switched chart type away from OHLCV, cleared in-progress candle.")
            chart_type_changed = True
            logging.info(f"Chart type changed to: {current_chart_type}")
        else:
            logging.warning(f"Invalid chart_type received: {new_chart_type}")
    
    if new_timeframe_str:
        new_granularity_seconds = TIMEFRAME_TO_SECONDS.get(new_timeframe_str.upper())
        if new_granularity_seconds:
            if new_granularity_seconds != current_granularity_seconds:
                with market_data_lock: # Lock for modifying granularity and in_progress_ohlcv_candle
                    current_granularity_seconds = new_granularity_seconds
                    in_progress_ohlcv_candle = None # Granularity change invalidates current forming candle
                    logging.info("Granularity changed, cleared in-progress candle.")
                granularity_changed = True
                logging.info(f"Chart granularity changed to: {current_granularity_seconds}s (from {new_timeframe_str})")
        else:
            logging.warning(f"Invalid timeframe_str received: {new_timeframe_str}")

    if chart_type_changed or granularity_changed:
        logging.info(f"Chart settings updated. ChartType: {current_chart_type}, Granularity: {current_granularity_seconds}s.")
        
        # Clear relevant historical data and request fresh historical data if needed
        with market_data_lock:
            market_data['ohlcv_candles'].clear() # Clear old candles on any significant change
            if current_chart_type == 'tick': # For tick chart, also clear price/timestamp arrays
                 market_data['prices'].clear() 
                 market_data['timestamps'].clear()
            # Reset all indicator arrays as they depend on the price data type and granularity
            for key in DEFAULT_MARKET_DATA.keys():
                if key not in ['ohlcv_candles', 'prices', 'timestamps', 'volumes', # Keep basic structures
                               'high_24h', 'low_24h', 'volume_24h'] and isinstance(market_data.get(key), list):
                    market_data[key].clear()
            logging.info("Cleared historical OHLCV, potentially tick data, and all indicator arrays due to settings change.")

            # Clear relevant parts of indicator_cache or the whole cache
            with indicator_cache_lock:
                # For simplicity, clear the whole cache. More granular clearing could be done.
                indicator_cache.clear()
                logging.info("Indicator cache cleared due to chart settings change.")

        # Re-request data using async functions (needs to be called from sync context if ws_app.send is used directly)
        # The current request_ohlcv_data and request_daily_data are synchronous wrappers around ws_app.send
        # which internally use run_coroutine_threadsafe. This part needs careful review if these
        # request functions are to be called directly here.
        # For now, assuming these calls will trigger the async send correctly.
        # The `start_deriv_ws` or other mechanisms usually handle data fetching on symbol/granularity change.
        # Let's verify if these request_... functions are async or sync wrappers.
        # They were made async def: request_ohlcv_data_async, request_daily_data_async
        # So, they cannot be directly called from this synchronous Flask handler.
        # This part of logic might need to be shifted to the on_open after a reconnect,
        # or use run_coroutine_threadsafe here if immediate re-request is desired.

        # For now, let's assume that a symbol/granularity change handled by start_deriv_ws()
        # will correctly call the async data request functions.
        # If ws_app is connected and we change settings, we might want to trigger these.
        # This is tricky from a sync context. The existing logic for data fetching on change
        # might be tied to start_deriv_ws.
        # The current `set_symbol` calls `start_deriv_ws` which handles this.
        # `set_chart_settings_endpoint` might need to do something similar if it needs to force a data refresh
        # beyond just clearing local data.
        # For now, we have cleared the cache. The next data arrival will repopulate.
        # If an immediate refresh is needed, it would require calling an async function from sync context:
        if ws_app and ws_app.open:
            if current_chart_type == 'ohlcv':
                logging.info(f"Requesting new historical OHLCV data (async from sync) for {SYMBOL}, Granularity={current_granularity_seconds}s.")
                asyncio.run_coroutine_threadsafe(request_ohlcv_data_async(ws_app, SYMBOL, current_granularity_seconds), asyncio_loop)
            asyncio.run_coroutine_threadsafe(request_daily_data_async(ws_app, SYMBOL), asyncio_loop)
        else:
            logging.warning("WebSocket not connected. Cannot fetch new data on settings change from set_chart_settings.")

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
    global SYMBOL, market_data, ws_app, API_TOKEN, current_tick_subscription_id, current_granularity_seconds, indicator_cache, indicator_cache_lock
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

    if not (ws_app and ws_app.open): # Check ws_app.open for websockets library
        with market_data_lock:
            logging.info(f"[/api/set_symbol] WebSocket not connected. Clearing market data for new symbol {SYMBOL} locally.")
            market_data.clear()
            market_data.update(copy.deepcopy(DEFAULT_MARKET_DATA))
            logging.info(f"[/api/set_symbol] Market data cleared and re-initialized locally.")
        with indicator_cache_lock:
            indicator_cache.clear() # Clear indicator cache on symbol change
            logging.info(f"[/api/set_symbol] Indicator cache cleared due to symbol change (WS not connected).")

    # start_deriv_ws handles data clearing and re-requests when symbol changes and WS is active
    logging.info(f"[/api/set_symbol] Calling start_deriv_ws for new symbol {SYMBOL}.")
    start_deriv_ws(API_TOKEN) # This will handle stopping old task, clearing data, and starting new.
                             # It also clears market_data and should ideally clear indicator_cache too.
                             # Let's ensure start_deriv_ws clears indicator_cache.

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
    global next_req_id, pending_api_requests, ws_app, shared_data_lock, asyncio_loop, deriv_api_request_semaphore

    if asyncio_loop is None or not asyncio_loop.is_running():
        logging.error("Asyncio loop is not running. Cannot send WebSocket request.")
        return None, "Asyncio loop not available."

    if ws_app is None or not ws_app.open:
        logging.error(f"WebSocket not connected or not open. Cannot send {payload_type} request.")
        return None, "WebSocket not connected."

    if not API_TOKEN and payload_type not in ['authorize_check']:
        logging.error(f"API TOKEN not set. Cannot send {payload_type} request that requires authorization.")
        return None, "API token not set."

    # --- Semaphore Acquisition ---
    logging.debug(f"Attempting to acquire Deriv API semaphore for {payload_type}...")
    
    async def _acquire_semaphore_async_helper(): # Helper coroutine
        await deriv_api_request_semaphore.acquire()

    acquire_future = asyncio.run_coroutine_threadsafe(_acquire_semaphore_async_helper(), asyncio_loop)
    try:
        acquire_future.result(timeout=timeout_seconds + 1) # Timeout slightly longer than request timeout
        logging.debug(f"Deriv API semaphore acquired for {payload_type}.")
    except asyncio.TimeoutError:
        logging.warning(f"Timeout acquiring Deriv API semaphore for {payload_type}.")
        return None, f"Timeout acquiring API semaphore for {payload_type}."
    except Exception as e:
        logging.error(f"Error acquiring Deriv API semaphore for {payload_type}: {e}", exc_info=True)
        return None, f"Error acquiring API semaphore for {payload_type}: {str(e)}"

    # --- Main Request Logic (after acquiring semaphore) ---
    try:
        response_event = threading.Event()
        request_entry = {'type': payload_type, 'event': response_event, 'data': None, 'error': None}

        with shared_data_lock:
            current_req_id = next_req_id
            next_req_id += 1
            payload['req_id'] = current_req_id
            pending_api_requests[current_req_id] = request_entry

        logging.info(f"Preparing to send {payload_type} request (req_id: {current_req_id}): {json.dumps(payload)}")

        async def do_send(ws_connection, payload_str):
            await ws_connection.send(payload_str)

        try:
            send_future = asyncio.run_coroutine_threadsafe(do_send(ws_app, json.dumps(payload)), asyncio_loop)
            send_future.result(timeout=5)
            logging.info(f"Successfully sent {payload_type} request (req_id: {current_req_id}) to WebSocket.")
        except Exception as e:
            logging.exception(f"Error sending {payload_type} request (req_id: {current_req_id}) via WebSocket: {e}")
            with shared_data_lock:
                pending_api_requests.pop(current_req_id, None)
            return None, f"Failed to send {payload_type} request: {str(e)}"

        if response_event.wait(timeout=timeout_seconds):
            with shared_data_lock:
                final_request_details = pending_api_requests.pop(current_req_id, request_entry)
            if final_request_details.get('error'):
                logging.error(f"{payload_type} request (req_id: {current_req_id}) failed: {final_request_details['error']}")
                return None, final_request_details['error']
            elif final_request_details.get('data') is not None:
                logging.info(f"{payload_type} request (req_id: {current_req_id}) successful.")
                return final_request_details['data'], None
            else:
                logging.warning(f"{payload_type} request (req_id: {current_req_id}) event set but no data/error found.")
                return None, "Response event set but no data or error recorded."
        else: # Timeout waiting for response_event
            logging.warning(f"{payload_type} request (req_id: {current_req_id}) timed out after {timeout_seconds}s waiting for Deriv response.")
            with shared_data_lock:
                pending_api_requests.pop(current_req_id, None)
            return None, f"{payload_type} request timed out."

    finally:
        # --- Semaphore Release ---
        # The release method of asyncio.Semaphore is thread-safe and can be called from any thread.
        # However, it's cleaner to schedule it on the loop if it was acquired by a task on that loop.
        # Since acquire was bridged, let's bridge release too for consistency.
        async def _do_release_semaphore_async():
            deriv_api_request_semaphore.release()
            logging.debug(f"Deriv API semaphore actually released by async helper for {payload_type}.")

        if asyncio_loop and asyncio_loop.is_running():
             asyncio.run_coroutine_threadsafe(_do_release_semaphore_async(), asyncio_loop)
             logging.debug(f"Deriv API semaphore release scheduled for {payload_type}.")
        else:
            logging.error(f"Asyncio loop not running, cannot schedule semaphore release for {payload_type}. Potential semaphore leak if this occurs often.")
            # Fallback: Try direct release if loop is gone (might raise error if loop is different from acquire)
            # deriv_api_request_semaphore.release() # This might be problematic depending on Semaphore impl details across threads without a loop.
            # Better to ensure loop is always running or handle this state more gracefully.

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


# --- Strategy Management API Endpoints ---
@app.route('/api/strategies', methods=['POST'], strict_slashes=False)
def create_strategy():
    global custom_strategies
    strategy_data = request.get_json()

    if not isinstance(strategy_data, dict) or not strategy_data.get('strategy_name') or not strategy_data.get('conditions_group'):
        logging.warning(f"Bad request for creating strategy: {strategy_data}")
        return jsonify({"error": "Invalid strategy payload. 'strategy_name' and 'conditions_group' are required."}), 400

    # Check if strategy_id is provided and if it already exists in the DB (more robust than just checking memory)
    # However, for a POST, we usually generate the ID.
    # If strategy_id is in payload, it's more like an "upsert" or client-managed ID.
    # For this implementation, we'll generate a new ID if one is not provided.
    # If an ID is provided, we'll respect it but the save_strategy_to_db will fail if it already exists (due to PRIMARY KEY constraint on strategies.strategy_id)
    # unless save_strategy_to_db is designed to handle that (e.g. with INSERT OR REPLACE, or checking existence first).
    # Current save_strategy_to_db will fail on INSERT if ID exists. This is acceptable for POST.

    strategy_id = strategy_data.get('strategy_id')
    if not strategy_id: # If no ID provided, generate one
        strategy_id = uuid.uuid4().hex
        strategy_data['strategy_id'] = strategy_id # Add to payload for saving
    else: # ID provided, check if it exists in memory (quick check, DB is the source of truth)
        with strategies_lock:
            if strategy_id in custom_strategies:
                logging.warning(f"Attempt to create strategy with existing ID (from payload): {strategy_id}")
                return jsonify({"error": f"Strategy with ID {strategy_id} already exists. Use PUT to update or omit ID for new."}), 409

    # Prepare the full strategy object for saving (includes all fields for tables)
    # `save_strategy_to_db` will use current time for created_at/updated_at for the strategy record
    # and created_at for the version record.
    
    # Ensure all required fields for DB are present before calling save_strategy_to_db
    full_strategy_payload_for_db = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_data["strategy_name"],
        "description": strategy_data.get("description", ""),
        "conditions_group": strategy_data["conditions_group"], # Must exist
        "actions": strategy_data.get("actions", []),
        "is_active": strategy_data.get("is_active", True),
        "version_notes": strategy_data.get("version_notes", f"Initial version created on {datetime.now(timezone.utc).isoformat()}") # Example default notes
    }

    if save_strategy_to_db(full_strategy_payload_for_db, is_update=False):
        # After successful save, update in-memory cache
        # Reconstruct what the strategy would look like if loaded from DB
        # Or, more simply, just reload all strategies to ensure consistency.
        # For now, let's construct it to return, and then client might re-fetch or we rely on next full load.
        # A better approach: save_strategy_to_db could return the created object, or load_strategies_from_db can be called.
        
        # Re-create the object as it would be loaded to be returned in response
        created_strategy_for_response = {
            "strategy_id": strategy_id,
            "strategy_name": full_strategy_payload_for_db["strategy_name"],
            "description": full_strategy_payload_for_db["description"],
            "conditions_group": full_strategy_payload_for_db["conditions_group"],
            "actions": full_strategy_payload_for_db["actions"],
            "is_active": full_strategy_payload_for_db["is_active"],
            "created_at": datetime.now(timezone.utc).isoformat(), # This will be set by DB, approximate here
            "updated_at": datetime.now(timezone.utc).isoformat()  # This will be set by DB, approximate here
        }
        # Update in-memory cache (important!)
        with strategies_lock:
            custom_strategies[strategy_id] = created_strategy_for_response # Add to cache

        logging.info(f"Created strategy via API with ID: {strategy_id}, Name: {full_strategy_payload_for_db['strategy_name']}")
        return jsonify(created_strategy_for_response), 201
    else:
        # This case implies a DB error during save_strategy_to_db
        # It could be due to various reasons, e.g. strategy_id (if client-provided) already exists.
        # The save_strategy_to_db function logs the specific DB error.
        logging.error(f"Failed to save new strategy (ID: {strategy_id}) to database.")
        # Check if ID was client provided and might be a duplicate
        if strategy_data.get('strategy_id'): # If client provided ID
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT strategy_id FROM strategies WHERE strategy_id = ?", (strategy_id,))
            if cursor.fetchone():
                conn.close()
                return jsonify({"error": f"Strategy with ID {strategy_id} already exists in DB. Use PUT to update or ensure ID is unique."}), 409 # Conflict
            conn.close()

        return jsonify({"error": "Failed to create strategy due to a database issue."}), 500


@app.route('/api/strategies', methods=['GET'], strict_slashes=False)
def get_all_strategies():
    # This endpoint should reflect the current state from the in-memory cache,
    # which is loaded from DB at startup and potentially updated after CUD operations.
    with strategies_lock:
        # Return a list of strategy objects, not the dict itself
        return jsonify(list(custom_strategies.values()))

@app.route('/api/strategies/<strategy_id>', methods=['GET'], strict_slashes=False)
def get_strategy_by_id(strategy_id):
    with strategies_lock: # Access the in-memory cache
        strategy = custom_strategies.get(strategy_id)
    if strategy:
        # Ensure the strategy is not marked as deleted in DB, though load_strategies_from_db should handle this.
        # This is a redundant check if cache is always consistent.
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT is_deleted FROM strategies WHERE strategy_id = ?", (strategy_id,))
        db_strategy_status = cursor.fetchone()
        conn.close()
        if db_strategy_status and db_strategy_status['is_deleted']:
            # Strategy is marked deleted in DB, so remove from cache if it's there by mistake
            with strategies_lock:
                if strategy_id in custom_strategies: custom_strategies.pop(strategy_id)
            logging.warning(f"Strategy with ID {strategy_id} found in cache but is deleted in DB. Cache corrected.")
            return jsonify({"error": "Strategy not found (marked as deleted)"}), 404

        return jsonify(strategy)
    else:
        logging.warning(f"Strategy with ID {strategy_id} not found in cache (GET).")
        return jsonify({"error": "Strategy not found"}), 404

@app.route('/api/strategies/<strategy_id>', methods=['PUT'], strict_slashes=False)
def update_strategy(strategy_id):
    strategy_updates = request.get_json()

    if not isinstance(strategy_updates, dict):
        return jsonify({"error": "Invalid payload. Expected a JSON object."}), 400

    with strategies_lock: # Check existence in cache first
        if strategy_id not in custom_strategies:
            logging.warning(f"Strategy with ID {strategy_id} not found in cache (PUT).")
            # Optionally, verify against DB if cache might be stale
            conn_check = get_db_connection()
            cursor_check = conn_check.cursor()
            cursor_check.execute("SELECT strategy_id FROM strategies WHERE strategy_id = ? AND is_deleted = FALSE", (strategy_id,))
            if not cursor_check.fetchone():
                conn_check.close()
                return jsonify({"error": "Strategy not found in database"}), 404
            conn_check.close()
            # If found in DB but not cache, it implies cache inconsistency. Log it.
            logging.warning(f"Strategy {strategy_id} found in DB but not in cache. Cache might be inconsistent.")
            # Proceeding with update, but ideally cache should be consistent.
            # For robust solution, one might load the strategy here before updating.

    # Prepare data for saving. Merge updates with existing data (from cache, or load if not in cache but in DB).
    # The save_strategy_to_db function will handle creating a new version.
    # We need to provide the full strategy data for the new version.

    # Get current state from cache to merge
    current_strategy_state = None
    with strategies_lock:
        current_strategy_state = copy.deepcopy(custom_strategies.get(strategy_id)) # Use deepcopy if complex objects

    if not current_strategy_state: # If, after all, it's not in cache (e.g. race condition or inconsistency)
        # Attempt to load it directly from DB to ensure we have the latest before updating
        # This is a fallback; ideally, the cache check above or a lock around the entire operation would be better.
        load_strategies_from_db() # Reload all; simpler than loading one and risking more inconsistency
        with strategies_lock:
            current_strategy_state = copy.deepcopy(custom_strategies.get(strategy_id))
        if not current_strategy_state:
             logging.error(f"Strategy {strategy_id} could not be loaded for update even after DB refresh.")
             return jsonify({"error": "Strategy not found for update after DB refresh check."}), 404


    # Construct the full payload for save_strategy_to_db, applying updates
    updated_strategy_payload_for_db = {
        "strategy_id": strategy_id, # Must be the same
        "strategy_name": strategy_updates.get("strategy_name", current_strategy_state["strategy_name"]),
        "description": strategy_updates.get("description", current_strategy_state.get("description")), # Use .get for safety
        "conditions_group": strategy_updates.get("conditions_group", current_strategy_state["conditions_group"]),
        "actions": strategy_updates.get("actions", current_strategy_state.get("actions")),
        "is_active": strategy_updates.get("is_active", current_strategy_state.get("is_active")), # is_active can be updated here too
        "version_notes": strategy_updates.get("version_notes", f"Updated on {datetime.now(timezone.utc).isoformat()}")
    }

    if save_strategy_to_db(updated_strategy_payload_for_db, is_update=True):
        # Update in-memory cache with the new state (reflecting new updated_at, and potentially new name/desc from version)
        # The save_strategy_to_db updated `strategies` table's `updated_at`, `strategy_name`, `description`.
        # The versioned part (conditions, actions) is in `strategy_versions`.
        # `load_strategies_from_db` correctly reconstructs this. So, best to reload.
        
        load_strategies_from_db() # Reload all strategies to reflect the update and new version.
                                  # This ensures the cache is perfectly consistent with DB.
        
        with strategies_lock: # Get the reloaded strategy for the response
            final_updated_strategy_for_response = custom_strategies.get(strategy_id)

        if final_updated_strategy_for_response:
            logging.info(f"Updated strategy with ID: {strategy_id} in DB and reloaded cache.")
            return jsonify(final_updated_strategy_for_response)
        else:
            # This should not happen if save was successful and load_strategies_from_db works
            logging.error(f"Strategy {strategy_id} disappeared from cache after successful update and reload.")
            return jsonify({"error": "Failed to reflect update in cache, though DB operation might be successful."}), 500
    else:
        logging.error(f"Failed to update strategy {strategy_id} in database.")
        return jsonify({"error": "Failed to update strategy due to a database issue."}), 500

@app.route('/api/strategies/<strategy_id>', methods=['DELETE'], strict_slashes=False)
def delete_strategy(strategy_id):
    # This will be a soft delete.
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if strategy exists and is not already deleted
        cursor.execute("SELECT strategy_id, strategy_name, is_deleted FROM strategies WHERE strategy_id = ?", (strategy_id,))
        strategy_to_delete = cursor.fetchone()

        if not strategy_to_delete:
            conn.close()
            logging.warning(f"Strategy with ID {strategy_id} not found in DB (DELETE).")
            return jsonify({"error": "Strategy not found"}), 404
        
        if strategy_to_delete['is_deleted']:
            conn.close()
            logging.info(f"Strategy with ID {strategy_id} is already marked as deleted (DELETE).")
            # Remove from cache if it's still there
            with strategies_lock:
                if strategy_id in custom_strategies: custom_strategies.pop(strategy_id)
            return jsonify({"message": "Strategy was already deleted"}), 200

        # Perform soft delete
        current_time_utc_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            UPDATE strategies
            SET is_deleted = TRUE, is_active = FALSE, updated_at = ?
            WHERE strategy_id = ?
        """, (current_time_utc_iso, strategy_id))
        conn.commit()

        deleted_strategy_name = strategy_to_delete['strategy_name']
        logging.info(f"Soft-deleted strategy with ID: {strategy_id}, Name: {deleted_strategy_name}")

        # Remove from in-memory cache
        with strategies_lock:
            if strategy_id in custom_strategies:
                del custom_strategies[strategy_id]

        return jsonify({"message": f"Strategy '{deleted_strategy_name}' (ID: {strategy_id}) soft-deleted successfully"}), 200

    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Database error soft-deleting strategy ID {strategy_id}: {e}")
        return jsonify({"error": "Failed to delete strategy due to a database issue."}), 500
    finally:
        if conn: conn.close()


@app.route('/api/strategies/<strategy_id>/enable', methods=['POST'], strict_slashes=False)
def enable_strategy(strategy_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        current_time_utc_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute("UPDATE strategies SET is_active = TRUE, updated_at = ? WHERE strategy_id = ? AND is_deleted = FALSE", (current_time_utc_iso, strategy_id))
        updated_rows = cursor.rowcount
        conn.commit()

        if updated_rows > 0:
            # Update cache
            with strategies_lock:
                if strategy_id in custom_strategies:
                    custom_strategies[strategy_id]['is_active'] = True
                    custom_strategies[strategy_id]['updated_at'] = current_time_utc_iso
                    strategy_name_for_log = custom_strategies[strategy_id].get('strategy_name', 'N/A')
                    logging.info(f"Enabled strategy in DB & Cache: ID {strategy_id}, Name: {strategy_name_for_log}")
                    return jsonify(custom_strategies[strategy_id]), 200
                else: # DB updated but not in cache - inconsistency
                    load_strategies_from_db() # Reload to fix cache
                    if strategy_id in custom_strategies and custom_strategies[strategy_id]['is_active']:
                         logging.info(f"Enabled strategy in DB, reloaded cache. ID: {strategy_id}")
                         return jsonify(custom_strategies[strategy_id]), 200
                    else:
                         logging.error(f"Failed to enable strategy {strategy_id} or reflect in cache properly after DB update.")
                         return jsonify({"error": "Strategy state in cache unclear after enable."}), 500
        else:
            # Check if strategy exists at all or is deleted
            cursor.execute("SELECT strategy_id, is_deleted FROM strategies WHERE strategy_id = ?", (strategy_id,))
            exists = cursor.fetchone()
            if not exists or exists['is_deleted']:
                logging.warning(f"Strategy with ID {strategy_id} not found or is deleted (ENABLE).")
                return jsonify({"error": "Strategy not found or is deleted"}), 404
            else: # Exists, not deleted, but wasn't updated (maybe already active, or other issue)
                 logging.warning(f"Strategy {strategy_id} enable did not update any rows, might be already active or other issue.")
                 # To be safe, reload and return current state
                 load_strategies_from_db()
                 with strategies_lock:
                    if strategy_id in custom_strategies: return jsonify(custom_strategies[strategy_id]), 200
                    else: return jsonify({"error": "Strategy not found after attempting enable"}), 404


    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Database error enabling strategy {strategy_id}: {e}")
        return jsonify({"error": "Database error enabling strategy"}), 500
    finally:
        if conn: conn.close()


@app.route('/api/strategies/<strategy_id>/disable', methods=['POST'], strict_slashes=False)
def disable_strategy(strategy_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        current_time_utc_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute("UPDATE strategies SET is_active = FALSE, updated_at = ? WHERE strategy_id = ? AND is_deleted = FALSE", (current_time_utc_iso, strategy_id))
        updated_rows = cursor.rowcount
        conn.commit()

        if updated_rows > 0:
            # Update cache
            with strategies_lock:
                if strategy_id in custom_strategies:
                    custom_strategies[strategy_id]['is_active'] = False
                    custom_strategies[strategy_id]['updated_at'] = current_time_utc_iso
                    strategy_name_for_log = custom_strategies[strategy_id].get('strategy_name', 'N/A')
                    logging.info(f"Disabled strategy in DB & Cache: ID {strategy_id}, Name: {strategy_name_for_log}")
                    return jsonify(custom_strategies[strategy_id]), 200
                else: # DB updated but not in cache - inconsistency
                    load_strategies_from_db() # Reload to fix cache
                    if strategy_id in custom_strategies and not custom_strategies[strategy_id]['is_active']:
                         logging.info(f"Disabled strategy in DB, reloaded cache. ID: {strategy_id}")
                         return jsonify(custom_strategies[strategy_id]), 200
                    else:
                         logging.error(f"Failed to disable strategy {strategy_id} or reflect in cache properly after DB update.")
                         return jsonify({"error": "Strategy state in cache unclear after disable."}), 500
        else:
            # Check if strategy exists at all or is deleted
            cursor.execute("SELECT strategy_id, is_deleted FROM strategies WHERE strategy_id = ?", (strategy_id,))
            exists = cursor.fetchone()
            if not exists or exists['is_deleted']:
                logging.warning(f"Strategy with ID {strategy_id} not found or is deleted (DISABLE).")
                return jsonify({"error": "Strategy not found or is deleted"}), 404
            else: # Exists, not deleted, but wasn't updated (maybe already inactive)
                 logging.warning(f"Strategy {strategy_id} disable did not update any rows, might be already inactive or other issue.")
                 load_strategies_from_db()
                 with strategies_lock:
                    if strategy_id in custom_strategies: return jsonify(custom_strategies[strategy_id]), 200
                    else: return jsonify({"error": "Strategy not found after attempting disable"}), 404

    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Database error disabling strategy {strategy_id}: {e}")
        return jsonify({"error": "Database error disabling strategy"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/strategies/<strategy_id>/history', methods=['GET'], strict_slashes=False)
def get_strategy_history(strategy_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # First, check if the base strategy exists and is not deleted
        cursor.execute("SELECT strategy_id FROM strategies WHERE strategy_id = ? AND is_deleted = FALSE", (strategy_id,))
        if not cursor.fetchone():
            logging.warning(f"History requested for non-existent or deleted strategy ID: {strategy_id}")
            return jsonify({"error": "Strategy not found"}), 404

        cursor.execute("""
            SELECT version_id, strategy_id, strategy_name, description, conditions_group, actions, version_notes, created_at
            FROM strategy_versions
            WHERE strategy_id = ?
            ORDER BY version_id DESC
        """, (strategy_id,))

        versions_rows = cursor.fetchall()
        history = []
        for row in versions_rows:
            try:
                version_data = dict(row) # Convert SQLite Row to dict
                version_data['conditions_group'] = json.loads(row['conditions_group'])
                version_data['actions'] = json.loads(row['actions'])
                history.append(version_data)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON for strategy ID {strategy_id}, version ID {row['version_id']} in history: {e}")
                # Skip this version or add with raw data? For now, skip.
            except Exception as e:
                logging.error(f"Error processing version ID {row['version_id']} for strategy {strategy_id} in history: {e}")

        logging.info(f"Retrieved {len(history)} versions for strategy ID: {strategy_id}")
        return jsonify(history)

    except sqlite3.Error as e:
        logging.error(f"Database error retrieving history for strategy ID {strategy_id}: {e}")
        return jsonify({"error": "Database error retrieving strategy history."}), 500
    finally:
        if conn: conn.close()

@app.route('/api/strategies/<strategy_id>/rollback/<int:version_id>', methods=['POST'], strict_slashes=False)
def rollback_strategy_version(strategy_id, version_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. Validate base strategy
        cursor.execute("SELECT strategy_id, is_active FROM strategies WHERE strategy_id = ? AND is_deleted = FALSE", (strategy_id,))
        base_strategy = cursor.fetchone()
        if not base_strategy:
            conn.close()
            logging.warning(f"Rollback requested for non-existent or deleted strategy ID: {strategy_id}")
            return jsonify({"error": "Base strategy not found or has been deleted"}), 404

        # 2. Validate target version
        cursor.execute("""
            SELECT strategy_name, description, conditions_group, actions
            FROM strategy_versions
            WHERE strategy_id = ? AND version_id = ?
        """, (strategy_id, version_id))
        target_version_data = cursor.fetchone()

        if not target_version_data:
            conn.close()
            logging.warning(f"Target version ID {version_id} not found for strategy ID {strategy_id} (Rollback).")
            return jsonify({"error": f"Version ID {version_id} not found for strategy {strategy_id}"}), 404

        # 3. Perform Rollback (Update main strategy table, Create new version)
        current_time_utc_iso = datetime.now(timezone.utc).isoformat()

        # Update the main 'strategies' table entry
        # Preserve original is_active state from base_strategy['is_active']
        # Or, decide to always make it active on rollback: base_strategy_is_active = True
        base_strategy_is_active = bool(base_strategy['is_active'])

        logging.info(f"Rolling back strategy {strategy_id} to version {version_id}. Current is_active: {base_strategy_is_active}")

        cursor.execute("""
            UPDATE strategies
            SET strategy_name = ?, description = ?, updated_at = ?, is_active = ?
            WHERE strategy_id = ?
        """, (target_version_data['strategy_name'], target_version_data['description'], current_time_utc_iso, base_strategy_is_active, strategy_id))

        # Insert a new version entry in 'strategy_versions' reflecting this rollback
        new_version_notes = f"Rolled back to version {version_id} data on {current_time_utc_iso}."

        # The conditions_group and actions are already JSON strings in target_version_data if fetched directly from DB
        # No, they are Python dicts because row_factory = sqlite3.Row and then we might have dict access.
        # The 'save_strategy_to_db' function expects python dicts for conditions_group and actions.
        # Here, target_version_data['conditions_group'] and ['actions'] are already JSON strings from the DB.
        # So we need to parse them first, then they will be re-stringified by save_strategy_to_db logic, OR
        # we directly use them if the save_strategy_to_db is not used, which is the case here.
        # The save_strategy_to_db function itself handles the JSON stringification.
        # Let's prepare a payload similar to what save_strategy_to_db's versioning part expects.

        payload_for_new_version = {
            "strategy_id": strategy_id,
            "strategy_name": target_version_data['strategy_name'],
            "description": target_version_data['description'],
            "conditions_group": json.loads(target_version_data['conditions_group']), # Parse for save_strategy_to_db or direct insert
            "actions": json.loads(target_version_data['actions']), # Parse for save_strategy_to_db or direct insert
            "version_notes": new_version_notes,
            # is_active is part of the main strategy table, not typically versioned directly in strategy_versions in this way
        }

        # Re-using the version creation logic from save_strategy_to_db is better if possible.
        # For now, direct insert:
        cursor.execute("""
            INSERT INTO strategy_versions (strategy_id, strategy_name, description, conditions_group, actions, version_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_id,
            payload_for_new_version['strategy_name'],
            payload_for_new_version['description'],
            json.dumps(payload_for_new_version['conditions_group']), # Stringify here for direct insert
            json.dumps(payload_for_new_version['actions']), # Stringify here
            payload_for_new_version['version_notes'],
            current_time_utc_iso
        ))

        new_rolled_back_version_id = cursor.lastrowid
        conn.commit()

        logging.info(f"Strategy {strategy_id} rolled back to data from version {version_id}. New version created: {new_rolled_back_version_id}.")

        # 4. Update in-memory cache
        load_strategies_from_db() # Reload all strategies to reflect the change.

        with strategies_lock: # Get the reloaded strategy for response
            reloaded_strategy = custom_strategies.get(strategy_id)

        if reloaded_strategy:
             # Add the new version_id to the response for clarity
            reloaded_strategy['rolled_back_to_source_version_id'] = version_id
            reloaded_strategy['new_current_version_id'] = new_rolled_back_version_id
            return jsonify({
                "message": f"Strategy {strategy_id} successfully rolled back to version {version_id}.",
                "strategy_details": reloaded_strategy
            }), 200
        else: # Should not happen
            logging.error(f"Strategy {strategy_id} disappeared from cache after successful rollback and reload.")
            return jsonify({"error": "Rollback successful but failed to reload strategy into cache."}), 500

    except sqlite3.Error as e:
        if conn: conn.rollback()
        logging.error(f"Database error rolling back strategy ID {strategy_id} to version {version_id}: {e}")
        return jsonify({"error": "Database error during rollback."}), 500
    except json.JSONDecodeError as e:
        if conn: conn.rollback()
        logging.error(f"JSON error during rollback for strategy {strategy_id} to version {version_id}: {e}")
        return jsonify({"error": "Error processing strategy data during rollback."}), 500
    except Exception as e:
        if conn: conn.rollback()
        logging.exception(f"Unexpected error during rollback of strategy {strategy_id} to version {version_id}: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        if conn: conn.close()

start_deriv_ws()

if __name__ == '__main__':
    # Note: app.run(debug=True) is suitable for development.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # load_strategies_from_db() # Already called above after app initialization

    # Start the asyncio event loop in a separate thread
    ws_thread = threading.Thread(target=run_asyncio_loop_in_thread, daemon=True)
    ws_thread.start()

    # Start Deriv WebSocket connection (initial, without token if not yet provided)
    # Wait a moment for the loop to be ready.
    time.sleep(0.5) # Simple way to wait for loop; more robust would use an event.
    if asyncio_loop and asyncio_loop.is_running():
        start_deriv_ws()
    else:
        logging.error("Failed to start asyncio loop, WebSocket cannot be initialized.")

    app.run(debug=True)
