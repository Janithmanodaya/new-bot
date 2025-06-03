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

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

app = Flask(__name__)
CORS(app)

# --- Strategy Management Globals and Functions ---
STRATEGIES_FILE = "custom_strategies.json"
custom_strategies = {} # In-memory storage for strategies: {strategy_id: strategy_object}
strategies_lock = threading.Lock() # Lock for accessing custom_strategies and STRATEGIES_FILE

def load_strategies_from_file():
    global custom_strategies
    with strategies_lock:
        if os.path.exists(STRATEGIES_FILE):
            try:
                with open(STRATEGIES_FILE, 'r') as f:
                    custom_strategies = json.load(f)
                    logging.info(f"Loaded {len(custom_strategies)} strategies from {STRATEGIES_FILE}")
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {STRATEGIES_FILE}. Initializing with empty strategies.")
                custom_strategies = {}
            except Exception as e:
                logging.error(f"Error loading strategies from {STRATEGIES_FILE}: {e}. Initializing with empty strategies.")
                custom_strategies = {}
        else:
            logging.info(f"{STRATEGIES_FILE} not found. Initializing with empty strategies.")
            custom_strategies = {}

def save_strategies_to_file():
    with strategies_lock:
        try:
            with open(STRATEGIES_FILE, 'w') as f:
                json.dump(custom_strategies, f, indent=4)
                logging.info(f"Saved {len(custom_strategies)} strategies to {STRATEGIES_FILE}")
        except Exception as e:
            logging.error(f"Error saving strategies to {STRATEGIES_FILE}: {e}")

# Load strategies at startup
load_strategies_from_file()
# --- End Strategy Management ---

# --- Strategy Evaluation Helper Functions ---
def get_indicator_values(indicator_name, market_state):
    """
    Safely retrieves current and previous values for a given indicator from market_state.
    Returns (current_value, previous_value) or (None, None) if not available.
    For 'PRICE', it checks current_chart_type to decide between tick prices or candle close.
    """
    indicator_key_map = {
        'RSI': 'rsi',
        'MACD': 'macd', # This is the MACD line. Signal line comparison is more complex.
        'STOCH': 'stochastic', # Stochastic %K
        'CCI': 'cci_20', # Assuming CCI20 specifically
        'EMA10': 'ema_10',
        'EMA20': 'ema_20',
        'EMA30': 'ema_30',
        'EMA50': 'ema_50',
        'EMA100': 'ema_100',
        'EMA200': 'ema_200',
        # Add other SMAs if needed, e.g., 'SMA50': 'sma_50'
    }

    current_value = None
    previous_value = None

    if indicator_name == 'PRICE':
        source_data = None
        if market_state.get('current_chart_type') == 'ohlcv' and market_state.get('ohlcv_candles'):
            source_data = [c['close'] for c in market_state['ohlcv_candles'] if isinstance(c, dict) and 'close' in c]
        elif market_state.get('current_chart_type') == 'tick' and market_state.get('prices'):
            source_data = market_state['prices']

        if source_data and len(source_data) > 0:
            current_value = source_data[-1]
        if source_data and len(source_data) > 1:
            previous_value = source_data[-2]

    elif indicator_name in indicator_key_map:
        key = indicator_key_map[indicator_name]
        if market_state.get(key) and isinstance(market_state[key], list) and len(market_state[key]) > 0:
            current_value = market_state[key][-1]
        if market_state.get(key) and isinstance(market_state[key], list) and len(market_state[key]) > 1:
            previous_value = market_state[key][-2]
    else:
        logging.warning(f"[ConditionCheck] Unknown indicator specified: {indicator_name}")
        return None, None

    # Ensure values are floats if not None
    try:
        current_value = float(current_value) if current_value is not None else None
        previous_value = float(previous_value) if previous_value is not None else None
    except (ValueError, TypeError):
        logging.error(f"[ConditionCheck] Could not convert indicator values to float for {indicator_name}. Current: {current_value}, Prev: {previous_value}")
        return None, None

    return current_value, previous_value

def check_single_condition(condition_item, market_state):
    """
    Evaluates a single strategy condition.
    condition_item (dict): e.g., {"indicator": "RSI", "operator": "<", "value": "30"}
                            or {"indicator": "PRICE", "operator": "CROSS_ABOVE", "value": "EMA50"}
    market_state (dict): A snapshot of the current market_data.
    Returns: True if condition is met, False otherwise.
    """
    indicator_name = condition_item.get('indicator')
    operator = condition_item.get('operator')
    target_value_str = str(condition_item.get('value', '')).strip() # Ensure it's a string and stripped

    if not all([indicator_name, operator, target_value_str]):
        logging.warning(f"[ConditionCheck] Invalid condition item (missing fields): {condition_item}")
        return False

    current_lhs, prev_lhs = get_indicator_values(indicator_name, market_state)

    if current_lhs is None:
        logging.debug(f"[ConditionCheck] LHS current value for '{indicator_name}' not available or not numeric. Condition false.")
        return False

    # Determine RHS value: either a numeric literal or another indicator's value
    current_rhs, prev_rhs = None, None
    # Check against uppercase for robust matching of indicator names
    is_rhs_indicator = target_value_str.upper() in [k.upper() for k in get_indicator_values.__globals__['indicator_key_map'].keys()] \
                       or target_value_str.upper() == 'PRICE'


    if is_rhs_indicator:
        current_rhs, prev_rhs = get_indicator_values(target_value_str.upper(), market_state)
        if current_rhs is None:
            logging.debug(f"[ConditionCheck] RHS indicator '{target_value_str}' value not available. Condition false.")
            return False
    else: # RHS is a literal value
        try:
            current_rhs = float(target_value_str)
            # prev_rhs remains None as it's not needed for literal comparisons unless operator implies it (not common)
        except ValueError:
            logging.warning(f"[ConditionCheck] Invalid numeric target value in condition: {target_value_str}")
            return False

    logging.debug(f"[ConditionCheck] Evaluating: {indicator_name} ({current_lhs}, prev:{prev_lhs}) {operator} {target_value_str} (RHS current:{current_rhs}, prev:{prev_rhs})")

    # Simple comparisons (evaluate current_lhs against current_rhs)
    if operator == '<':
        return current_lhs < current_rhs
    elif operator == '<=':
        return current_lhs <= current_rhs
    elif operator == '==':
        # Floating point comparisons need tolerance, but for now direct.
        # Consider np.isclose for more robust float equality if needed.
        return current_lhs == current_rhs
    elif operator == '>=':
        return current_lhs >= current_rhs
    elif operator == '>':
        return current_lhs > current_rhs

    # Crossover comparisons (require previous values for both LHS and RHS)
    elif operator == 'CROSS_ABOVE':
        # For RHS literal, prev_rhs effectively becomes current_rhs
        if not is_rhs_indicator:
            prev_rhs = current_rhs

        if prev_lhs is None or prev_rhs is None:
            if is_rhs_indicator and prev_rhs is None:
                logging.debug(f"[ConditionCheck] CROSS_ABOVE: Previous RHS value for '{target_value_str}' not available.")
            elif prev_lhs is None:
                logging.debug(f"[ConditionCheck] CROSS_ABOVE: Previous LHS value for '{indicator_name}' not available.")
            # If RHS is literal, prev_rhs is already set to current_rhs, so this path for prev_rhs being None won't be hit unless current_rhs was None (handled earlier)
            return False

        # Condition: prev_lhs <= prev_rhs AND current_lhs > current_rhs
        return prev_lhs <= prev_rhs and current_lhs > current_rhs

    elif operator == 'CROSS_BELOW':
        if not is_rhs_indicator:
            prev_rhs = current_rhs

        if prev_lhs is None or prev_rhs is None:
            if is_rhs_indicator and prev_rhs is None:
                logging.debug(f"[ConditionCheck] CROSS_BELOW: Previous RHS value for '{target_value_str}' not available.")
            elif prev_lhs is None:
                logging.debug(f"[ConditionCheck] CROSS_BELOW: Previous LHS value for '{indicator_name}' not available.")
            return False

        # Condition: prev_lhs >= prev_rhs AND current_lhs < current_rhs
        return prev_lhs >= prev_rhs and current_lhs < current_rhs

    else:
        logging.warning(f"[ConditionCheck] Unknown operator in condition: {operator}")
        return False

def trigger_strategy_actions(strategy, current_market_state):
    """
    Executes the actions defined in a strategy.
    strategy (dict): The strategy object whose conditions were met.
    current_market_state (dict): The market data snapshot at the time of condition evaluation.
                                  Used to fetch current price for simulated trades.
    """
    strategy_name = strategy.get('strategy_name', strategy.get('strategy_id', 'Unknown Strategy'))
    actions = strategy.get('actions', [])

    if not actions:
        logging.info(f"[ActionTrigger] No actions defined for strategy '{strategy_name}'.")
        return

    logging.info(f"[ActionTrigger] Executing actions for strategy '{strategy_name}'...")

    # Determine current price for trade simulations
    # This relies on get_indicator_values being able to fetch 'PRICE'
    current_price, _ = get_indicator_values('PRICE', current_market_state)
    if current_price is None:
        logging.warning(f"[ActionTrigger] Could not determine current price for SYMBOL '{SYMBOL}'. Trade simulations might be affected.")
        # Fallback or decide if actions like BUY/SELL should proceed without a known price.
        # For simulation, logging 'N/A' for price is okay.

    for action_item in actions:
        action_type = action_item.get('type')
        action_details_str = action_item.get('details', '') # Details like amount, email address, etc.

        logging.info(f"[ActionTrigger] Processing action: {action_type}, Details: '{action_details_str}' for strategy '{strategy_name}'")

        if action_type == 'NOTIFY_ALERT':
            # Log to console/server log. Frontend alert would require WebSocket push or different mechanism.
            log_message = f"ALERT from strategy '{strategy_name}': Symbol {SYMBOL}, Details: {action_details_str} (Current Price: {current_price if current_price is not None else 'N/A'})"
            logging.info(log_message) # Main log
            print(f"STRATEGY_ALERT: {log_message}") # Also print for more visibility if running locally

        elif action_type == 'NOTIFY_EMAIL':
            # Simulate email notification by logging.
            # Actual email sending would require SMTP setup, email libraries, templates, etc.
            recipient = action_details_str if action_details_str else "default_recipient@example.com"
            log_message = f"SIMULATED EMAIL to {recipient} from strategy '{strategy_name}': Symbol {SYMBOL}, Details: (Further details from strategy/market state could be included here). Current Price: {current_price if current_price is not None else 'N/A'}."
            logging.info(log_message)
            print(f"STRATEGY_EMAIL_SIM: {log_message}")


        elif action_type == 'BUY':
            # Simulate a BUY order. Actual trading needs Deriv API integration for 'buy' call.
            # action_details might contain stake amount, contract type, etc.
            stake = action_details_str if action_details_str else "default_stake" # Example
            log_message = f"SIMULATED TRADE (BUY): Strategy '{strategy_name}', Symbol: {SYMBOL}, Price: {current_price if current_price is not None else 'N/A'}, Stake/Details: {stake}"
            logging.info(log_message)
            print(f"STRATEGY_TRADE_SIM: {log_message}")

            # Placeholder for actual trade execution logic using Deriv API
            # proposal_payload = { ... }
            # proposal_data, proposal_error = send_ws_request_and_wait('trade_proposal', proposal_payload)
            # if proposal_data:
            #     buy_payload = { "buy": proposal_data['proposal']['id'], "price": proposal_data['proposal']['ask_price'] }
            #     buy_data, buy_error = send_ws_request_and_wait('trade_buy', buy_payload)
            #     if buy_data: logging.info(f"Actual BUY executed: {buy_data}")
            #     else: logging.error(f"Actual BUY failed: {buy_error}")
            # else: logging.error(f"Trade proposal failed for BUY action: {proposal_error}")


        elif action_type == 'SELL':
            # Simulate a SELL order.
            stake = action_details_str if action_details_str else "default_stake" # Example
            log_message = f"SIMULATED TRADE (SELL): Strategy '{strategy_name}', Symbol: {SYMBOL}, Price: {current_price if current_price is not None else 'N/A'}, Stake/Details: {stake}"
            logging.info(log_message)
            print(f"STRATEGY_TRADE_SIM: {log_message}")

            # Placeholder for actual trade execution logic (similar to BUY)

        else:
            logging.warning(f"[ActionTrigger] Unknown action type '{action_type}' in strategy '{strategy_name}'.")

def evaluate_strategies():
    """
    Main function to iterate through all custom strategies, evaluate their conditions,
    and trigger actions if all conditions for a strategy are met.
    """
    active_strategies = {}
    with strategies_lock:
        if not custom_strategies:
            # logging.debug("[StrategyEval] No custom strategies to evaluate.")
            return
        # Create a copy to iterate over in case strategies are modified concurrently (less likely here)
        active_strategies = copy.deepcopy(custom_strategies)

    current_market_state_snapshot = {}
    with market_data_lock:
        # Create a deepcopy for consistent view of market data during this evaluation cycle
        current_market_state_snapshot = copy.deepcopy(market_data)

    # Basic check: Do we have any price data to evaluate against?
    # This check can be made more specific based on strategy requirements (e.g., specific indicators ready)
    if not current_market_state_snapshot.get('prices') and not current_market_state_snapshot.get('ohlcv_candles'):
        logging.debug("[StrategyEval] Market data (prices/candles) not yet sufficient for evaluation.")
        return

    # Add current_chart_type to the snapshot for check_single_condition if it's not already there
    # It's added to market_data by /api/market_data but not intrinsically part of core market_data updates.
    # For robustness, let's ensure it's available for get_indicator_values.
    if 'current_chart_type' not in current_market_state_snapshot:
        global current_chart_type # Access the global
        current_market_state_snapshot['current_chart_type'] = current_chart_type
        logging.debug(f"[StrategyEval] Added global current_chart_type ('{current_chart_type}') to market_state_snapshot.")


    # logging.debug(f"[StrategyEval] Starting evaluation for {len(active_strategies)} strategies against symbol {SYMBOL}.")

    for strategy_id, strategy in active_strategies.items():
        strategy_name = strategy.get('strategy_name', strategy_id)

        if not strategy.get('is_active', True): # Check if strategy is active
            # logging.debug(f"[StrategyEval] Strategy '{strategy_name}' is not active. Skipping.")
            continue

        # TODO: Future: Check if strategy is applicable to the current SYMBOL

        conditions_group = strategy.get('conditions_group')
        if not conditions_group or not conditions_group.get('conditions'):
            # logging.debug(f"[StrategyEval] Strategy '{strategy_name}' has no conditions. Skipping.")
            continue

        all_conditions_met = True # Assuming AND logic for conditions in the group

        # logging.debug(f"[StrategyEval] Evaluating strategy '{strategy_name}'...")
        for condition_item in conditions_group['conditions']:
            try:
                condition_met = check_single_condition(condition_item, current_market_state_snapshot)
                if not condition_met:
                    all_conditions_met = False
                    # logging.debug(f"[StrategyEval] Condition {condition_item} for '{strategy_name}' not met. Stopping evaluation for this strategy.")
                    break # For AND logic, if one condition fails, the group fails
            except Exception as e:
                logging.error(f"[StrategyEval] Error checking condition {condition_item} for strategy '{strategy_name}': {e}", exc_info=True)
                all_conditions_met = False
                break # Error in condition checking, treat as not met

        if all_conditions_met:
            logging.info(f"[StrategyEval] All conditions met for strategy '{strategy_name}'.")
            try:
                trigger_strategy_actions(strategy, current_market_state_snapshot)
            except Exception as e:
                logging.error(f"[StrategyEval] Error triggering actions for strategy '{strategy_name}': {e}", exc_info=True)
        # else:
            # logging.debug(f"[StrategyEval] Not all conditions met for strategy '{strategy_name}'.")
    # logging.debug("[StrategyEval] Finished strategy evaluation cycle.")
# --- End Strategy Evaluation Helper Functions ---



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
    global market_data, SYMBOL, current_tick_subscription_id, current_chart_type, current_granularity_seconds, in_progress_ohlcv_candle
    global pending_api_requests, api_response_events, api_response_data, shared_data_lock, market_data_lock
    
    data = json.loads(message)
    # logging.debug(f"Raw WS message received: {message[:500]}") # Can be too verbose

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

            # Calculate indicators based on the determined input data
            if not indicator_input_data_for_calc:
                logging.warning("No data (tick or ohlcv) available for indicator calculation on new tick. Skipping.")
            else:
                updated_data = calculate_indicators(indicator_input_data_for_calc)
                for key, value in updated_data.items():
                    if key in market_data: 
                        if isinstance(value, list): market_data[key] = value 
                        elif isinstance(value, str): market_data[key] = value

                # <<< Call evaluate_strategies after indicators are updated >>>
                if ws.keep_running: # Check if WebSocket is still supposed to be running
                    try:
                        evaluate_strategies()
                    except Exception as e:
                        logging.error(f"[StrategyEvalLoop] Error during evaluate_strategies from tick: {e}", exc_info=True)

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
                        updated_data_from_ohlcv = calculate_indicators(parsed_ohlcv_candles) # Use the new candles
                        for key, value in updated_data_from_ohlcv.items():
                            if key in market_data:
                                if isinstance(value, list): market_data[key] = value
                                elif isinstance(value, str): market_data[key] = value

                        # <<< Call evaluate_strategies after indicators are updated from new candles >>>
                        if ws.keep_running: # Check if WebSocket is still supposed to be running
                            try:
                                evaluate_strategies()
                            except Exception as e:
                                logging.error(f"[StrategyEvalLoop] Error during evaluate_strategies from candles: {e}", exc_info=True)
                        logging.info(f"All indicators recalculated using new OHLCV data (req_id: {req_id_of_message}, type: {request_type_str}). ATR: {'Yes' if 'atr_14' in updated_data_from_ohlcv and updated_data_from_ohlcv['atr_14'] else 'No/Empty'}")
            else: # This else is for if not parsed_ohlcv_candles
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
    global ws_thread, ws_app, API_TOKEN, market_data, current_tick_subscription_id, in_progress_ohlcv_candle
    if token: API_TOKEN = token
    
    if ws_app and ws_app.sock and ws_app.sock.connected:
        logging.info("Existing WebSocket connection found. Closing it before restarting.")
        try:
            ws_app.keep_running = False # Signal the run_forever loop to stop
            ws_app.close()
            if ws_thread and ws_thread.is_alive(): ws_thread.join(timeout=5) # Wait for thread to finish
        except Exception as e: logging.error(f"Error closing existing WebSocket: {e}")
    
    logging.info(f"Starting WebSocket connection for SYMBOL: {SYMBOL}.")
    with market_data_lock: # Ensure thread-safe access to market_data and in_progress_ohlcv_candle
        market_data.clear()
        market_data.update(copy.deepcopy(DEFAULT_MARKET_DATA))
        in_progress_ohlcv_candle = None # Reset in-progress candle
        logging.info(f"Market data and in-progress candle cleared and re-initialized for {SYMBOL}.")
    
    current_tick_subscription_id = None # Reset subscription ID

    def run_ws(): # This function runs in a separate thread
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
    global current_chart_type, current_granularity_seconds, ws_app, SYMBOL, market_data_lock, market_data, in_progress_ohlcv_candle

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

        if ws_app and ws_app.sock and ws_app.sock.connected:
            if current_chart_type == 'ohlcv':
                logging.info(f"Requesting new historical OHLCV data for {SYMBOL}, Granularity={current_granularity_seconds}s.")
                request_ohlcv_data(ws_app, SYMBOL, current_granularity_seconds)
            # If chart type is 'tick', live ticks will start populating. Historical ticks are generally not re-fetched unless it's a symbol change.
            # Daily summary data might also be re-requested if it's considered part of the "main" data display.
            request_daily_data(ws_app, SYMBOL) # Refresh daily summary too
        else:
            logging.warning("WebSocket not connected. Cannot fetch new data on settings change.")
    
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
    # Note: app.run(debug=True) is suitable for development.
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # load_strategies_from_file() # Already called above after app initialization
    app.run(debug=True)


# --- Strategy Management API Endpoints ---
@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    global custom_strategies
    strategy_data = request.get_json()

    if not isinstance(strategy_data, dict) or not strategy_data.get('strategy_name') or not strategy_data.get('conditions_group'):
        logging.warning(f"Bad request for creating strategy: {strategy_data}")
        return jsonify({"error": "Invalid strategy payload. 'strategy_name' and 'conditions_group' are required."}), 400

    strategy_id = strategy_data.get('strategy_id', uuid.uuid4().hex)
    
    with strategies_lock:
        if strategy_id in custom_strategies:
             # If user provides an ID that already exists, it's a conflict, unless we decide PUT-like behavior for POST
             # For strict POST, generate a new ID if one is not provided or if provided one conflicts.
             # Simpler: if ID is given and exists, error. If not given, generate.
             if strategy_data.get('strategy_id'): # If ID was in payload
                logging.warning(f"Attempt to create strategy with existing ID: {strategy_id}")
                return jsonify({"error": f"Strategy with ID {strategy_id} already exists. Use PUT to update or omit ID for new."}), 409 # Conflict
             else: # ID was generated
                 while strategy_id in custom_strategies: # Ensure generated ID is unique (highly unlikely for UUID, but good practice)
                     strategy_id = uuid.uuid4().hex
        
        new_strategy = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_data.get("strategy_name"),
            "description": strategy_data.get("description", ""),
            "conditions_group": strategy_data.get("conditions_group"),
            "actions": strategy_data.get("actions", []),
            "is_active": strategy_data.get("is_active", True), # Defaults to True
            "status_message": strategy_data.get("status_message", "Active"), # Default status
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        custom_strategies[strategy_id] = new_strategy
    
    save_strategies_to_file()
    logging.info(f"Created strategy with ID: {strategy_id}, Name: {new_strategy['strategy_name']}")
    return jsonify(new_strategy), 201

@app.route('/api/strategies', methods=['GET'])
def get_all_strategies():
    with strategies_lock:
        # Return a list of strategy objects, not the dict itself
        return jsonify(list(custom_strategies.values()))

@app.route('/api/strategies/<strategy_id>', methods=['GET'])
def get_strategy_by_id(strategy_id):
    with strategies_lock:
        strategy = custom_strategies.get(strategy_id)
    if strategy:
        return jsonify(strategy)
    else:
        logging.warning(f"Strategy with ID {strategy_id} not found (GET).")
        return jsonify({"error": "Strategy not found"}), 404

@app.route('/api/strategies/<strategy_id>', methods=['PUT'])
def update_strategy(strategy_id):
    global custom_strategies
    strategy_updates = request.get_json()

    if not isinstance(strategy_updates, dict):
        return jsonify({"error": "Invalid payload. Expected a JSON object."}), 400

    with strategies_lock:
        if strategy_id not in custom_strategies:
            logging.warning(f"Strategy with ID {strategy_id} not found (PUT).")
            return jsonify({"error": "Strategy not found"}), 404

        existing_strategy = custom_strategies[strategy_id]
        
        # Update fields if present in the payload
        existing_strategy["strategy_name"] = strategy_updates.get("strategy_name", existing_strategy["strategy_name"])
        existing_strategy["description"] = strategy_updates.get("description", existing_strategy["description"])
        existing_strategy["conditions_group"] = strategy_updates.get("conditions_group", existing_strategy["conditions_group"])
        existing_strategy["actions"] = strategy_updates.get("actions", existing_strategy["actions"])
        existing_strategy["is_active"] = strategy_updates.get("is_active", existing_strategy.get("is_active", True))
        existing_strategy["status_message"] = strategy_updates.get("status_message", existing_strategy.get("status_message", ""))
        existing_strategy["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        custom_strategies[strategy_id] = existing_strategy
    
    save_strategies_to_file()
    logging.info(f"Updated strategy with ID: {strategy_id}")
    return jsonify(existing_strategy)

@app.route('/api/strategies/<strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id):
    global custom_strategies
    with strategies_lock:
        if strategy_id not in custom_strategies:
            logging.warning(f"Strategy with ID {strategy_id} not found (DELETE).")
            return jsonify({"error": "Strategy not found"}), 404
        
        deleted_strategy_name = custom_strategies[strategy_id].get("strategy_name", "N/A")
        del custom_strategies[strategy_id]
    
    save_strategies_to_file()
    logging.info(f"Deleted strategy with ID: {strategy_id}, Name: {deleted_strategy_name}")
    return jsonify({"message": "Strategy deleted successfully"}), 200

@app.route('/api/strategies/<strategy_id>/enable', methods=['POST'])
def enable_strategy(strategy_id):
    global custom_strategies
    with strategies_lock:
        if strategy_id not in custom_strategies:
            logging.warning(f"Strategy with ID {strategy_id} not found (enable).")
            return jsonify({"error": "Strategy not found"}), 404

        custom_strategies[strategy_id]['is_active'] = True
        custom_strategies[strategy_id]['status_message'] = "Active"
        custom_strategies[strategy_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
        strategy = custom_strategies[strategy_id]

    save_strategies_to_file()
    logging.info(f"Enabled strategy ID: {strategy_id}, Name: {strategy.get('strategy_name')}")
    return jsonify(strategy)

@app.route('/api/strategies/<strategy_id>/disable', methods=['POST'])
def disable_strategy(strategy_id):
    global custom_strategies
    with strategies_lock:
        if strategy_id not in custom_strategies:
            logging.warning(f"Strategy with ID {strategy_id} not found (disable).")
            return jsonify({"error": "Strategy not found"}), 404

        custom_strategies[strategy_id]['is_active'] = False
        custom_strategies[strategy_id]['status_message'] = "Disabled by user"
        custom_strategies[strategy_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
        strategy = custom_strategies[strategy_id]

    save_strategies_to_file()
    logging.info(f"Disabled strategy ID: {strategy_id}, Name: {strategy.get('strategy_name')}")
    return jsonify(strategy)
