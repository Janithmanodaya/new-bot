import os
import sys
import time
import math
import asyncio
import logging
import json
import sqlite3
import io
import re
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from collections import deque
from decimal import Decimal, ROUND_DOWN, getcontext
import argparse

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from binance.client import Client
from binance.exceptions import BinanceAPIException

from dotenv import load_dotenv

# Load .env file into environment (if present)
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("backtester")


# --- Default Configuration ---
# This dictionary contains the default settings for the backtest.
# A 'config.json' file will be created on first run, which you can edit.
CONFIG = {
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "TIMEFRAME": "15m",
    "BIG_TIMEFRAME": "4h",
    "INITIAL_CAPITAL": 10000.0,
    "BINANCE_FEE": 0.0004,  # 0.04% fee for futures market orders

    "KAMA_LENGTH": 14,
    "KAMA_FAST": 2,
    "KAMA_SLOW": 30,

    "ATR_LENGTH": 14,
    "SL_TP_ATR_MULT": 2.5,

    "RISK_PCT_LARGE": 0.02,  # 2% risk per trade
    "RISK_SMALL_BALANCE_THRESHOLD": 50.0,
    "RISK_SMALL_FIXED_USDT": 0.5,
    "MAX_RISK_USDT": 0.0,  # 0 disables cap

    "VOLATILITY_ADJUST_ENABLED": True,
    "TRENDING_ADX": 40.0,
    "TRENDING_CHOP": 40.0,
    "TRENDING_RISK_MULT": 1.5,
    "CHOPPY_ADX": 25.0,
    "CHOPPY_CHOP": 60.0,
    "CHOPPY_RISK_MULT": 0.5,

    "ADX_LENGTH": 14,
    "ADX_THRESHOLD": 30.0,

    "CHOP_LENGTH": 14,
    "CHOP_THRESHOLD": 60.0,

    "BB_LENGTH": 20,
    "BB_STD": 2.0,
    "BBWIDTH_THRESHOLD": 12.0,

    "MIN_CANDLES_AFTER_CLOSE": 10,

    "TRAILING_ENABLED": True,
    "BE_AUTO_MOVE_ENABLED": True,

    "DYN_SLTP_ENABLED": True,
    "TP1_ATR_MULT": 1.0,
    "TP2_ATR_MULT": 2.0,
    "TP3_ATR_MULT": 3.0,
    "TP1_CLOSE_PCT": 0.5,
    "TP2_CLOSE_PCT": 0.25,

    "MIN_NOTIONAL_USDT": 5.0,
    "INTRABAR_EXECUTION_RULE": "assume_next_bar_open", # Or "assume_current_bar_close"
    "HEDGING_ENABLED": False,
}

def setup_config(config_file="config.json"):
    """
    Loads configuration from a JSON file. If the file doesn't exist,
    it creates one with default values.
    """
    global CONFIG
    if os.path.exists(config_file):
        log.info(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            CONFIG.update(user_config)
    else:
        log.info(f"Configuration file not found. Creating default {config_file}")
        with open(config_file, 'w') as f:
            json.dump(CONFIG, f, indent=4)
    return CONFIG


# --- Exchange Info Cache & Helpers (from app.py) ---
EXCHANGE_INFO_CACHE = {"data": None}

def fetch_and_cache_exchange_info(client):
    """
    Fetches exchange info and caches it. This should be called once at startup.
    """
    global EXCHANGE_INFO_CACHE
    try:
        log.info("Fetching and caching exchange information...")
        exchange_info = client.futures_exchange_info()
        EXCHANGE_INFO_CACHE["data"] = exchange_info
        log.info("Successfully cached exchange information.")
    except BinanceAPIException as e:
        log.error(f"Failed to fetch exchange info: {e}. The backtester may fail on rounding.")
        sys.exit(1)

def get_exchange_info_sync():
    """Gets the cached exchange information."""
    return EXCHANGE_INFO_CACHE["data"]

def get_symbol_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Retrieves the cached exchange info for a specific symbol."""
    info = get_exchange_info_sync()
    if not info:
        return None
    try:
        symbols = info.get('symbols', [])
        return next((s for s in symbols if s.get('symbol') == symbol), None)
    except Exception:
        return None

def round_price(symbol: str, price: float) -> float:
    """
    Rounds the price to the correct number of decimal places based on the
    symbol's PRICE_FILTER tickSize.
    """
    try:
        info = get_exchange_info_sync()
        if not info or not isinstance(info, dict):
            return float(price)
        symbol_info = next((s for s in info.get('symbols', []) if s.get('symbol') == symbol), None)
        if not symbol_info:
            return float(price)
        for f in symbol_info.get('filters', []):
            if f.get('filterType') == 'PRICE_FILTER':
                tick_size = Decimal(str(f.get('tickSize', '0.00000001')))
                getcontext().prec = 28
                p = Decimal(str(price))
                rounded_price = p.quantize(tick_size, rounding=ROUND_DOWN)
                return float(rounded_price)
    except Exception:
        log.exception("round_price failed; falling back to float")
    return float(price)

def get_max_leverage(symbol: str) -> int:
    """Gets the max leverage for a symbol from the cached exchange info."""
    try:
        s = get_symbol_info(symbol)
        if s:
            # The 'leverages' list contains leverage brackets. We want the highest one.
            # Example: ['125', '100', '50', '20', '10', '5', '4', '3', '2', '1']
            if 'leverages' in s and isinstance(s['leverages'], list) and s['leverages']:
                return int(s['leverages'][0])
        # Fallback for older structures or if 'leverages' is not present
        return 125
    except Exception:
        return 125

def round_qty(symbol: str, qty: float) -> float:
    """
    Rounds the quantity to the correct number of decimal places based on the
    symbol's LOT_SIZE stepSize.
    """
    try:
        info = get_exchange_info_sync()
        if not info or not isinstance(info, dict):
            return float(qty)
        symbol_info = next((s for s in info.get('symbols', []) if s.get('symbol') == symbol), None)
        if not symbol_info:
            return float(qty)
        for f in symbol_info.get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                step = Decimal(str(f.get('stepSize', '1')))
                getcontext().prec = 28
                q = Decimal(str(qty))
                steps = (q // step)
                quant = (steps * step).quantize(step, rounding=ROUND_DOWN)
                if quant <= 0:
                    return 0.0
                return float(quant)
    except Exception:
        log.exception("round_qty failed; falling back to float")
    return float(qty)


# --- Indicators (copied from app.py) ---

def init_binance_client():
    """Initializes the Binance client from environment variables."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        log.error("Binance API key/secret not found in environment variables.")
        log.error("Please set BINANCE_API_KEY and BINANCE_API_SECRET to download data.")
        sys.exit(1)
    
    client = Client(api_key, api_secret)
    log.info("Binance client initialized successfully.")
    return client

def download_historical_data(client, config, data_file="historical_data.parquet"):
    """
    Downloads historical k-line data from Binance and saves it to a file.
    """
    try:
        days_input = input("How many days of historical data would you like to download? ")
        days = int(days_input)
        if days <= 0:
            raise ValueError("Please enter a positive number of days.")
    except (ValueError, EOFError) as e:
        log.error(f"Invalid input: {e}. Please run the script again and enter a positive number.")
        sys.exit(1)

    all_klines_df = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%d %b, %Y")

    log.info(f"Downloading {days} days of data from {start_str} for symbols: {config['SYMBOLS']}")

    for symbol in config['SYMBOLS']:
        log.info(f"Fetching data for {symbol}...")
        try:
            klines_generator = client.get_historical_klines_generator(
                symbol,
                config['TIMEFRAME'],
                start_str=start_str
            )
            
            cols = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore']
            df = pd.DataFrame(klines_generator, columns=cols)
            
            if df.empty:
                log.warning(f"No data found for {symbol} in the specified range.")
                continue

            # --- Data Cleaning and Type Conversion ---
            df['symbol'] = symbol
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume', 'qav', 'taker_base', 'taker_quote']:
                df[col] = pd.to_numeric(df[col])
            df.set_index('close_time', inplace=True)
            
            all_klines_df.append(df)
        except BinanceAPIException as e:
            log.error(f"Error downloading data for {symbol}: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred for {symbol}: {e}")


    if not all_klines_df:
        log.error("Failed to download any data. Exiting.")
        sys.exit(1)

    final_df = pd.concat(all_klines_df)
    log.info(f"Downloaded a total of {len(final_df)} k-lines.")
    
    # Save to efficient parquet format
    final_df.to_parquet(data_file)
    log.info(f"Data saved successfully to {data_file}")
    return final_df

def kama(series: pd.Series, length: int, fast: int, slow: int) -> pd.Series:
    price = series.values
    n = len(price)
    kama_arr = np.zeros(n)
    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)
    if n >= length:
        kama_arr[:length] = np.mean(price[:length])
    else:
        kama_arr[:] = price.mean()
    for i in range(length, n):
        change = abs(price[i] - price[i - length])
        volatility = np.sum(np.abs(price[i - length + 1:i + 1] - price[i - length:i]))
        er = 0.0
        if volatility != 0:
            er = change / volatility
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        kama_arr[i] = kama_arr[i - 1] + sc * (price[i] - kama_arr[i - 1])
    return pd.Series(kama_arr, index=series.index)

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def adx(df: pd.DataFrame, length: int) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_w = tr.rolling(length, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(length, min_periods=1).sum() / atr_w)
    minus_di = 100 * (minus_dm.rolling(length, min_periods=1).sum() / atr_w)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0).fillna(0) * 100
    return dx.rolling(length, min_periods=1).mean()

def choppiness_index(df: pd.DataFrame, length: int) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    sum_tr = tr.rolling(length, min_periods=1).sum()
    hh = high.rolling(length, min_periods=1).max()
    ll = low.rolling(length, min_periods=1).min()
    denom = hh - ll
    denom = denom.replace(0, np.nan)
    chop = 100 * (np.log10(sum_tr / denom) / np.log10(length))
    chop = chop.replace([np.inf, -np.inf], 100).fillna(100)
    return chop

def bb_width(df: pd.DataFrame, length: int, std_mult: float) -> pd.Series:
    ma = df['close'].rolling(length, min_periods=1).mean()
    std = df['close'].rolling(length, min_periods=1).std().fillna(0)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    mid = ma.replace(0, np.nan)
    width = (upper - lower) / mid
    width = width.replace([np.inf, -np.inf], 100).fillna(100)
    return width

# --- Main application logic will go here ---

def calculate_risk_amount(account_balance: float, config: dict) -> float:
    """
    Calculates the USDT amount to risk for a trade based on the rules from app.py.
    """
    if account_balance < config["RISK_SMALL_BALANCE_THRESHOLD"]:
        risk = config["RISK_SMALL_FIXED_USDT"]
    else:
        risk = account_balance * config["RISK_PCT_LARGE"]
    
    max_cap = config.get("MAX_RISK_USDT", 0.0)
    if max_cap and max_cap > 0:
        risk = min(risk, max_cap)
        
    return float(risk)


def calculate_trade_details(symbol: str, price: float, stop_price: float, balance: float, risk_multiplier: float, config: dict) -> Optional[Dict[str, Any]]:
    """
    Calculates the full trade details (qty, notional, leverage, risk) based on the app.py logic.
    Returns a dictionary with trade details or None if the trade is invalid.
    """
    risk_usdt = calculate_risk_amount(balance, config) * risk_multiplier
    if risk_usdt <= 0:
        log.debug(f"Skipping trade for {symbol}: Risk amount is not positive ({risk_usdt})")
        return None

    price_distance = abs(price - stop_price)
    if price_distance <= 0:
        log.debug(f"Skipping trade for {symbol}: Price distance for SL is zero.")
        return None

    # Initial quantity calculation
    qty = risk_usdt / price_distance
    qty = round_qty(symbol, qty)
    if qty <= 0:
        log.debug(f"Skipping trade for {symbol}: Quantity rounded to zero.")
        return None

    notional = qty * price
    min_notional = config.get("MIN_NOTIONAL_USDT", 5.0)

    # Boost to meet minimum notional if necessary
    if notional < min_notional:
        required_qty = min_notional / price
        new_risk = required_qty * price_distance
        
        # In a backtest, we assume we can always take the risk if needed to meet the minimum.
        # A live bot might have a check here (`if new_risk > balance`).
        risk_usdt = new_risk
        qty = round_qty(symbol, required_qty)
        if qty <= 0:
            log.debug(f"Skipping trade for {symbol}: Boosted quantity rounded to zero.")
            return None
        notional = qty * price

    # Final check after potential boosting
    if notional < min_notional:
        log.debug(f"Skipping trade for {symbol}: Notional value ({notional}) is still less than minimum ({min_notional}).")
        return None

    # Leverage Calculation (from app.py)
    margin_to_use = risk_usdt
    if balance < config["RISK_SMALL_BALANCE_THRESHOLD"]:
        # In the live app, this is a separate config, but we can derive it for simplicity
        margin_to_use = config.get("MARGIN_USDT_SMALL_BALANCE", 2.0)
        
    margin_to_use = min(margin_to_use, notional)
    leverage = int(math.floor(notional / max(margin_to_use, 1e-9)))

    # Apply safety caps on leverage
    max_leverage_from_config = config.get("MAX_BOT_LEVERAGE", 20)
    max_leverage_from_exchange = get_max_leverage(symbol)
    leverage = max(1, min(leverage, max_leverage_from_config, max_leverage_from_exchange))

    return {
        "qty": qty,
        "notional": notional,
        "leverage": leverage,
        "risk_usdt": risk_usdt
    }

def simulate_place_market_order_with_sl_tp(
    symbol: str, side: str, qty: float, leverage: int,
    entry_price: float, sl_price: float, tp_price: float,
    timestamp: datetime, risk_usdt: float, notional: float,
    tp1: Optional[float], tp2: Optional[float]
) -> Dict[str, Any]:
    """
    Simulates the placement of a market order and returns a trade object
    structured like the one in app.py.
    """
    trade_id = f"{symbol}_{int(timestamp.timestamp())}"
    
    # In a backtest, the order is "filled" instantly at the determined entry_price.
    # The structure here mimics the 'meta' object from app.py.
    trade = {
        "id": trade_id,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "entry_time": timestamp,
        "initial_qty": qty,
        "qty": qty,
        "notional": notional,
        "leverage": leverage,
        "sl": sl_price,
        "tp": tp_price,
        "risk_usdt": risk_usdt,
        "trade_phase": 0,
        "be_moved": False,
        "tp1": tp1,
        "tp2": tp2,
        # sltp_orders would be populated in a live environment, but is less critical here
        "sltp_orders": {"stop_order": {"status": "NEW"}, "tp_order": {"status": "NEW"}},
    }
    return trade


def run_backtest(config, data_df):
    """The core backtesting engine, now with advanced trade management."""
    log.info("Starting backtest with advanced trade management...")
    
    # --- Initialization ---
    capital = config["INITIAL_CAPITAL"]
    equity_curve = []
    open_trades = []
    closed_trades = []
    last_trade_close_time = {}
    trade_id_counter = 0

    # --- Pre-calculate Indicators ---
    log.info("Pre-calculating indicators for all symbols...")
    indicator_dfs = []
    for symbol, group_df in data_df.groupby('symbol'):
        group_df_copy = group_df.copy()
        group_df_copy['kama'] = kama(group_df_copy['close'], config["KAMA_LENGTH"], config["KAMA_FAST"], config["KAMA_SLOW"])
        group_df_copy['atr'] = atr(group_df_copy, config["ATR_LENGTH"])
        group_df_copy['adx'] = adx(group_df_copy, config["ADX_LENGTH"])
        group_df_copy['chop'] = choppiness_index(group_df_copy, config["CHOP_LENGTH"])
        group_df_copy['bbw'] = bb_width(group_df_copy, config["BB_LENGTH"], config["BB_STD"]) * 100.0
        indicator_dfs.append(group_df_copy)

    if not indicator_dfs:
        log.error("Could not calculate indicators for any symbol. Exiting.")
        return None
        
    data_df = pd.concat(indicator_dfs).sort_index()

    data_df_resample = data_df.set_index('open_time')
    big_tf_df = data_df_resample.groupby('symbol').resample(config['BIG_TIMEFRAME']).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    big_tf_df['kama'] = big_tf_df.groupby('symbol')['close'].transform(lambda x: kama(x, config["KAMA_LENGTH"], config["KAMA_FAST"], config["KAMA_SLOW"]))
    big_tf_df['trend_big'] = big_tf_df.groupby('symbol')['kama'].diff().apply(lambda x: 'bull' if x > 0 else 'bear')
    
    data_df_reset = data_df.reset_index()
    big_tf_df_reset = big_tf_df.reset_index()
    data_df_sorted = data_df_reset.sort_values('open_time')
    big_tf_df_sorted = big_tf_df_reset.sort_values('open_time')
    
    merged_df = pd.merge_asof(left=data_df_sorted, right=big_tf_df_sorted[['symbol', 'open_time', 'trend_big']], on='open_time', by='symbol', direction='backward')
    data_df = merged_df.set_index('close_time').sort_index()
    
    data_df['prev_close'] = data_df.groupby('symbol')['close'].shift(1)
    data_df['prev_kama'] = data_df.groupby('symbol')['kama'].shift(1)

    log.info("Indicator calculation complete. Starting main backtest loop...")
    
    # --- Main Loop ---
    # Use .iterrows() to get index (timestamp) and row data
    for i, (timestamp, row) in enumerate(data_df.iterrows()):
        
        # --- Manage Open Trades ---
        # This part checks existing trades against the current bar's data (e.g., for SL/TP hits)
        # --- Manage Open Trades ---
        unrealized_pnl_for_symbol = 0
        for trade in open_trades[:]:
            if trade['symbol'] != row['symbol']:
                continue

            # Update unrealized PnL for the current symbol
            unrealized_pnl_for_symbol += (row['close'] - trade['entry_price']) * trade['qty'] if trade['side'] == 'BUY' else (trade['entry_price'] - row['close']) * trade['qty']

            # --- Dynamic TP & SL Management ---
            # Phase 1: TP1 Hit -> Move SL to Breakeven
            if config["DYN_SLTP_ENABLED"] and trade['phase'] == 0:
                hit_tp1 = (trade['side'] == 'BUY' and row['high'] >= trade['tp1']) or \
                          (trade['side'] == 'SELL' and row['low'] <= trade['tp1'])
                if hit_tp1:
                    qty_to_close = round_qty(row['symbol'], trade['initial_qty'] * config['TP1_CLOSE_PCT'])
                    if qty_to_close > 0:
                        exit_price = trade['tp1']
                        pnl = (exit_price - trade['entry_price']) * qty_to_close if trade['side'] == 'BUY' else (trade['entry_price'] - exit_price) * qty_to_close
                        exit_notional = qty_to_close * exit_price
                        fee = exit_notional * config['BINANCE_FEE']
                        capital += pnl - fee
                        
                        closed_trades.append({'id': f"{trade['id']}_p1", 'exit_price': exit_price, 'exit_time': timestamp, 'pnl': pnl - fee, 'qty': qty_to_close})
                        
                        trade['qty'] -= qty_to_close
                        trade['sl'] = trade['entry_price']
                        trade['phase'] = 1
                        log.info(f"Trade {trade['id']} hit TP1. Closed {qty_to_close} units. Moved SL to BE.")

            # Phase 2: TP2 Hit -> Move SL to TP1
            if config["DYN_SLTP_ENABLED"] and trade['phase'] == 1:
                hit_tp2 = (trade['side'] == 'BUY' and row['high'] >= trade['tp2']) or \
                          (trade['side'] == 'SELL' and row['low'] <= trade['tp2'])
                if hit_tp2:
                    qty_to_close = round_qty(row['symbol'], trade['initial_qty'] * config['TP2_CLOSE_PCT'])
                    if qty_to_close > 0:
                        exit_price = trade['tp2']
                        pnl = (exit_price - trade['entry_price']) * qty_to_close if trade['side'] == 'BUY' else (trade['entry_price'] - exit_price) * qty_to_close
                        exit_notional = qty_to_close * exit_price
                        fee = exit_notional * config['BINANCE_FEE']
                        capital += pnl - fee
                        
                        closed_trades.append({'id': f"{trade['id']}_p2", 'exit_price': exit_price, 'exit_time': timestamp, 'pnl': pnl - fee, 'qty': qty_to_close})
                        
                        trade['qty'] -= qty_to_close
                        trade['sl'] = trade['tp1']
                        trade['phase'] = 2
                        log.info(f"Trade {trade['id']} hit TP2. Closed {qty_to_close} units. Moved SL to TP1.")

            # --- Final SL/TP check for the remaining position ---
            closed = False
            exit_price = 0
            if trade['side'] == 'BUY':
                if row['low'] <= trade['sl']: closed = True; exit_price = trade['sl']
                elif row['high'] >= trade['tp']: closed = True; exit_price = trade['tp']
            elif trade['side'] == 'SELL':
                if row['high'] >= trade['sl']: closed = True; exit_price = trade['sl']
                elif row['low'] <= trade['tp']: closed = True; exit_price = trade['tp']

            if closed:
                pnl = (exit_price - trade['entry_price']) * trade['qty'] if trade['side'] == 'BUY' else (trade['entry_price'] - exit_price) * trade['qty']
                exit_notional = trade['qty'] * exit_price
                fee = exit_notional * config['BINANCE_FEE']
                capital += pnl - fee
                
                closed_trades.append({'id': f"{trade['id']}_final", 'exit_price': exit_price, 'exit_time': timestamp, 'pnl': pnl - fee, 'qty': trade['qty']})
                open_trades.remove(trade)
                last_trade_close_time[row['symbol']] = timestamp
                continue
        
        # --- Check for New Entries ---
        existing_trade_for_symbol = next((t for t in open_trades if t['symbol'] == row['symbol']), None)
        if not config.get("HEDGING_ENABLED", False) and existing_trade_for_symbol:
            continue
        
        if row['symbol'] in last_trade_close_time and (timestamp - last_trade_close_time[row['symbol']]) / pd.Timedelta(config['TIMEFRAME']) < config["MIN_CANDLES_AFTER_CLOSE"]:
            continue
        
        kama_now, kama_prev, prev_close = row['kama'], row['prev_kama'], row['prev_close']
        if pd.isna(kama_prev) or pd.isna(prev_close): continue
        
        trend_small = 'bull' if (kama_now - kama_prev) > 0 else 'bear'
        trend_big = row['trend_big']
        
        if trend_small != trend_big: continue
        if not (row['adx'] >= config["ADX_THRESHOLD"] and row['chop'] < config["CHOP_THRESHOLD"] and row['bbw'] < config["BBWIDTH_THRESHOLD"]): continue
        
        crossed_above = (prev_close <= kama_prev) and (row['close'] > kama_now)
        crossed_below = (prev_close >= kama_prev) and (row['close'] < kama_now)
        
        side = None
        if crossed_above and trend_small == 'bull': side = 'BUY'
        elif crossed_below and trend_small == 'bear': side = 'SELL'
        
        if side:
            # --- Get Entry Price based on Execution Rule ---
            if i + 1 >= len(data_df): continue # Cannot enter on the last bar
            next_bar = data_df.iloc[i+1]
            if next_bar['symbol'] != row['symbol']: continue # Ensure next bar is the same symbol in a multi-asset df

            entry_price = next_bar['open']
            entry_timestamp = next_bar.name # This is the index (timestamp) of the next bar
            
            # --- Calculate SL/TP based on the signal bar's data ---
            risk_multiplier = 1.0
            if config["VOLATILITY_ADJUST_ENABLED"]:
                if row['adx'] > config["TRENDING_ADX"] and row['chop'] < config["TRENDING_CHOP"]: risk_multiplier = config["TRENDING_RISK_MULT"]
                elif row['adx'] < config["CHOPPY_ADX"] or row['chop'] > config["CHOPPY_CHOP"]: risk_multiplier = config["CHOPPY_RISK_MULT"]
            
            sl_distance = config["SL_TP_ATR_MULT"] * row['atr']
            if sl_distance <= 0: continue
            
            stop_price = entry_price - sl_distance if side == 'BUY' else entry_price + sl_distance
            
            # --- Sizing & Leverage ---
            trade_details = calculate_trade_details(
                symbol=row['symbol'], price=entry_price, stop_price=stop_price,
                balance=capital, risk_multiplier=risk_multiplier, config=config
            )
            if not trade_details: continue

            # --- Dynamic TP Calculation ---
            tp1, tp2 = None, None
            final_tp = entry_price + sl_distance if side == 'BUY' else entry_price - sl_distance
            if config["DYN_SLTP_ENABLED"]:
                tp1 = entry_price + (config["TP1_ATR_MULT"] * row['atr']) if side == 'BUY' else entry_price - (config["TP1_ATR_MULT"] * row['atr'])
                tp2 = entry_price + (config["TP2_ATR_MULT"] * row['atr']) if side == 'BUY' else entry_price - (config["TP2_ATR_MULT"] * row['atr'])
                final_tp = entry_price + (config["TP3_ATR_MULT"] * row['atr']) if side == 'BUY' else entry_price - (config["TP3_ATR_MULT"] * row['atr'])

            # --- Simulate Order Placement ---
            new_trade = simulate_place_market_order_with_sl_tp(
                symbol=row['symbol'], side=side, qty=trade_details['qty'], leverage=trade_details['leverage'],
                entry_price=entry_price, sl_price=stop_price, tp_price=final_tp,
                timestamp=entry_timestamp, risk_usdt=trade_details['risk_usdt'], notional=trade_details['notional'],
                tp1=tp1, tp2=tp2
            )
            open_trades.append(new_trade)
            capital -= (trade_details['notional'] * config["BINANCE_FEE"]) # Fee on entry
            
        # --- Update Equity Curve ---
        # Calculate unrealized PnL based on the close of the current bar
        unrealized_pnl = sum(
            (row['close'] - t['entry_price']) * t['qty'] if t['side'] == 'BUY'
            else (t['entry_price'] - row['close']) * t['qty']
            for t in open_trades if t['symbol'] == row['symbol']
        )
        equity_curve.append(capital + unrealized_pnl)

    log.info("Backtest loop finished.")
    
    for trade in open_trades:
        final_row = data_df[data_df['symbol'] == trade['symbol']].iloc[-1]
        exit_price = final_row['close']
        fee = (trade['notional'] + (trade['qty'] * exit_price)) * config['BINANCE_FEE']
        pnl = (exit_price - trade['entry_price']) * trade['qty'] if trade['side'] == 'BUY' else (trade['entry_price'] - exit_price) * trade['qty']
        capital += pnl - fee
        closed_trades.append({'id': f"{trade['id']}_force_closed", 'exit_price': exit_price, 'exit_time': final_row.name, 'pnl': pnl - fee, 'qty': trade['qty']})

    results = {"closed_trades": closed_trades, "equity_curve": equity_curve, "config": config}
    log.info(f"Backtest complete. Total trades: {len(closed_trades)}. Final equity: {capital:.2f}")
    return results

def generate_html_report(results, config, filename="backtest_report.html"):
    """Generates a comprehensive HTML report with stats and interactive charts."""
    log.info(f"Generating HTML report to {filename}...")

    closed_trades = results.get('closed_trades', [])
    if not closed_trades:
        log.warning("No trades were made. Cannot generate a report.")
        html_content = "<html><body><h1>Backtest Report</h1><p>No trades were executed in this backtest.</p></body></html>"
        with open(filename, 'w') as f:
            f.write(html_content)
        return

    trades_df = pd.DataFrame(closed_trades)
    # The exit_time is already a datetime object from the backtest loop
    # trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # --- KPI Calculations ---
    initial_capital = config['INITIAL_CAPITAL']
    total_trades = len(trades_df)
    winning_trades_df = trades_df[trades_df['pnl'] > 0].copy()
    losing_trades_df = trades_df[trades_df['pnl'] <= 0].copy()
    
    win_count = len(winning_trades_df)
    loss_count = len(losing_trades_df)
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    final_equity = initial_capital + total_pnl
    total_return_pct = (total_pnl / initial_capital) * 100

    gross_profit = winning_trades_df['pnl'].sum()
    gross_loss = abs(losing_trades_df['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = winning_trades_df['pnl'].mean() if not winning_trades_df.empty else 0
    avg_loss = abs(losing_trades_df['pnl'].mean()) if not losing_trades_df.empty else 0
    risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # --- Corrected Drawdown Calculation ---
    equity_curve = pd.Series(results['equity_curve'], dtype=np.float64)
    if equity_curve.empty or equity_curve.iloc[0] != initial_capital:
        equity_curve = pd.concat([pd.Series([initial_capital]), equity_curve], ignore_index=True)
    
    running_max = equity_curve.cummax()
    # Avoid division by zero. Replace 0 with NaN and then fill with 0.
    drawdown = (running_max - equity_curve) / running_max.replace(0, np.nan)
    drawdown[equity_curve < 0] = 1.0 # If equity is negative, drawdown is 100%
    drawdown.fillna(0, inplace=True)
    max_drawdown_pct = drawdown.max() * 100 if not drawdown.empty else 0
    
    # Find drawdown period for chart annotation
    max_dd_end_idx = drawdown.idxmax() if not drawdown.empty else -1
    max_dd_start_idx = -1
    if max_dd_end_idx > 0:
        try:
            relevant_equity = equity_curve.iloc[:max_dd_end_idx + 1]
            peak_value = relevant_equity.cummax().iloc[-1]
            max_dd_start_idx = relevant_equity[relevant_equity >= peak_value].index[-1]
        except (IndexError, KeyError):
            max_dd_start_idx = 0 # Fallback

    # --- Plotly Charts (with Dark Theme and Enhancements) ---
    template = 'plotly_dark'
    
    # Adaptive Equity Chart
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(
        x=np.arange(len(equity_curve)), 
        y=equity_curve, 
        mode='lines', 
        name='Account Balance', 
        line=dict(color='#17becf', width=2)
    ))
    if max_dd_start_idx != -1 and max_dd_end_idx != -1 and (max_dd_end_idx > max_dd_start_idx):
        equity_fig.add_vrect(
            x0=max_dd_start_idx, x1=max_dd_end_idx,
            fillcolor="rgba(239, 83, 80, 0.2)", line_width=0,
            annotation_text=f"Max Drawdown: {max_drawdown_pct:.2f}%", 
            annotation_position="top left",
            annotation=dict(font=dict(color='white'))
        )
    equity_fig.update_layout(
        title_text='<b>Account Balance Over Time</b>',
        xaxis_title='Trade/Event Number',
        yaxis_title='Account Balance (USDT)',
        template=template,
        height=500,
        legend=dict(x=0.01, y=0.99, bordercolor='rgba(255,255,255,0.2)', borderwidth=1)
    )

    # Monthly PnL Chart
    monthly_pnl = trades_df.set_index('exit_time')['pnl'].resample('M').sum()
    monthly_fig = go.Figure(data=[go.Bar(
        x=monthly_pnl.index.strftime('%Y-%m'), 
        y=monthly_pnl, 
        name='Monthly PnL',
        marker_color=['#28a745' if p > 0 else '#dc3545' for p in monthly_pnl]
    )])
    monthly_fig.update_layout(
        title_text='<b>Monthly Profit & Loss</b>',
        xaxis_title='Month', 
        yaxis_title='Total PnL (USDT)',
        template=template
    )
    
    # --- HTML Tables for Trades ---
    display_cols = ['id', 'exit_time', 'exit_price', 'pnl', 'qty']
    winning_trades_df['pnl'] = winning_trades_df['pnl'].round(2)
    losing_trades_df['pnl'] = losing_trades_df['pnl'].round(2)
    win_table_html = winning_trades_df[display_cols].to_html(classes='trade-table', index=False)
    loss_table_html = losing_trades_df[display_cols].to_html(classes='trade-table', index=False)

    # --- HTML Assembly ---
    equity_chart_html = equity_fig.to_html(full_html=False, include_plotlyjs='cdn')
    monthly_chart_html = monthly_fig.to_html(full_html=False, include_plotlyjs=False)

    # Create parameters table
    params_html = "<h3>Parameters Used</h3><table class='trade-table'>"
    for key, value in config.items():
        params_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
    params_html += "</table>"

    html_template = f"""
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            :root {{
                --bg-color: #1e1e1e; --primary-text: #d4d4d4; --secondary-text: #8c8c8c;
                --card-bg: #2a2a2a; --border-color: #444; --positive: #28a745; --negative: #dc3545;
                --link-color: #3794ff;
            }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 20px; background-color: var(--bg-color); color: var(--primary-text);
            }}
            h1, h2, h3 {{ text-align: center; color: var(--primary-text); font-weight: 300; }}
            h1 {{ font-size: 2.5em; }}
            h2 {{ font-size: 1.8em; margin-top: 40px; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; }}
            h3 {{ font-size: 1.4em; text-align: left; margin-top: 30px; margin-bottom: 15px; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .kpi-box {{ background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; text-align: center; }}
            .kpi-title {{ font-weight: 500; font-size: 0.9em; color: var(--secondary-text); margin-bottom: 8px; }}
            .kpi-value {{ font-size: 1.6em; font-weight: 700; }}
            .positive {{ color: var(--positive); }}
            .negative {{ color: var(--negative); }}
            .chart-container {{ background-color: var(--card-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color); margin-bottom: 20px; }}
            .table-container {{ background-color: var(--card-bg); border-radius: 8px; border: 1px solid var(--border-color); margin-bottom: 20px; overflow-x: auto; padding: 15px; }}
            .trade-table {{ width: 100%; border-collapse: collapse; }}
            .trade-table th, .trade-table td {{ border-bottom: 1px solid var(--border-color); padding: 12px 15px; text-align: left; }}
            .trade-table th {{ background-color: #333; font-weight: 600; }}
            .trade-table tr:hover {{ background-color: #3c3c3c; }}
        </style>
    </head>
    <body>
        <h1>Strategy Backtest Report</h1>
        <h2>Key Performance Indicators</h2>
        <div class="kpi-grid">
            <div class="kpi-box">
                <div class="kpi-title">Total Return</div>
                <div class="kpi-value {'positive' if total_return_pct >= 0 else 'negative'}">{total_return_pct:.2f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Total PnL</div>
                <div class="kpi-value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:,.2f}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Max Drawdown</div>
                <div class="kpi-value negative">{max_drawdown_pct:.2f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Win Rate</div>
                <div class="kpi-value {'positive' if win_rate > 50 else 'negative'}">{win_rate:.2f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Profit Factor</div>
                <div class="kpi-value {'positive' if profit_factor > 1 else 'negative'}">{profit_factor:.2f}</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Avg Win / Loss</div>
                <div class="kpi-value {'positive' if risk_reward_ratio > 1 else 'negative'}">{risk_reward_ratio:.2f}</div>
            </div>
             <div class="kpi-box">
                <div class="kpi-title">Total Trades</div>
                <div class="kpi-value">{total_trades}</div>
            </div>
        </div>

        <h2>Charts</h2>
        <div class="chart-container">{equity_chart_html}</div>
        <div class="chart-container">{monthly_chart_html}</div>

        <h2>Configuration</h2>
        <div class="table-container">
            {params_html}
        </div>

        <h2>Trade History</h2>
        <div class="table-container">
            <h3>Winning Trades ({win_count})</h3>
            {win_table_html}
        </div>
        <div class="table-container">
            <h3>Losing Trades ({loss_count})</h3>
            {loss_table_html}
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_template)
    log.info("Report generated successfully.")

def run_manual_mode(config, historical_df):
    """
    Runs the backtester in manual mode with a single configuration.
    """
    log.info("--- Starting Manual Mode ---")
    
    # Run the backtest
    backtest_results = run_backtest(config, historical_df)

    # Generate the final report
    if backtest_results:
        generate_html_report(backtest_results, config, "manual_backtest_report.html")


def run_auto_mode(base_config, historical_df):
    """
    Runs the backtester in auto-optimization mode, trying different parameter combinations.
    """
    import itertools
    import copy

    log.info("--- Starting Auto-Optimization Mode ---")

    param_grid = {
        'KAMA_LENGTH': [10, 14, 20],
        'ADX_THRESHOLD': [25, 30, 35],
        'CHOP_THRESHOLD': [55, 60, 65],
        'SL_TP_ATR_MULT': [2.0, 2.5, 3.0]
    }
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Generated {len(param_combinations)} parameter combinations to test.")
    
    all_results = []
    
    for i, params in enumerate(param_combinations):
        run_config = copy.deepcopy(base_config)
        run_config.update(params)
        
        print("-" * 70)
        print(f"--> Running Test {i+1}/{len(param_combinations)} | Params: {params}")
        
        results = run_backtest(run_config, historical_df.copy())
        
        if results and results['closed_trades']:
            trades_df = pd.DataFrame(results['closed_trades'])
            final_equity = run_config['INITIAL_CAPITAL'] + trades_df['pnl'].sum()
            total_trades = len(trades_df)
            win_rate = (len(trades_df[trades_df['pnl'] > 0]) / total_trades) * 100 if total_trades > 0 else 0
        else:
            final_equity = run_config['INITIAL_CAPITAL']
            total_trades = 0
            win_rate = 0
        
        print(f"<-- Result: Final Equity ${final_equity:,.2f} | Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
            
        all_results.append({
            'params': params,
            'final_equity': final_equity,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'config': run_config
        })

    if not all_results:
        log.error("No results from any optimization run. Exiting.")
        return

    sorted_results = sorted(all_results, key=lambda x: x['final_equity'], reverse=True)
    
    print("\n" + "="*70)
    print("--- Optimization Complete ---")
    print(f"Finished running {len(param_combinations)} parameter sets.")
    print("\n--- Top 10 Performing Parameter Sets ---")

    top_ten = sorted_results[:10]
    print(f"{'Rank':<5} | {'Final Equity':<15} | {'Trades':<7} | {'Win Rate':<10} | {'Parameters'}")
    print("-" * 80)
    for i, result in enumerate(top_ten):
        rank = i + 1
        equity_str = f"${result['final_equity']:,.2f}"
        win_rate_str = f"{result['win_rate']:.1f}%"
        print(f"{rank:<5} | {equity_str:<15} | {result['total_trades']:<7} | {win_rate_str:<10} | {result['params']}")
    print("="*70)
    
    top_three = sorted_results[:3]
    for i, result in enumerate(top_three):
        rank = i + 1
        log.info(f"Re-running backtest for Rank #{rank} to generate detailed report...")
        final_run_config = result['config']
        final_run_results = run_backtest(final_run_config, historical_df.copy())
        
        if final_run_results:
            report_filename = f"optimization_report_rank_{rank}.html"
            generate_html_report(final_run_results, final_run_config, report_filename)

if __name__ == '__main__':
    log.info("Backtester starting up.")
    CONFIG = setup_config()
    log.info("Loaded configuration.")

    # --- Get Initial Capital from User ---
    while True:
        try:
            capital_input = input(f"Enter initial capital (or press Enter for default ${CONFIG['INITIAL_CAPITAL']}): ")
            if not capital_input:
                break # Keep default
            user_capital = float(capital_input)
            if user_capital <= 0:
                print("Please enter a positive value for the capital.")
                continue
            CONFIG['INITIAL_CAPITAL'] = user_capital
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            log.info("\nExiting.")
            sys.exit(0)
            
    log.info(f"Starting backtest with initial capital of ${CONFIG['INITIAL_CAPITAL']:,.2f}")

    DATA_FILE = "historical_data.parquet"
    
    if os.path.exists(DATA_FILE):
        log.info(f"Loading historical data from {DATA_FILE}...")
        historical_df = pd.read_parquet(DATA_FILE)
        log.info(f"Loaded {len(historical_df)} k-lines from local file.")
    else:
        log.info("Historical data file not found.")
        client = init_binance_client()
        historical_df = download_historical_data(client, CONFIG, DATA_FILE)
        fetch_and_cache_exchange_info(client)

    # --- Mode Selection ---
    while True:
        print("\n--- Select Run Mode ---")
        print("1: Manual Mode (run once with config.json)")
        print("2: Auto-Optimization Mode (find best parameters)")
        
        try:
            choice = input("Enter your choice (1 or 2): ")
            if choice == '1':
                run_manual_mode(CONFIG, historical_df)
                break
            elif choice == '2':
                run_auto_mode(CONFIG, historical_df.copy())
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except (KeyboardInterrupt, EOFError):
            log.info("\nExiting.")
            break
        except Exception as e:
            log.error(f"An unhandled error occurred: {e}")
            log.error("--- Full Traceback ---")
            traceback.print_exc()
            log.error("--- End Traceback ---")
            break

    log.info("Backtest script finished.")
