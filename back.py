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

def round_qty(qty: float) -> float:
    """A simplified rounding function for the backtester."""
    # For this backtest, we'll assume a precision of 3 decimal places for most assets.
    # A real implementation should use symbol-specific step sizes from exchange info.
    return round(qty, 3)

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
    for timestamp, row in data_df.iterrows():
        current_price = row['close']
        
        # --- Manage Open Trades ---
        for trade in open_trades[:]:
            if trade['symbol'] != row['symbol']:
                continue

            # --- Dynamic TP & SL Management ---
            # Phase 1: TP1 Hit -> Move SL to Breakeven
            if config["DYN_SLTP_ENABLED"] and trade['phase'] == 0:
                hit_tp1 = (trade['side'] == 'BUY' and row['high'] >= trade['tp1']) or \
                          (trade['side'] == 'SELL' and row['low'] <= trade['tp1'])
                if hit_tp1:
                    qty_to_close = trade['initial_qty'] * config['TP1_CLOSE_PCT']
                    qty_to_close = round_qty(qty_to_close)
                    
                    if qty_to_close > 0:
                        fee = (trade['entry_price'] * qty_to_close + trade['tp1'] * qty_to_close) * config['BINANCE_FEE']
                        pnl = (trade['tp1'] - trade['entry_price']) * qty_to_close if trade['side'] == 'BUY' else (trade['entry_price'] - trade['tp1']) * qty_to_close
                        capital += pnl - fee
                        
                        closed_trades.append({'id': f"{trade['id']}_p1", 'exit_price': trade['tp1'], 'exit_time': timestamp, 'pnl': pnl - fee, 'qty': qty_to_close})
                        
                        trade['qty'] -= qty_to_close
                        trade['notional'] = trade['qty'] * trade['entry_price']
                        trade['sl'] = trade['entry_price']
                        trade['phase'] = 1
                        log.info(f"Trade {trade['id']} hit TP1. Closed {qty_to_close} units. Moved SL to BE.")

            # Phase 2: TP2 Hit -> Move SL to TP1
            if config["DYN_SLTP_ENABLED"] and trade['phase'] == 1:
                hit_tp2 = (trade['side'] == 'BUY' and row['high'] >= trade['tp2']) or \
                          (trade['side'] == 'SELL' and row['low'] <= trade['tp2'])
                if hit_tp2:
                    qty_to_close = trade['initial_qty'] * config['TP2_CLOSE_PCT']
                    qty_to_close = round_qty(qty_to_close)
                    
                    if qty_to_close > 0:
                        fee = (trade['entry_price'] * qty_to_close + trade['tp2'] * qty_to_close) * config['BINANCE_FEE']
                        pnl = (trade['tp2'] - trade['entry_price']) * qty_to_close if trade['side'] == 'BUY' else (trade['entry_price'] - trade['tp2']) * qty_to_close
                        capital += pnl - fee
                        
                        closed_trades.append({'id': f"{trade['id']}_p2", 'exit_price': trade['tp2'], 'exit_time': timestamp, 'pnl': pnl - fee, 'qty': qty_to_close})
                        
                        trade['qty'] -= qty_to_close
                        trade['notional'] = trade['qty'] * trade['entry_price']
                        trade['sl'] = trade['tp1']
                        trade['phase'] = 2
                        log.info(f"Trade {trade['id']} hit TP2. Closed {qty_to_close} units. Moved SL to TP1.")

            # --- Final SL/TP check ---
            closed = False
            pnl = 0
            if trade['side'] == 'BUY':
                if row['low'] <= trade['sl']:
                    closed = True; exit_price = trade['sl']
                elif row['high'] >= trade['tp']:
                    closed = True; exit_price = trade['tp']
            elif trade['side'] == 'SELL':
                if row['high'] >= trade['sl']:
                    closed = True; exit_price = trade['sl']
                elif row['low'] <= trade['tp']:
                    closed = True; exit_price = trade['tp']

            if closed:
                fee = (trade['notional'] + (trade['qty'] * exit_price)) * config['BINANCE_FEE']
                pnl = (exit_price - trade['entry_price']) * trade['qty'] if trade['side'] == 'BUY' else (trade['entry_price'] - exit_price) * trade['qty']
                capital += pnl - fee
                
                closed_trades.append({'id': f"{trade['id']}_final", 'exit_price': exit_price, 'exit_time': timestamp, 'pnl': pnl - fee, 'qty': trade['qty']})
                open_trades.remove(trade)
                last_trade_close_time[row['symbol']] = timestamp
                continue
        
        # --- Check for New Entries ---
        if any(t['symbol'] == row['symbol'] for t in open_trades): continue
        if row['symbol'] in last_trade_close_time and (timestamp - last_trade_close_time[row['symbol']]) / pd.Timedelta(config['TIMEFRAME']) < config["MIN_CANDLES_AFTER_CLOSE"]: continue
        
        kama_now, kama_prev, prev_close = row['kama'], row['prev_kama'], row['prev_close']
        if pd.isna(kama_prev) or pd.isna(prev_close): continue
        
        trend_small = 'bull' if (kama_now - kama_prev) > 0 else 'bear'
        trend_big = row['trend_big']
        
        if trend_small != trend_big: continue
        if not (row['adx'] >= config["ADX_THRESHOLD"] and row['chop'] < config["CHOP_THRESHOLD"] and row['bbw'] < config["BBWIDTH_THRESHOLD"]): continue
        
        crossed_above = (prev_close <= kama_prev) and (current_price > kama_now)
        crossed_below = (prev_close >= kama_prev) and (current_price < kama_now)
        
        side = None
        if crossed_above and trend_small == 'bull': side = 'BUY'
        elif crossed_below and trend_small == 'bear': side = 'SELL'
        
        if side:
            risk_multiplier = 1.0
            if config["VOLATILITY_ADJUST_ENABLED"]:
                if row['adx'] > config["TRENDING_ADX"] and row['chop'] < config["TRENDING_CHOP"]: risk_multiplier = config["TRENDING_RISK_MULT"]
                elif row['adx'] < config["CHOPPY_ADX"] or row['chop'] > config["CHOPPY_CHOP"]: risk_multiplier = config["CHOPPY_RISK_MULT"]
            
            risk_usdt = calculate_risk_amount(capital, config) * risk_multiplier
            sl_distance = config["SL_TP_ATR_MULT"] * row['atr']
            if sl_distance <= 0: continue
            
            stop_price = current_price - sl_distance if side == 'BUY' else current_price + sl_distance
            price_distance = abs(current_price - stop_price)
            if price_distance <= 0: continue
            
            qty = round_qty(risk_usdt / price_distance)
            notional = qty * current_price
            if notional < config["MIN_NOTIONAL_USDT"]: continue
            
            capital -= (notional * config["BINANCE_FEE"])
            trade_id_counter += 1
            
            tp1, tp2, tp3 = None, None, None
            final_tp = current_price + sl_distance if side == 'BUY' else current_price - sl_distance
            if config["DYN_SLTP_ENABLED"]:
                tp1 = current_price + (config["TP1_ATR_MULT"] * row['atr']) if side == 'BUY' else current_price - (config["TP1_ATR_MULT"] * row['atr'])
                tp2 = current_price + (config["TP2_ATR_MULT"] * row['atr']) if side == 'BUY' else current_price - (config["TP2_ATR_MULT"] * row['atr'])
                final_tp = current_price + (config["TP3_ATR_MULT"] * row['atr']) if side == 'BUY' else current_price - (config["TP3_ATR_MULT"] * row['atr'])

            open_trades.append({
                'id': trade_id_counter, 'symbol': row['symbol'], 'side': side, 'entry_price': current_price,
                'entry_time': timestamp, 'qty': qty, 'initial_qty': qty, 'notional': notional,
                'sl': stop_price, 'tp': final_tp, 'risk_usdt': risk_usdt,
                'phase': 0, 'be_moved': False, 'tp1': tp1, 'tp2': tp2
            })
            
        unrealized_pnl = sum((current_price - t['entry_price']) * t['qty'] if t['side'] == 'BUY' else (t['entry_price'] - current_price) * t['qty'] for t in open_trades)
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

    # --- Drawdown Calculation ---
    equity_curve = pd.Series(results['equity_curve'])
    running_max = equity_curve.cummax()
    drawdown = (running_max - equity_curve) / running_max
    max_drawdown_pct = drawdown.max() * 100

    # --- Plotly Charts ---
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'))
    equity_fig.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Equity (USDT)')

    drawdown_fig = go.Figure()
    drawdown_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * -100, mode='lines', name='Drawdown', fill='tozeroy', line_color='red'))
    drawdown_fig.update_layout(title='Drawdown Curve', xaxis_title='Time', yaxis_title='Drawdown (%)')

    monthly_pnl = trades_df.set_index('exit_time')['pnl'].resample('M').sum()
    monthly_fig = go.Figure(data=[go.Bar(x=monthly_pnl.index.strftime('%Y-%m'), y=monthly_pnl, name='Monthly PnL')])
    monthly_fig.update_layout(title='Monthly PnL', xaxis_title='Month', yaxis_title='PnL (USDT)')
    
    # --- HTML Tables for Trades ---
    display_cols = ['id', 'exit_time', 'exit_price', 'pnl', 'qty']
    
    # Format for display
    winning_trades_df['pnl_display'] = winning_trades_df['pnl'].map('${:,.2f}'.format)
    losing_trades_df['pnl_display'] = losing_trades_df['pnl'].map('${:,.2f}'.format)
    
    win_table_html = winning_trades_df[display_cols].to_html(classes='trade-table', index=False)
    loss_table_html = losing_trades_df[display_cols].to_html(classes='trade-table', index=False)

    # --- HTML Assembly ---
    equity_chart_html = equity_fig.to_html(full_html=False, include_plotlyjs='cdn')
    drawdown_chart_html = drawdown_fig.to_html(full_html=False, include_plotlyjs=False)
    monthly_chart_html = monthly_fig.to_html(full_html=False, include_plotlyjs=False)

    html_template = f"""
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f0f2f5; }}
            h1, h2, h3 {{ color: #333; text-align: center; }}
            .container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
            .kpi-box {{ background-color: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .kpi-title {{ font-weight: bold; font-size: 1em; color: #666; }}
            .kpi-value {{ font-size: 1.8em; font-weight: bold; margin-top: 10px; }}
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
            .chart-container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .table-container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .trade-table {{ width: 100%; border-collapse: collapse; }}
            .trade-table th, .trade-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .trade-table th {{ background-color: #f2f2f2; }}
            .trade-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Strategy Backtest Report</h1>
        <h2>Key Performance Indicators</h2>
        <div class="container">
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
                <div class="kpi-title">Win / Loss Count</div>
                <div class="kpi-value"><span class="positive">{win_count}</span> / <span class="negative">{loss_count}</span></div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Win Rate</div>
                <div class="kpi-value">{win_rate:.2f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Profit Factor</div>
                <div class="kpi-value">{profit_factor:.2f}</div>
            </div>
             <div class="kpi-box">
                <div class="kpi-title">Avg. Win / Avg. Loss</div>
                <div class="kpi-value">{risk_reward_ratio:.2f}</div>
            </div>
        </div>

        <h2>Charts</h2>
        <div class="chart-container">{equity_chart_html}</div>
        <div class="chart-container">{drawdown_chart_html}</div>
        <div class="chart-container">{monthly_chart_html}</div>

        <h2>Trade History</h2>
        <div class="table-container">
            <h3>Winning Trades</h3>
            {win_table_html}
        </div>
        <div class="table-container">
            <h3>Losing Trades</h3>
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
