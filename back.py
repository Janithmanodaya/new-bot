#!/usr/bin/env python3
"""
back.py

Combined downloader + backtester script.
 - Uses a fixed symbol list (per user request)
 - Downloads klines from Binance only if the expected CSV for the requested date range doesn't already exist
 - Runs Backtesting.py using a Bollinger-limit strategy (Version 2 rules)
 - Uses robust sizing for small accounts with leverage

Usage: python back.py
Dependencies: requests, pandas, backtesting
Install: pip install requests pandas backtesting plotly
"""

import os
import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------------------
# Config / Defaults
# ----------------------
BINANCE_REST = "https://api.binance.com"
DATA_DIR = "data"
ENV_FILE = ".env"
DEFAULT_INTERVAL = "15m"
KLIMIT = 1000
REQUEST_SLEEP = 0.25

INTERVAL_MS = {
    '1m': 60_000,
    '3m': 3*60_000,
    '5m': 5*60_000,
    '15m': 15*60_000,
    '30m': 30*60_000,
    '1h': 60*60_000,
    '2h': 2*60*60_000,
    '4h': 4*60*60_000,
    '6h': 6*60*60_000,
    '8h': 8*60*60_000,
    '12h': 12*60*60_000,
    '1d': 24*60*60_000,
    '3d': 3*24*60*60_000,
    '1w': 7*24*60*60_000,
    '1M': 30*24*60*60_000
}

# Fixed symbol list as requested
FIXED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "TRXUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "XLMUSDT",
    "TONUSDT",
    "LTCUSDT",
    "DOTUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "ETCUSDT",
    "TRUMPUSDT",
]

# ----------------------
# Helper functions (Download, Indicators - unchanged)
# ----------------------

def write_env(api_key: str, api_secret: str, symbols: list, starting_balance, days):
    symbols_str = ",".join(symbols)
    content_lines = [
        "# Auto-generated .env - keep this file safe (contains API secret)",
        f"BINANCE_API_KEY={api_key}",
        f"BINANCE_API_SECRET={api_secret}",
        f"SYMBOLS={symbols_str}",
        f"STARTING_BALANCE={starting_balance}",
        f"DATA_DAYS={days}",
    ]
    with open(ENV_FILE, 'w') as f:
        f.write("\n".join(content_lines) + "\n")
    print(f"Wrote {ENV_FILE} with {len(symbols)} symbol(s).")


def _klines_url(symbol, interval, start_time=None, end_time=None, limit=1000):
    url = BINANCE_REST + "/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    if start_time is not None:
        params['startTime'] = int(start_time)
    if end_time is not None:
        params['endTime'] = int(end_time)
    return url, params


def download_klines(symbol: str, interval: str, days: int, out_dir=DATA_DIR):
    """Download klines for `symbol` from (now - days) to now in chunks and save CSV.

    If the exact file already exists on disk, skip downloading and return its path.
    Returns path to saved CSV or None on failure.
    """
    assert interval in INTERVAL_MS, f"Unsupported interval {interval}"
    os.makedirs(out_dir, exist_ok=True)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(days * 24 * 60 * 60 * 1000)

    start_ts = datetime.utcfromtimestamp(start_ms/1000).strftime('%Y%m%d')
    end_ts = datetime.utcfromtimestamp(now_ms/1000).strftime('%Y%m%d')
    filename = f"{symbol}_{interval}_{start_ts}_{end_ts}.csv"
    path = os.path.join(out_dir, filename)
    if os.path.exists(path):
        print(f"Found existing {path}, skipping download.")
        return path

    step_ms = INTERVAL_MS[interval] * KLIMIT

    all_rows = []
    fetch_start = start_ms
    total = 0
    print(f"Downloading {symbol} from {datetime.utcfromtimestamp(start_ms/1000).isoformat()} to {datetime.utcfromtimestamp(now_ms/1000).isoformat()} ({days} days) with interval {interval}...")
    while fetch_start < now_ms:
        fetch_end = min(fetch_start + step_ms - 1, now_ms)
        url, params = _klines_url(symbol, interval, start_time=fetch_start, end_time=fetch_end, limit=KLIMIT)
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"Warning: request failed for {symbol}: {e}. Retrying after sleep...")
            time.sleep(1)
            continue
        data = r.json()
        if not data:
            break
        for row in data: # Fixed the syntax error here
            all_rows.append(row)
        total += len(data)
        last_open = data[-1][0]
        fetch_start = last_open + INTERVAL_MS[interval]
        time.sleep(REQUEST_SLEEP)

    print(f"Fetched {total} candles for {symbol}.")

    if not all_rows:
        print(f"No data for {symbol}, skipping file write.")
        return None

    df = pd.DataFrame(all_rows, columns=[
        'OpenTime','Open','High','Low','Close','Volume','CloseTime','QuoteAssetVolume','NumTrades','TakerBuyBaseVol','TakerBuyQuoteVol','Ignore'
    ])
    df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df['CloseTime'] = pd.to_datetime(df['CloseTime'], unit='ms')
    numeric_cols = ['Open','High','Low','Close','Volume','QuoteAssetVolume','TakerBuyBaseVol','TakerBuyQuoteVol']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")
    return path

# ----------------------
# Indicators / Strategy helpers
# ----------------------

def bollinger_bands(series: pd.Series, length=20, std_mul=2.0):
    ma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = ma + std_mul * std
    lower = ma - std_mul * std
    return ma, upper, lower


def atr(df: pd.DataFrame, length=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ----------------------
# Strategy (Backtesting.py)
# ----------------------
try:
    from backtesting import Strategy, Backtest
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    Strategy = None
    Backtest = None

class BollingerLimitV2(Strategy if Strategy is not None else object):
    bb_length = 20
    bb_std = 2.0
    atr_length = 14
    atr_mult = 2.5
    limit_pct = 0.005  # 0.5% limit order distance
    cooldown_hours = 6
    rr = 2.0  # Risk/Reward ratio
    be_trigger_pct = 0.006  # 0.6% for break-even trigger
    be_sl_offset = 0.001  # Move SL to 0.1% from entry

    def init(self):
        # Convert backtesting arrays to pandas Series with datetime index
        idx = self.data.index
        close = pd.Series(self.data.Close, index=idx).astype(float)
        high = pd.Series(self.data.High, index=idx).astype(float)
        low = pd.Series(self.data.Low, index=idx).astype(float)

        self.ma, self.upper, self.lower = bollinger_bands(close, self.bb_length, self.bb_std)
        ohlc_df = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
        self.atr = atr(ohlc_df, self.atr_length)

        self.pending = None
        self.cooldown_until = None

    def next(self):
        i = len(self.data.Close) - 1
        now_ts = self.data.index[i]

        # Check if pending order expired
        if self.pending is not None and i > self.pending.get('expire_idx', -1):
            self.pending = None

        # Manage open position
        if self.position:
            # Use the entry_price stored by the strategy when the trade was opened
            entry_price = self.position.entry_price
            is_long = self.position.is_long
            cur_close = float(self.data.Close[-1])

            # Calculate current PnL percentage
            if is_long:
                pnl_pct = (cur_close / entry_price) - 1.0
            else:
                pnl_pct = 1.0 - (cur_close / entry_price)

            # Break-even and trailing logic (when +0.6% profit)
            if pnl_pct >= self.be_trigger_pct:
                # Set break-even SL
                if is_long:
                    new_sl = entry_price * (1.0 + self.be_sl_offset)
                else:
                    new_sl = entry_price * (1.0 - self.be_sl_offset)

                # Only update SL if it's better than current
                if self.position.sl is None:
                    self.position.sl = new_sl
                else:
                    if is_long and new_sl > self.position.sl:
                        self.position.sl = new_sl
                    if (not is_long) and new_sl < self.position.sl:
                        self.position.sl = new_sl

                # Apply trailing stop using ATR
                atr_now = float(self.atr.iloc[-1]) if not np.isnan(self.atr.iloc[-1]) else None
                if atr_now is not None:
                    if is_long:
                        trailing_sl = cur_close - atr_now
                        if trailing_sl > (self.position.sl or 0):
                            self.position.sl = trailing_sl
                    else:
                        trailing_sl = cur_close + atr_now
                        if (self.position.sl is None) or trailing_sl < self.position.sl:
                            self.position.sl = trailing_sl

            # Partial TP when original TP hit
            tp_price = getattr(self.position, 'custom_tp', None) # Use custom TP
            if tp_price is not None:
                half_taken = getattr(self.position, 'half_taken', False)
                if not half_taken:
                    if is_long and float(self.data.High[-1]) >= tp_price:
                        self.position.half_taken = True
                        # Close 50% position
                        self.position.close(portion=0.5)
                    elif (not is_long) and float(self.data.Low[-1]) <= tp_price:
                        self.position.half_taken = True
                        self.position.close(portion=0.5)

        # If no open position, handle pending fills or create new pending
        if not self.position:
            # Check pending fill on this bar
            if self.pending is not None:
                pend = self.pending
                high = float(self.data.High[-1])
                low = float(self.data.Low[-1])
                filled = False

                if pend['type'] == 'long':
                    if low <= pend['limit_price'] <= high:
                        filled = True
                        entry_price = pend['limit_price']
                        self._enter_trade_long(entry_price)
                else:  # short
                    if low <= pend['limit_price'] <= high:
                        filled = True
                        entry_price = pend['limit_price']
                        self._enter_trade_short(entry_price)

                if filled:
                    self.pending = None

            # Create new pending orders if no position and no pending order
            if self.pending is None:
                # Check cooldown period
                if self.cooldown_until is not None and now_ts < self.cooldown_until:
                    return

                close = float(self.data.Close[-1])
                upper = float(self.upper.iloc[-1]) if not np.isnan(self.upper.iloc[-1]) else None
                lower = float(self.lower.iloc[-1]) if not np.isnan(self.lower.iloc[-1]) else None

                if upper is None or lower is None:
                    return

                # Entry conditions
                if close >= upper:  # Short signal - price touches/crosses upper band
                    limit_price = close * (1.0 - self.limit_pct)  # 0.5% below close
                    self.pending = {
                        'type': 'short',
                        'limit_price': limit_price,
                        'expire_idx': i + 1,  # Expires after 1 candle
                        'entry_close': close
                    }
                elif close <= lower:  # Long signal - price touches/crosses lower band
                    limit_price = close * (1.0 + self.limit_pct)  # 0.5% above close
                    self.pending = {
                        'type': 'long',
                        'limit_price': limit_price,
                        'expire_idx': i + 1,  # Expires after 1 candle
                        'entry_close': close
                    }

    def _calculate_risk_amount(self):
        """Calculate risk amount based on account balance"""
        equity = self.equity
        if equity < 50:
            return 0.5  # Fixed 0.5 USDT risk
        else:
            return equity * 0.02  # 2% of account balance

    def _enter_trade_long(self, entry_price):
        atr_now = float(self.atr.iloc[-1]) if not np.isnan(self.atr.iloc[-1]) else None
        if atr_now is None:
            return

        # SL = entry - (2.5 * ATR)
        sl_price = entry_price - (self.atr_mult * atr_now)
        
        # Validate SL calculation
        if sl_price >= entry_price or sl_price <= 0:
            return
            
        risk_per_unit = entry_price - sl_price

        # Validate risk calculation
        if risk_per_unit <= 0:
            return

        # Calculate risk amount
        risk_amount = self._calculate_risk_amount()

        # Avoid division by zero or negative risk
        if risk_amount <= 0:
            return

        # --- Simplified and Robust Size Calculation ---
        # 1. Estimate the risk percentage based on SL distance
        try:
            sl_pct_risk = risk_per_unit / entry_price
        except:
            return # Avoid division by zero if entry_price is somehow 0

        if sl_pct_risk <= 0 or sl_pct_risk > 1: # Sanity check: SL risk should be small and positive
            return

        # 2. Calculate the maximum position value we are willing to have based on risk
        #    Position Value * SL% = Risk Amount
        #    Position Value = Risk Amount / SL%
        try:
            max_position_value = risk_amount / sl_pct_risk
        except:
            return # Avoid division by zero

        # 3. Calculate the maximum size based on this value and entry price
        try:
            max_size = max_position_value / entry_price
        except:
            return # Avoid division by zero

        # 4. Use a very conservative fraction of this max size to ensure sufficient margin
        #    This is key to avoiding "insufficient margin" errors.
        conservative_fraction = 0.1 # Use 10% of calculated max size
        final_size = max_size * conservative_fraction

        # 5. Ensure the final size is a valid, positive number
        if not (final_size > 0 and np.isfinite(final_size)):
            return

        # 6. Round to a reasonable number of decimals to avoid precision issues
        final_size = round(final_size, 8)

        # --- End Simplified Size Calculation ---

        # TP = entry + (2 * risk) for 1:2 R/R
        tp_price = entry_price + (self.rr * risk_per_unit)

        # Place the trade
        try:
            # Place the order with the conservative size and SL
            order = self.buy(size=final_size, sl=sl_price)

            # Store additional trade info in the position object
            if self.position:
                self.position.custom_tp = tp_price  # Custom TP attribute
                self.position.half_taken = False
                # entry_price is already set by the backtesting framework

        except Exception as e:
            print(f"Warning: Failed to enter long trade (size={final_size:.8f}, value={final_size*entry_price:.2f}): {type(e).__name__}")

    def _enter_trade_short(self, entry_price):
        atr_now = float(self.atr.iloc[-1]) if not np.isnan(self.atr.iloc[-1]) else None
        if atr_now is None:
            return

        # SL = entry + (2.5 * ATR)
        sl_price = entry_price + (self.atr_mult * atr_now)
        
         # Validate SL calculation
        if sl_price <= entry_price or sl_price <= 0:
            return
            
        risk_per_unit = sl_price - entry_price

        # Validate risk calculation
        if risk_per_unit <= 0:
            return

        # Calculate risk amount
        risk_amount = self._calculate_risk_amount()

        # Avoid division by zero or negative risk
        if risk_amount <= 0:
            return

        # --- Simplified and Robust Size Calculation ---
        # 1. Estimate the risk percentage based on SL distance
        try:
            sl_pct_risk = risk_per_unit / entry_price
        except:
            return # Avoid division by zero if entry_price is somehow 0

        if sl_pct_risk <= 0 or sl_pct_risk > 1: # Sanity check
            return

        # 2. Calculate the maximum position value
        try:
            max_position_value = risk_amount / sl_pct_risk
        except:
            return # Avoid division by zero

        # 3. Calculate the maximum size
        try:
            max_size = max_position_value / entry_price
        except:
            return # Avoid division by zero

        # 4. Use a very conservative fraction
        conservative_fraction = 0.1 # Use 10% of calculated max size
        final_size = max_size * conservative_fraction

        # 5. Ensure the final size is valid
        if not (final_size > 0 and np.isfinite(final_size)):
            return

        # 6. Round
        final_size = round(final_size, 8)

        # --- End Simplified Size Calculation ---

        # TP = entry - (2 * risk) for 1:2 R/R
        tp_price = entry_price - (self.rr * risk_per_unit)

        # Place the trade
        try:
            # Place the order with the conservative size and SL
            order = self.sell(size=final_size, sl=sl_price)

            # Store additional trade info
            if self.position:
                self.position.custom_tp = tp_price  # Custom TP attribute
                self.position.half_taken = False
                # entry_price is already set

        except Exception as e:
            print(f"Warning: Failed to enter short trade (size={final_size:.8f}, value={final_size*entry_price:.2f}): {type(e).__name__}")

    def on_trade_close(self, trade):
        # Apply cooldown only for losing trades
        if hasattr(trade, '_pl') and trade._pl < 0:
            last_idx = len(self.data.Close) - 1
            last_ts = self.data.index[last_idx]
            self.cooldown_until = last_ts + pd.Timedelta(hours=self.cooldown_hours)

# ----------------------
# Plotting Function (Placeholder - include full function from previous code)
# ----------------------
def create_enhanced_plot(df, stats, symbol, results_dir):
    """Create enhanced plot with trade markers and SL/TP levels"""
    # Include the full plotting function from the previous working version
    try:
        # Extract trade data
        trades = stats['_trades']
        if trades.empty:
            return None

        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} Price Action with Trade Markers', 'Volume')
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Upper'],
                mode='lines',
                name='Upper BB',
                line=dict(color='rgba(100, 100, 200, 0.7)', width=1),
                hovertemplate='Upper BB: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Lower'],
                mode='lines',
                name='Lower BB',
                line=dict(color='rgba(100, 100, 200, 0.7)', width=1),
                hovertemplate='Lower BB: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA'],
                mode='lines',
                name='MA',
                line=dict(color='rgba(200, 200, 100, 0.7)', width=1),
                hovertemplate='MA: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add volume bars
        colors = ['rgba(44, 160, 44, 0.7)' if df['Close'][i] >= df['Open'][i]
                 else 'rgba(214, 39, 40, 0.7)' for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

        # Add trade markers
        long_entries = []
        short_entries = []
        exits = []
        sl_levels = []
        tp_levels = []

        for _, trade in trades.iterrows():
            entry_idx = trade['EntryBar']
            exit_idx = trade['ExitBar']

            if entry_idx < len(df) and exit_idx < len(df):
                entry_time = df.index[entry_idx]
                exit_time = df.index[exit_idx]
                entry_price = trade['EntryPrice']
                exit_price = trade['ExitPrice']

                # Entry markers
                if trade['Size'] > 0:  # Long
                    long_entries.append(dict(
                        x=entry_time,
                        y=entry_price,
                        text=f"LONG<br>Size: {trade['Size']:.4f}<br>Entry: {entry_price:.4f}",
                        color='green'
                    ))
                else:  # Short
                    short_entries.append(dict(
                        x=entry_time,
                        y=entry_price,
                        text=f"SHORT<br>Size: {abs(trade['Size']):.4f}<br>Entry: {entry_price:.4f}",
                        color='red'
                    ))

                # Exit markers
                exits.append(dict(
                    x=exit_time,
                    y=exit_price,
                    text=f"EXIT<br>P/L: {trade['PnL']:.2f}<br>Return: {trade['ReturnPct']:.2f}%",
                    color='orange' if trade['PnL'] > 0 else 'purple'
                ))

                # SL/TP levels (if available)
                if 'SL' in trade and not np.isnan(trade['SL']):
                    sl_levels.append(dict(
                        x=entry_time,
                        y=trade['SL'],
                        text=f"SL: {trade['SL']:.4f}",
                        color='red'
                    ))

                if 'TP' in trade and not np.isnan(trade['TP']):
                    tp_levels.append(dict(
                        x=entry_time,
                        y=trade['TP'],
                        text=f"TP: {trade['TP']:.4f}",
                        color='green'
                    ))

        # Add entry markers
        if long_entries:
            fig.add_trace(
                go.Scatter(
                    x=[t['x'] for t in long_entries],
                    y=[t['y'] for t in long_entries],
                    mode='markers',
                    name='Long Entries',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='white')
                    ),
                    text=[t['text'] for t in long_entries],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

        if short_entries:
            fig.add_trace(
                go.Scatter(
                    x=[t['x'] for t in short_entries],
                    y=[t['y'] for t in short_entries],
                    mode='markers',
                    name='Short Entries',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='white')
                    ),
                    text=[t['text'] for t in short_entries],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add exit markers
        if exits:
            fig.add_trace(
                go.Scatter(
                    x=[t['x'] for t in exits],
                    y=[t['y'] for t in exits],
                    mode='markers',
                    name='Exits',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color=[t['color'] for t in exits],
                        line=dict(width=2, color='white')
                    ),
                    text=[t['text'] for t in exits],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add SL/TP markers
        if sl_levels:
            fig.add_trace(
                go.Scatter(
                    x=[t['x'] for t in sl_levels],
                    y=[t['y'] for t in sl_levels],
                    mode='markers',
                    name='Stop Loss',
                    marker=dict(
                        symbol='x',
                        size=8,
                        color='red'
                    ),
                    text=[t['text'] for t in sl_levels],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

        if tp_levels:
            fig.add_trace(
                go.Scatter(
                    x=[t['x'] for t in tp_levels],
                    y=[t['y'] for t in tp_levels],
                    mode='markers',
                    name='Take Profit',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='green'
                    ),
                    text=[t['text'] for t in tp_levels],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )

        # Update layout
        fig.update_layout(
            title=f'{symbol} - Enhanced Backtest Results<br>' +
                  f'Return: {stats["Return [%]"]:.2f}% | Trades: {int(stats["# Trades"])} | ' +
                  f'Win Rate: {stats["Win Rate [%]"]:.1f}% | Max DD: {stats["Max. Drawdown [%]"]:.2f}%',
            template='plotly_dark',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Save plot
        plot_path = os.path.join(results_dir, f"{symbol}_enhanced_plot.html")
        fig.write_html(plot_path)
        return plot_path

    except Exception as e:
        print(f"Error creating enhanced plot for {symbol}: {e}")
        return None

# ----------------------
# Report Generation (Placeholder - include full function from previous code)
# ----------------------
def generate_dark_html_report(results, starting_balance, commission=0.0001):
    """Generate a single dark-themed HTML report with all details"""
    # Include the full report generation function from the previous working version
    if not results:
        return "<h1>No results to report</h1>"

    res_df = pd.DataFrame(results).set_index('Symbol')

    # Calculate overall metrics
    total_symbols = len(res_df)
    total_initial_capital = starting_balance * total_symbols
    total_final_equity = res_df['Equity Final [$]'].sum()
    total_profit = total_final_equity - total_initial_capital
    total_return_pct = ((total_final_equity / total_initial_capital) - 1) * 100 if total_initial_capital > 0 else 0
    avg_return_pct = res_df['Return [%]'].mean()
    total_trades = res_df['# Trades'].sum()
    avg_win_rate = res_df['Win Rate [%]'].mean()
    max_dd = res_df['Max. Drawdown [%]'].min() if 'Max. Drawdown [%]' in res_df.columns else 0
    avg_max_dd = res_df['Max. Drawdown [%]'].mean() if 'Max. Drawdown [%]' in res_df.columns else 0

    # Risk metrics
    profitable_symbols = len(res_df[res_df['Return [%]'] > 0])
    unprofitable_symbols = len(res_df[res_df['Return [%]'] < 0])
    break_even_symbols = len(res_df[res_df['Return [%]'] == 0])

    # Sort symbols by performance
    top_performers = res_df.sort_values('Return [%]', ascending=False).head(10)
    worst_performers = res_df.sort_values('Return [%]', ascending=True).head(10)

    # Commission info
    total_commission_paid = res_df['# Trades'].sum() * commission * starting_balance * 2  # Rough estimate

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backtesting Report</title>
        <style>
            :root {{
                --bg-dark: #1e1e2e;
                --bg-card: #2d2d44;
                --text-primary: #e0e0e0;
                --text-secondary: #a0a0c0;
                --accent-primary: #bb86fc;
                --accent-secondary: #03dac6;
                --success: #4caf50;
                --warning: #ff9800;
                --danger: #f44336;
                --border: #444466;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: var(--bg-dark);
                color: var(--text-primary);
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}

            header {{
                text-align: center;
                padding: 20px 0;
                border-bottom: 2px solid var(--border);
                margin-bottom: 30px;
            }}

            h1, h2, h3 {{
                color: var(--accent-primary);
                margin-top: 0;
            }}

            .summary-cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}

            .card {{
                background: var(--bg-card);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                border: 1px solid var(--border);
            }}

            .card-title {{
                font-size: 1rem;
                color: var(--text-secondary);
                margin-bottom: 10px;
            }}

            .card-value {{
                font-size: 1.8rem;
                font-weight: bold;
                margin: 0;
            }}

            .positive {{
                color: var(--success);
            }}

            .negative {{
                color: var(--danger);
            }}

            .neutral {{
                color: var(--warning);
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: var(--bg-card);
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 30px;
            }}

            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid var(--border);
            }}

            th {{
                background-color: rgba(187, 134, 252, 0.1);
                color: var(--accent-primary);
                font-weight: 600;
            }}

            tr:hover {{
                background-color: rgba(255, 255, 255, 0.05);
            }}

            .section {{
                margin-bottom: 40px;
            }}

            .symbol-performance {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}

            @media (max-width: 768px) {{
                .symbol-performance {{
                    grid-template-columns: 1fr;
                }}

                .summary-cards {{
                    grid-template-columns: 1fr 1fr;
                }}
            }}

            @media (max-width: 480px) {{
                .summary-cards {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>📊 Backtesting Report</h1>
                <p>Strategy: Bollinger Band Limit Orders with Dynamic Risk Management</p>
            </header>

            <div class="summary-cards">
                <div class="card">
                    <div class="card-title">Total Symbols</div>
                    <h2 class="card-value">{total_symbols}</h2>
                </div>
                <div class="card">
                    <div class="card-title">Total Return</div>
                    <h2 class="card-value {('positive' if total_return_pct > 0 else 'negative' if total_return_pct < 0 else 'neutral')}">
                        {total_return_pct:.2f}%
                    </h2>
                </div>
                <div class="card">
                    <div class="card-title">Total Profit</div>
                    <h2 class="card-value {('positive' if total_profit > 0 else 'negative' if total_profit < 0 else 'neutral')}">
                        ${total_profit:.2f}
                    </h2>
                </div>
                <div class="card">
                    <div class="card-title">Total Trades</div>
                    <h2 class="card-value">{int(total_trades)}</h2>
                </div>
                <div class="card">
                    <div class="card-title">Avg Win Rate</div>
                    <h2 class="card-value {('positive' if avg_win_rate > 50 else 'negative')}">
                        {avg_win_rate:.1f}%
                    </h2>
                </div>
                <div class="card">
                    <div class="card-title">Avg Max Drawdown</div>
                    <h2 class="card-value negative">
                        {avg_max_dd:.2f}%
                    </h2>
                </div>
            </div>

            <div class="section">
                <h2>📈 Overall Performance</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Starting Balance per Symbol</td>
                        <td>${starting_balance:.2f}</td>
                    </tr>
                    <tr>
                        <td>Total Initial Capital</td>
                        <td>${total_initial_capital:.2f}</td>
                    </tr>
                    <tr>
                        <td>Total Final Equity</td>
                        <td>${total_final_equity:.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Return per Symbol</td>
                        <td>{avg_return_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Profitable Symbols</td>
                        <td class="positive">{profitable_symbols}</td>
                    </tr>
                    <tr>
                        <td>Unprofitable Symbols</td>
                        <td class="negative">{unprofitable_symbols}</td>
                    </tr>
                    <tr>
                        <td>Break-even Symbols</td>
                        <td class="neutral">{break_even_symbols}</td>
                    </tr>
                    <tr>
                        <td>Estimated Commission Paid</td>
                        <td>${total_commission_paid:.2f}</td>
                    </tr>
                </table>
            </div>

            <div class="symbol-performance">
                <div class="section">
                    <h2>🏆 Top Performers</h2>
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Return %</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                        </tr>"""

    # Add top performers
    for idx, row in top_performers.iterrows():
        win_rate = row['Win Rate [%]'] if 'Win Rate [%]' in row else 0
        html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td class="{'positive' if row['Return [%]'] > 0 else 'negative' if row['Return [%]'] < 0 else 'neutral'}">
                                {row['Return [%]']:.2f}%
                            </td>
                            <td>{int(row['# Trades'])}</td>
                            <td>{win_rate:.1f}%</td>
                        </tr>"""

    html_content += f"""
                    </table>
                </div>

                <div class="section">
                    <h2>📉 Worst Performers</h2>
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Return %</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                        </tr>"""

    # Add worst performers
    for idx, row in worst_performers.iterrows():
        win_rate = row['Win Rate [%]'] if 'Win Rate [%]' in row else 0
        html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td class="{'positive' if row['Return [%]'] > 0 else 'negative' if row['Return [%]'] < 0 else 'neutral'}">
                                {row['Return [%]']:.2f}%
                            </td>
                            <td>{int(row['# Trades'])}</td>
                            <td>{win_rate:.1f}%</td>
                        </tr>"""

    html_content += f"""
                    </table>
                </div>
            </div>

            <div class="section">
                <h2>📋 All Symbol Results</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Return %</th>
                        <th>Final Equity</th>
                        <th># Trades</th>
                        <th>Win Rate</th>
                        <th>Max DD %</th>
                        <th>Profit Factor</th>
                    </tr>"""

    # Add all symbols
    for idx, row in res_df.iterrows():
        win_rate = row['Win Rate [%]'] if 'Win Rate [%]' in row else 0
        max_dd = row['Max. Drawdown [%]'] if 'Max. Drawdown [%]' in row else 0
        profit_factor = row['Profit Factor'] if 'Profit Factor' in row else 0

        html_content += f"""
                    <tr>
                        <td>{idx}</td>
                        <td class="{'positive' if row['Return [%]'] > 0 else 'negative' if row['Return [%]'] < 0 else 'neutral'}">
                            {row['Return [%]']:.2f}%
                        </td>
                        <td>${row['Equity Final [$]']:.2f}</td>
                        <td>{int(row['# Trades'])}</td>
                        <td>{win_rate:.1f}%</td>
                        <td class="negative">{max_dd:.2f}%</td>
                        <td>{profit_factor:.2f}</td>
                    </tr>"""

    html_content += f"""
                </table>
            </div>

            <footer style="text-align: center; padding: 20px; color: var(--text-secondary); border-top: 1px solid var(--border); margin-top: 30px;">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Strategy: Bollinger Band Limit Orders with Dynamic Risk Management</p>
            </footer>
        </div>
    </body>
    </html>"""

    return html_content


# ----------------------
# Main: downloader + backtests
# ----------------------

def main():
    print("=== Binance Data Downloader + Backtester ===")

    existing_env = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k,v = line.split('=',1)
                    existing_env[k.strip()] = v.strip()

    api_key = existing_env.get('BINANCE_API_KEY') or input('Enter your BINANCE_API_KEY (leave blank to skip): ').strip()
    api_secret = existing_env.get('BINANCE_API_SECRET') or input('Enter your BINANCE_API_SECRET (leave blank to skip): ').strip()

    symbols = FIXED_SYMBOLS
    print(f"Using fixed symbol list with {len(symbols)} symbols.")

    # prompt for day count and starting balance
    while True:
        days_input = input('How many past days of data to download per symbol? (e.g. 7): ').strip()
        try:
            days = int(days_input)
            if days <= 0:
                raise ValueError()
            break
        except Exception:
            print('Please enter a positive integer for days.')

    while True:
        bal_input = input('Enter starting balance (e.g. 10000): ').strip()
        try:
            starting_balance = float(bal_input)
            if starting_balance <= 0:
                 print('Please enter a positive starting balance.')
                 continue
            break
        except Exception:
            print('Please enter a numeric starting balance.')

    write_env(api_key, api_secret, symbols, starting_balance, days)

    interval = input(f"Interval to download (e.g. {DEFAULT_INTERVAL}): ").strip() or DEFAULT_INTERVAL
    if interval not in INTERVAL_MS:
        print(f"Interval {interval} not supported. Supported intervals: {', '.join(list(INTERVAL_MS.keys()))}")
        return

    print("Beginning downloads... (this may take a while)")
    downloaded = []
    for idx, sym in enumerate(symbols, start=1):
        try:
            print(f"[{idx}/{len(symbols)}] -> {sym}")
            path = download_klines(sym, interval, days)
            if path:
                downloaded.append(path)
        except KeyboardInterrupt:
            print('Interrupted by user, stopping downloads.')
            break
        except Exception as e:
            print(f"Error downloading {sym}: {e}")
            time.sleep(1)

    print('Done. Downloaded files:')
    for p in downloaded:
        print('  ', p)

    # Run backtests (if backtesting available)
    if Backtest is None:
        print('\nWARNING: backtesting.py not installed. To run backtests, install with: pip install backtesting plotly')
        print('All finished.')
        return

    results = []
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    print('\nStarting backtests on downloaded files...')
    print('Using fixed 10x leverage (margin=0.1) for all symbols.')

    for path in downloaded:
        try:
            sym = os.path.basename(path).split('_')[0]
            print(f'\n--- Backtest: {sym} ---')
            df = pd.read_csv(path, parse_dates=['OpenTime'])
            df = df.rename(columns={
                'OpenTime': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            df = df.set_index('Date')
            # ensure numeric
            for c in ['Open','High','Low','Close','Volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            # Add indicators to dataframe for plotting
            ma, upper, lower = bollinger_bands(df['Close'], 20, 2.0)
            df['MA'] = ma
            df['Upper'] = upper
            df['Lower'] = lower

            # --- FIXED LEVERAGE ---
            # Run backtest with 0.01% commission (0.0001), FIXED 10x leverage (margin=0.1)
            bt = Backtest(df, BollingerLimitV2, cash=starting_balance, commission=0.0001,
                         trade_on_close=False, exclusive_orders=True, margin=0.1) # 10x leverage
            stats = bt.run()

            print_stats = {
                'Return [%]': stats.get('Return [%]', 0),
                'Equity Final [$]': stats.get('Equity Final [$]', starting_balance),
                '# Trades': stats.get('# Trades', 0),
                'Win Rate [%]': stats.get('Win Rate [%]', 0),
                'Max. Drawdown [%]': stats.get('Max. Drawdown [%]', 0),
                'Profit Factor': stats.get('Profit Factor', 1)
            }

            print(f"Return [%]: {print_stats['Return [%]']:.2f} | Equity Final [$]: {print_stats['Equity Final [$]']:.2f} | # Trades: {int(print_stats['# Trades'])} | Win Rate [%]: {print_stats['Win Rate [%]']:.2f} | Max DD [%]: {print_stats['Max. Drawdown [%]']:.2f}")

            # Create enhanced plot with trade markers (only if trades occurred)
            if int(print_stats['# Trades']) > 0:
                plot_path = create_enhanced_plot(df, stats, sym, results_dir)
                if plot_path:
                    print(f"Saved enhanced plot to {plot_path}")
            else:
                print("No trades executed, skipping plot generation.")

            sdict = {k: stats[k] for k in stats.index}
            sdict['Symbol'] = sym
            results.append(sdict)

        except Exception as e:
            print(f"Backtest failed for {path}: {e}")

    if results:
        # Generate HTML report
        html_report = generate_dark_html_report(results, starting_balance, commission=0.0001)
        report_path = os.path.join(results_dir, 'backtest_report.html')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f'\nSaved comprehensive HTML report to {report_path}')

        # Save CSV summary
        res_df = pd.DataFrame(results).set_index('Symbol')
        summary_path = os.path.join(results_dir, 'backtests_summary.csv')
        res_df.to_csv(summary_path)
        print(f'Saved CSV summary to {summary_path}')

    print('All finished.')

if __name__ == '__main__':
    main()
