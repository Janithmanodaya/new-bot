import os
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_CEILING
from strategy import *
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "data"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- Exchange Info Helpers ---
symbol_info_cache = {}

def get_symbol_info_once(symbol):
    """
    Fetches and caches symbol information from Binance.
    This is to avoid repeated API calls for the same symbol info.
    """
    if symbol in symbol_info_cache:
        return symbol_info_cache[symbol]
    
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, tld='us')
    try:
        exchange_info = client.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                symbol_info_cache[symbol] = s
                return s
    except Exception as e:
        print(f"Could not fetch exchange info: {e}")
    return None

def get_step_size(symbol: str, symbol_info: dict) -> Decimal:
    """Extracts the lot step size from the symbol information."""
    if not symbol_info: return Decimal('1')
    for f in symbol_info.get('filters', []):
        if f.get('filterType') == 'LOT_SIZE':
            return Decimal(str(f.get('stepSize', '1')))
    return Decimal('1')

def round_qty(qty: float, step_size: Decimal) -> float:
    """Rounds a quantity down to the nearest valid step size."""
    if step_size == 0: return float(qty)
    qty_decimal = Decimal(str(qty))
    rounded_qty = (qty_decimal / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
    return float(rounded_qty)

def fetch_and_cache_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical k-line data from Binance and caches it locally.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{DATA_DIR}/{symbol.upper()}-{timeframe}-{start_date}-{end_date}.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    print(f"Fetching new data for {symbol} ({timeframe}) from {start_date} to {end_date}")
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, tld='us')
    try:
        klines = client.get_historical_klines(symbol, timeframe, start_date, end_date)
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
    if not klines: return pd.DataFrame()
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.to_csv(filename)
    print(f"Data for {symbol} saved to {filename}")
    return df

class Backtester:
    """
    A class to run a vectorized backtest for a given trading strategy.
    It handles data loading, trade execution simulation, risk management, and performance reporting.
    """
    def __init__(self, symbol, start_date, end_date, initial_capital, data_5m, data_15m, data_1h, commission=0.0004, entry_threshold=2.8):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position = None
        self.trades = []
        self.commission = commission
        self.entry_threshold = entry_threshold
        self.data_5m, self.data_15m, self.data_1h = data_5m, data_15m, data_1h
        if not self.data_15m.empty: self.data_15m['atr'] = atr(self.data_15m, 15)
        self.symbol_info = get_symbol_info_once(self.symbol)
        self.step_size = get_step_size(self.symbol, self.symbol_info)
        self.current_time = None
        self.ob_cooldown = {}
        self.cooldown_period = pd.Timedelta(hours=1)

    def calculate_risk_amount(self):
        """Calculates the amount to risk on a trade based on the current balance."""
        if self.balance < 50.0: return 0.5
        else: return self.balance * 0.02

    def calculate_position_size(self, entry_price, sl_price):
        """Calculates the position size based on risk amount and stop loss distance."""
        risk_amount_usdt = self.calculate_risk_amount()
        price_distance = abs(entry_price - sl_price)
        if price_distance == 0: return 0
        position_size = risk_amount_usdt / price_distance
        return round_qty(position_size, self.step_size)

    def execute_trade(self, side, entry_price, sl_price, tp1, tp2, order_block):
        """Simulates opening a new trade."""
        size = self.calculate_position_size(entry_price, sl_price)
        if size <= 0: return

        if order_block:
            ob_key = (order_block['origin_idx'], tuple(order_block['zone']))
            self.ob_cooldown[ob_key] = self.current_time

        notional_value = size * entry_price
        commission_cost = notional_value * self.commission
        self.balance -= commission_cost
        self.position = {'side': side, 'entry_price': entry_price, 'size': size, 'sl': sl_price, 'tp1': tp1, 'tp2': tp2, 'notional': notional_value, 'open_time': self.current_time, 'be_moved': False, 'trailing_active': False}
        print(f"[{self.current_time}] OPEN {side} | Size: {size:.4f} | Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f}")

    def manage_position(self):
        """Manages the currently open position, checking for SL/TP hits and trailing SL."""
        if not self.position: return
        candle = self.data_5m.loc[self.current_time]
        if self.position['side'] == 'buy':
            if candle['low'] <= self.position['sl']: self.close_trade(self.position['sl'], 'SL')
            elif candle['high'] >= self.position['tp2']: self.close_trade(self.position['tp2'], 'TP2')
        else:
            if candle['high'] >= self.position['sl']: self.close_trade(self.position['sl'], 'SL')
            elif candle['low'] <= self.position['tp2']: self.close_trade(self.position['tp2'], 'TP2')
        if not self.position: return

        hist_1h = self.data_1h[self.data_1h.index <= self.current_time]
        if len(hist_1h) > 50:
            htf_bias = detect_htf_market_structure(hist_1h)['side']
            if (self.position['side'] == 'buy' and htf_bias == 'sell') or (self.position['side'] == 'sell' and htf_bias == 'buy'):
                self.close_trade(candle['close'], 'HTF_FLIP'); return
        if not self.position['be_moved']:
            tp1_dist = abs(self.position['tp1'] - self.position['entry_price'])
            if tp1_dist > 0:
                if (self.position['side'] == 'buy' and (candle['high'] - self.position['entry_price']) >= 0.8 * tp1_dist) or \
                   (self.position['side'] == 'sell' and (self.position['entry_price'] - candle['low']) >= 0.8 * tp1_dist):
                    self.position['sl'] = self.position['entry_price']; self.position['be_moved'] = True
        if not self.position['trailing_active']:
            if (self.position['side'] == 'buy' and candle['high'] >= self.position['tp1']) or (self.position['side'] == 'sell' and candle['low'] <= self.position['tp1']):
                self.position['trailing_active'] = True
        if self.position['trailing_active']:
            atr_15m = self.data_15m.asof(self.current_time)['atr']
            if pd.notna(atr_15m):
                trail_dist = atr_15m * 0.8
                if self.position['side'] == 'buy':
                    new_sl = candle['high'] - trail_dist
                    if new_sl > self.position['sl']: self.position['sl'] = new_sl
                else:
                    new_sl = candle['low'] + trail_dist
                    if new_sl < self.position['sl']: self.position['sl'] = new_sl

    def close_trade(self, exit_price, reason):
        """Simulates closing the current trade and records the result."""
        pnl = (exit_price - self.position['entry_price']) * self.position['size'] if self.position['side'] == 'buy' else (self.position['entry_price'] - exit_price) * self.position['size']
        commission_cost = self.position['notional'] * self.commission
        self.balance += pnl - commission_cost
        trade_log = {**self.position, 'exit_price': exit_price, 'pnl': pnl, 'exit_reason': reason, 'close_time': self.current_time}
        self.trades.append(trade_log)
        print(f"[{self.current_time}] CLOSE {self.position['side']} | Exit: {exit_price:.2f} | PnL: {pnl:.2f} | Balance: {self.balance:.2f} | Reason: {reason}")
        self.position = None

    def generate_report(self):
        """Generates and prints a performance report and saves a PnL chart."""
        print("\n--- Backtest Report ---")
        if not self.trades:
            print("No trades were executed."); return

        trades_df = pd.DataFrame(self.trades); trades_df['close_time'] = pd.to_datetime(trades_df['close_time']); trades_df.set_index('close_time', inplace=True)
        total_pnl = trades_df['pnl'].sum()
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = winning_trades['pnl'].sum() / abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) > 0 else float('inf')
        cumulative_pnl = trades_df['pnl'].cumsum()
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
        sharpe_ratio = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(252 * 288) if trades_df['pnl'].std() > 0 else 0.0

        print(f"Initial Capital: {self.initial_capital:.2f}\nFinal Balance: {self.balance:.2f}\nTotal PnL: {total_pnl:.2f}")
        print(f"Total Trades: {total_trades}\nWin Rate: {win_rate:.2f}%\nProfit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}\nSharpe Ratio: {sharpe_ratio:.2f}")
        
        plt.figure(figsize=(12, 6)); (self.initial_capital + cumulative_pnl).plot(); plt.title('Portfolio Equity Curve')
        plt.xlabel('Date'); plt.ylabel('Equity (USDT)'); plt.grid(True); plt.savefig('pnl_curve.png')
        print("\nChart saved to pnl_curve.png")

    def run(self):
        """Runs the main backtesting loop."""
        print("Running backtest...");
        if self.data_5m.empty: return
        for i in range(1, len(self.data_5m)):
            self.current_time = self.data_5m.index[i]
            if self.position: self.manage_position(); continue
            hist_5m, hist_15m, hist_1h = self.data_5m.iloc[:i+1], self.data_15m[self.data_15m.index <= self.current_time], self.data_1h[self.data_1h.index <= self.current_time]
            if len(hist_1h) < 60 or len(hist_15m) < 110 or len(hist_5m) < 70: continue
            
            detection_results = {}
            htf_market_bias = detect_htf_market_structure(hist_1h); detection_results['htf_market_bias'] = htf_market_bias
            order_blocks = find_order_blocks(hist_15m, '15m'); last_ob = order_blocks[-1] if order_blocks else None
            ob_score = 1.0 if last_ob and last_ob.get('type') == htf_market_bias.get('side') else 0.0
            detection_results['order_block'] = {'score': ob_score, 'side': htf_market_bias.get('side'), 'meta': last_ob}
            fvgs = find_fair_value_gaps(hist_15m, '15m'); last_fvg = fvgs[-1] if fvgs else None
            fvg_score = 1.0 if last_fvg and last_fvg.get('side') == htf_market_bias.get('side') else 0.0
            detection_results['fvg'] = {'score': fvg_score, 'side': htf_market_bias.get('side'), 'meta': last_fvg}
            detection_results.update({'liquidity_sweep': detect_liquidity_sweep(hist_5m), 'ote': compute_ote(hist_15m), 'breaker': detect_breaker_blocks(hist_15m, '15m'), 'vwap_hvn': institutional_imbalance_vwap(hist_15m), 'momentum': momentum_volume_filter(hist_5m, '5m'), 'candle_body_confirm': candle_body_confirmation(hist_5m.iloc[-1], last_ob['zone'] if last_ob else None), 'session_spread': session_and_spread_filter(self.current_time)})
            
            composite_score = compute_confluence_score(detection_results)['composite_score']
            if composite_score >= self.entry_threshold:
                side = detection_results['htf_market_bias']['side']
                if side != 'neutral':
                    # --- Strengthened Entry Logic ---
                    if detection_results['htf_market_bias']['score'] == 0:
                        continue
                    if detection_results['momentum']['score'] == 0:
                        continue

                    last_ob = detection_results['order_block']['meta']
                    if last_ob:
                        ob_key = (last_ob['origin_idx'], tuple(last_ob['zone']))
                        if ob_key in self.ob_cooldown and (self.current_time - self.ob_cooldown[ob_key] < self.cooldown_period):
                            continue

                    atr_15m = self.data_15m.asof(self.current_time)['atr']
                    plan = plan_entry_action(self.symbol, side, composite_score, detection_results, hist_5m.iloc[-1]['close'], atr_15m)
                    self.execute_trade(side, plan['entry_price'], plan['stop_loss'], plan['targets'][0], plan['targets'][1], last_ob)
        print("Backtest finished.")
        self.generate_report()

if __name__ == "__main__":
    """
    Main execution block to run the backtest.
    
    --- How to Use ---
    1. Configure the parameters below (symbol, dates, capital).
    2. Ensure you have your Binance API keys set as environment variables if you need to download new data.
    3. Run the script: `python backtest.py`
    4. The backtest report and PnL curve chart will be generated.
    """
    # --- Configuration ---
    symbol_to_test = "BTCUSDT"
    start_date_str = "2025-08-01"
    end_date_str = "2025-08-11"
    initial_capital = 100.0
    
    print("Backtesting framework starting...")
    print("Ensuring data is available...")
    data_to_load = {f'data_{tf}': fetch_and_cache_data(symbol_to_test, tf, start_date_str, end_date_str) for tf in ['5m', '15m', '1h']}
    if not any(df.empty for df in data_to_load.values()):
        backtester = Backtester(symbol_to_test, start_date_str, end_date_str, initial_capital, **data_to_load)
        backtester.run()
    else:
        print("Could not load data, aborting backtest.")
