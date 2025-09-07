import os
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_CEILING
from strategy import *
import matplotlib.pyplot as plt

# --- Configuration ---
config = {
    'entry_threshold': 3.4,
    'require_htf_score': True,
    'require_momentum': True,
    'commission': 0.0006,
    'cooldown_period_hours': 3,

    'htf_market_structure': {
        'lookback': 60,
        'atr_period': 60,
        'atr_mult': 1.2,
        'score_threshold': 0.45,
    },

    'order_blocks': {
        'atr_period': 20,
        'impulse_mult': 1.5,
        'max_width_mult': 3.0,
        'strength_threshold': 0.35,
    },

    'fair_value_gaps': {'strength_threshold': 0.35},

    'liquidity_sweep': {'lookback': 30, 'atr_period': 20, 'atr_mult': 0.7},

    'momentum_filter': {'consecutive_candles': 3, 'ratio_min': 0.7, 'ratio_max': 1.7, 'score_divisor': 1.0},

    'entry_planning': {
        'min_sl_dist_pct': 0.002,
        'tp1_rr': 0.9,
        'tp2_rr': 1.8,
        'buffer_atr_mult': 1.5,
        'buffer_price_pct': 0.0025,
    },

    'trailing_stop': {'breakeven_dist_mult': 1.05, 'trail_dist_atr_mult': 0.7},

    'weights': {
        'htf_market_bias': 1.8,
        'order_block': 2.0,
        'fvg': 1.6,
        'liquidity_sweep': 1.2,
        'ote': 0.6,
        'breaker': 0.6,
        'vwap_hvn': 0.6,
        'momentum': 1.4,
        'candle_body_confirm': 1.2,
        'session_spread': 0.4,
    }
}






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
    symbol_upper = symbol.upper()
    if symbol_upper in symbol_info_cache:
        return symbol_info_cache[symbol_upper]
    
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, tld='us')
    try:
        exchange_info = client.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol_upper:
                symbol_info_cache[symbol_upper] = s
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
    df.to_csv(filename, index_label='timestamp')
    print(f"Data for {symbol} saved to {filename}")
    return df

class Backtester:
    """
    A class to run a vectorized backtest for a given trading strategy.
    It handles data loading, trade execution simulation, risk management, and performance reporting.
    """
    def __init__(self, symbol, start_date, end_date, initial_capital, data_5m, data_15m, data_1h, config):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position = None
        self.trades = []
        self.config = config
        self.commission = config['commission']
        self.entry_threshold = config['entry_threshold']
        self.require_htf_score = config['require_htf_score']
        self.require_momentum = config['require_momentum']
        self.data_5m, self.data_15m, self.data_1h = data_5m, data_15m, data_1h
        if not self.data_15m.empty: self.data_15m['atr'] = atr(self.data_15m, self.config['order_blocks']['atr_period'])
        self.symbol_info = get_symbol_info_once(self.symbol)
        self.step_size = get_step_size(self.symbol, self.symbol_info)
        self.current_time = None
        self.ob_cooldown = {}
        self.cooldown_period = pd.Timedelta(hours=config['cooldown_period_hours'])

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
        # commission_cost = notional_value * self.commission
        # self.balance -= commission_cost
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
            htf_bias = detect_htf_market_structure(hist_1h, **self.config['htf_market_structure'])['side']
            if (self.position['side'] == 'buy' and htf_bias == 'sell') or (self.position['side'] == 'sell' and htf_bias == 'buy'):
                self.close_trade(candle['close'], 'HTF_FLIP'); return
        if not self.position['be_moved']:
            tp1_dist = abs(self.position['tp1'] - self.position['entry_price'])
            if tp1_dist > 0:
                if (self.position['side'] == 'buy' and (candle['high'] - self.position['entry_price']) >= self.config['trailing_stop']['breakeven_dist_mult'] * tp1_dist) or \
                   (self.position['side'] == 'sell' and (self.position['entry_price'] - candle['low']) >= self.config['trailing_stop']['breakeven_dist_mult'] * tp1_dist):
                    self.position['sl'] = self.position['entry_price']; self.position['be_moved'] = True
        if not self.position['trailing_active']:
            if (self.position['side'] == 'buy' and candle['high'] >= self.position['tp1']) or (self.position['side'] == 'sell' and candle['low'] <= self.position['tp1']):
                self.position['trailing_active'] = True
        if self.position['trailing_active']:
            atr_15m = self.data_15m.asof(self.current_time)['atr']
            if pd.notna(atr_15m):
                trail_dist = atr_15m * self.config['trailing_stop']['trail_dist_atr_mult']
                if self.position['side'] == 'buy':
                    new_sl = candle['high'] - trail_dist
                    if new_sl > self.position['sl']: self.position['sl'] = new_sl
                else:
                    new_sl = candle['low'] + trail_dist
                    if new_sl < self.position['sl']: self.position['sl'] = new_sl

    def close_trade(self, exit_price, reason):
        """Simulates closing the current trade and records the result."""
        if self.position:
            print(f"DEBUG: closing {self.position['side']} at {exit_price} reason={reason} stored_SL={self.position['sl']:.2f} tp1={self.position['tp1']:.2f} tp2={self.position['tp2']:.2f}")
        pnl = (exit_price - self.position['entry_price']) * self.position['size'] if self.position['side'] == 'buy' else (self.position['entry_price'] - exit_price) * self.position['size']
        commission_cost = self.position['notional'] * self.commission
        self.balance += pnl - commission_cost
        trade_log = {**self.position, 'symbol': self.symbol, 'exit_price': exit_price, 'pnl': pnl, 'exit_reason': reason, 'close_time': self.current_time}
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
        
        losing_sum = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        if losing_sum == 0:
            profit_factor_display = 'N/A'
        else:
            profit_factor_display = f"{winning_trades['pnl'].sum() / losing_sum:.2f}"

        cumulative_pnl = trades_df['pnl'].cumsum()
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
        
        # compute daily equity returns, then sharpe:
        equity = (self.initial_capital + cumulative_pnl).resample('D').last().ffill().pct_change().dropna()
        if len(equity) > 1:
            sharpe_ratio = equity.mean() / equity.std() * np.sqrt(252) if equity.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        print(f"Initial Capital: {self.initial_capital:.2f}\nFinal Balance: {self.balance:.2f}\nTotal PnL: {total_pnl:.2f}")
        print(f"Total Trades: {total_trades}\nWin Rate: {win_rate:.2f}%\nProfit Factor: {profit_factor_display}")
        print(f"Max Drawdown: {max_drawdown:.2f}\nSharpe Ratio: {sharpe_ratio:.2f}")
        
        plt.figure(figsize=(12, 6)); (self.initial_capital + cumulative_pnl).plot(); plt.title(f'Portfolio Equity Curve - {self.symbol}')
        plt.xlabel('Date'); plt.ylabel('Equity (USDT)'); plt.grid(True); plt.savefig(f'pnl_curve_{self.symbol}.png')
        print(f"\nChart saved to pnl_curve_{self.symbol}.png")

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
            htf_market_bias = detect_htf_market_structure(hist_1h, **self.config['htf_market_structure'])
            detection_results['htf_market_bias'] = htf_market_bias
            
            if htf_market_bias['score'] < self.config['htf_market_structure']['score_threshold']:
                continue
            
            side = htf_market_bias.get('side')
            
            order_blocks = find_order_blocks(hist_15m, '15m', **self.config['order_blocks']); last_ob = order_blocks[-1] if order_blocks else None
            ob_score = 1.0 if last_ob and last_ob.get('type') == side else 0.0
            detection_results['order_block'] = {'score': ob_score, 'side': side, 'meta': last_ob}
            
            fvgs = find_fair_value_gaps(hist_15m, '15m', **self.config['fair_value_gaps']); last_fvg = fvgs[-1] if fvgs else None
            fvg_score = 1.0 if last_fvg and last_fvg.get('side') == side else 0.0
            detection_results['fvg'] = {'score': fvg_score, 'side': side, 'meta': last_fvg}
            
            zone = last_ob.get('zone') if last_ob else None
            detection_results.update({
                'liquidity_sweep': detect_liquidity_sweep(hist_5m, **self.config['liquidity_sweep']),
                'ote': compute_ote(hist_15m),
                'breaker': detect_breaker_blocks(hist_15m, '15m'),
                'vwap_hvn': institutional_imbalance_vwap(hist_15m),
                'momentum': momentum_volume_filter(hist_5m, '5m', **self.config['momentum_filter']),
                'candle_body_confirm': candle_body_confirmation(hist_5m.iloc[-1], zone, side),
                'session_spread': session_and_spread_filter(self.current_time)
            })
            
            confluence_results = compute_confluence_score(detection_results, self.config['weights'])
            composite_score = confluence_results['composite_score']
            
            if composite_score >= self.entry_threshold:
                side = detection_results['htf_market_bias']['side']
                if side != 'neutral':
                    # --- Strengthened Entry Logic ---
                    if self.require_momentum and detection_results['momentum']['score'] == 0:
                        continue

                    last_ob = detection_results['order_block']['meta']
                    if last_ob and last_ob.get('strength', 0) < 0.2:
                        continue
                    
                    last_fvg = detection_results['fvg']['meta']
                    if last_fvg and last_fvg.get('strength', 0) < 0.2:
                        continue

                    if last_ob:
                        ob_key = (last_ob['origin_idx'], tuple(last_ob['zone']))
                        if ob_key in self.ob_cooldown and (self.current_time - self.ob_cooldown[ob_key] < self.cooldown_period):
                            continue

                    atr_15m = self.data_15m.asof(self.current_time)['atr']
                    if pd.isna(atr_15m) or atr_15m == 0:
                        price = hist_5m.iloc[-1]['close']
                        atr_15m = self.data_15m['atr'].dropna().iloc[-1] if self.data_15m['atr'].dropna().any() else max(1.0, price * 0.001)
                    plan = plan_entry_action(self.symbol, side, composite_score, detection_results, hist_5m.iloc[-1]['close'], atr_15m, self.config['entry_planning'])
                    self.execute_trade(side, plan['entry_price'], plan['stop_loss'], plan['targets'][0], plan['targets'][1], last_ob)
            else:
                # Optional: Log rejected candidates for tuning
                if composite_score > 1.5: # Log only potentially interesting ones
                    print(f"[{self.current_time}] REJECT | Score: {composite_score:.2f} | Votes: {confluence_results['votes']}")
        print("Backtest finished.")
        if self.position:
            # close at last available close price (end-of-data)
            last_close = self.data_5m.iloc[-1]['close']
            self.current_time = self.data_5m.index[-1]
            print(f"[{self.current_time}] EOD - forcing close of open position")
            self.close_trade(last_close, 'EOD')
        self.generate_report()
        return self.balance

def generate_consolidated_report(all_trades, initial_capital, final_balance):
    """
    Generates a consolidated report for all symbols.
    """
    if not all_trades:
        print("No trades were executed across all symbols.")
        return

    print("\n--- Consolidated Report ---")
    
    # Create a DataFrame from all trades
    trades_df = pd.DataFrame(all_trades)
    trades_df['close_time'] = pd.to_datetime(trades_df['close_time'])
    trades_df.sort_values(by='close_time', inplace=True)
    
    # Save all trades to CSV
    trades_df.to_csv('all_trade_details.csv', index=False)
    print("All trade details saved to all_trade_details.csv")

    # --- Account Summary ---
    summary_list = []
    for symbol, symbol_trades in trades_df.groupby('symbol'):
        total_trades = len(symbol_trades)
        if total_trades == 0: continue
        
        winning_trades = symbol_trades[symbol_trades['pnl'] > 0]
        losing_trades = symbol_trades[symbol_trades['pnl'] <= 0]
        win_rate = (len(winning_trades) / total_trades) * 100
        total_pnl = symbol_trades['pnl'].sum()
        
        summary_list.append({
            'Symbol': symbol,
            'Total Trades': total_trades,
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Win Rate (%)': f"{win_rate:.2f}",
            'Total PnL': f"{total_pnl:.2f}"
        })

    # Consolidated summary
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100
    total_pnl = trades_df['pnl'].sum()

    summary_list.append({
        'Symbol': 'Total',
        'Total Trades': total_trades,
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades),
        'Win Rate (%)': f"{win_rate:.2f}",
        'Total PnL': f"{total_pnl:.2f}"
    })
    
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv('account_summary.csv', index=False)
    print("Account summary saved to account_summary.csv")
    
    # --- Consolidated PnL Curve ---
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    plt.figure(figsize=(12, 6))
    (initial_capital + trades_df['cumulative_pnl']).plot()
    plt.title('Consolidated Portfolio Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity (USDT)')
    plt.grid(True)
    plt.savefig('final_pnl_curve.png')
    print("\nConsolidated PnL curve saved to final_pnl_curve.png")

    # Print final summary to console
    print(f"\nInitial Capital: {initial_capital:.2f}")
    print(f"Final Balance: {final_balance:.2f}")
    print(f"Total PnL across all symbols: {total_pnl:.2f}")
    print(f"Total Trades across all symbols: {total_trades}")

if __name__ == "__main__":
    """
    Main execution block to run the backtest.
    
    --- How to Use ---
    1.  Set the SYMBOLS environment variable to a comma-separated list of symbols to test (e.g., "BTCUSDT,ETHUSDT").
    2.  Configure the parameters below (dates, capital).
    3.  Ensure you have your Binance API keys set as environment variables if you need to download new data.
    4.  Run the script: `python back.py`
    5.  The backtest report and PnL curve chart will be generated for each symbol.
    """
    # --- Configuration ---
    symbols_to_test = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,TRXUSDT,TONUSDT,LTCUSDT,AAVEUSDT").split(',')
    start_date_str = "2025-08-30"
    end_date_str = "2025-09-06"
    initial_capital = 100.0
    
    print("Backtesting framework starting...")
    
    final_balance = initial_capital
    all_trades = []

    for symbol_to_test in symbols_to_test:
        print(f"--- Running backtest for {symbol_to_test} ---")
        print("Ensuring data is available...")
        data_to_load = {f'data_{tf}': fetch_and_cache_data(symbol_to_test, tf, start_date_str, end_date_str) for tf in ['5m', '15m', '1h']}
        if not any(df.empty for df in data_to_load.values()):
            backtester = Backtester(symbol_to_test, start_date_str, end_date_str, final_balance, **data_to_load, config=config)
            final_balance = backtester.run()
            all_trades.extend(backtester.trades)
        else:
            print(f"Could not load data for {symbol_to_test}, aborting backtest for this symbol.")
    
    generate_consolidated_report(all_trades, initial_capital, final_balance)
