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
    'tld': 'us',
    'min_trade_value_override': 1.0,
    'enforce_max_position_value': False,
    'max_position_value_pct': 0.40,   # never more than 40% of account in any single position
    'risk_pct': 0.02,                # 1.5% of balance (small)
    'min_risk_amount': 0.02,          # absolute floor USD (tiny)
    'max_risk_per_trade': 0.5,        # absolute USD cap
    'entry_threshold': 3.0,
    'sniper_threshold_delta': 2.0,
    'soft_penalty': 0.3,
    'require_htf_score': True,
    'require_momentum': True,
    'commission': 0.0006,
    'slippage': 0.0010,
    'spread_pct_random_range': (0.0002, 0.0008), # Simulate variable spread between 0.02% and 0.08%
    'cooldown_period_hours': 3,
    'htf_market_structure': {'lookback': 100, 'atr_period': 60, 'atr_mult':1.2, 'score_threshold':0.45},
    'order_blocks': {'atr_period':20, 'impulse_mult':1.5, 'max_width_mult':3.0, 'strength_threshold':0.4},
    'fair_value_gaps': {'strength_threshold':0.4},
    'liquidity_sweep': {'lookback':20, 'atr_period':14, 'atr_mult':0.6},
    'momentum_filter': {'consecutive_candles':4, 'ratio_min':0.8, 'ratio_max':1.7, 'score_divisor':0.9},
    'entry_planning': {'min_sl_dist_pct':0.003, 'max_sl_pct':0.02, 'tp1_rr':0.7, 'tp2_rr':1.8, 'buffer_atr_mult':2.0, 'buffer_price_pct':0.003},
    'trailing_stop': {'breakeven_dist_mult':1.05, 'trail_dist_atr_mult':0.5},
    'micro_sniper_trigger': {'lookback':24, 'wick_atr_mult':0.25, 'volume_z_k':1.5, 'atr_period':14},
    'correlation_check': {'enabled': True, 'threshold': 0.85, 'lookback_candles': 100},
    'weights': {'htf_market_bias':2.5,'order_block':2.2,'fvg':1.6,'liquidity_sweep':1.2,'ote':0.4,'breaker':1.2,'vwap_hvn':0.4,'momentum':2.2,'candle_body_confirm':1.6,'session_spread':0.4,'micro_sniper_trigger':2.0}
}





DATA_DIR = "data"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- Exchange Info Helpers ---
symbol_info_cache = {}

def get_symbol_info_once(symbol, tld='com'):
    """
    Fetches and caches symbol information from Binance.
    This is to avoid repeated API calls for the same symbol info.
    """
    symbol_upper = symbol.upper()
    if symbol_upper in symbol_info_cache:
        return symbol_info_cache[symbol_upper]
    
    try:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, tld=tld)
        exchange_info = client.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol_upper:
                symbol_info_cache[symbol_upper] = s
                return s
    except Exception as e:
        print(f"Could not fetch exchange info for {symbol} using tld={tld}: {e}")
    
    print(f"WARNING: Could not find symbol info for {symbol}. Using default values.")
    if symbol_upper == 'BTCUSDT':
        return {
            'symbol': 'BTCUSDT',
            'filters': [
                {'filterType': 'LOT_SIZE', 'stepSize': '0.00001'},
                {'filterType': 'MIN_NOTIONAL', 'minNotional': '5.0'}
            ]
        }
    return None

def get_step_size(symbol: str, symbol_info: dict) -> Decimal:
    """Extracts the lot step size from the symbol information."""
    if not symbol_info:
        print(f"WARNING: symbol_info missing for {symbol}. Using default step_size=1.")
        return Decimal('1')
    for f in symbol_info.get('filters', []):
        if f.get('filterType') == 'LOT_SIZE':
            return Decimal(str(f.get('stepSize', '1')))
    return Decimal('1')

def get_min_notional(symbol: str, symbol_info: dict) -> Decimal:
    """Extracts the min notional value from the symbol information."""
    if not symbol_info:
        print(f"WARNING: symbol_info missing for {symbol}. Using default min_notional=0.")
        return Decimal('0')
    for f in symbol_info.get('filters', []):
        if f.get('filterType') == 'MIN_NOTIONAL':
            return Decimal(str(f.get('minNotional', '0')))
    return Decimal('0')

def round_qty(qty: float, step_size: Decimal) -> float:
    """Rounds a quantity down to the nearest valid step size."""
    if step_size == 0: return float(qty)
    qty_decimal = Decimal(str(qty))
    rounded_qty = (qty_decimal / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
    return float(rounded_qty)

def fetch_and_cache_data(symbol: str, timeframe: str, start_date: str, end_date: str, tld: str = 'com') -> pd.DataFrame:
    """
    Fetches historical k-line data from Binance and caches it locally.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{DATA_DIR}/{symbol.upper()}-{timeframe}-{start_date}-{end_date}.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    print(f"Fetching new data for {symbol} ({timeframe}) from {start_date} to {end_date}")
    try:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, tld=tld)
        klines = client.get_historical_klines(symbol, timeframe, start_date, end_date)
    except Exception as e:
        print(f"An error occurred while fetching data for {symbol}: {e}")
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
    def __init__(self, symbol, start_date, end_date, initial_capital, data_1m, data_5m, data_15m, data_1h, config, all_trades=None, all_data=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position = None
        self.trades = []
        self.all_trades = all_trades if all_trades is not None else []
        self.all_data = all_data if all_data is not None else {}
        self.ledger = []
        self.candidates = []
        self.config = config
        self.commission = config['commission']
        self.entry_threshold = config['entry_threshold']
        self.require_htf_score = config['require_htf_score']
        self.require_momentum = config['require_momentum']
        self.consecutive_losses = 0
        self.data_1m, self.data_5m, self.data_15m, self.data_1h = data_1m, data_5m, data_15m, data_1h
        if not self.data_15m.empty: self.data_15m['atr'] = atr(self.data_15m, self.config['order_blocks']['atr_period'])
        self.symbol_info = get_symbol_info_once(self.symbol, self.config.get('tld', 'com'))
        self.step_size = get_step_size(self.symbol, self.symbol_info)
        self.min_notional = get_min_notional(self.symbol, self.symbol_info)

        # make a float copy for numeric comparisons and log defaults
        try:
            self.min_notional_float = float(self.min_notional)
        except Exception:
            self.min_notional_float = 0.0

        if self.min_notional_float == 0.0:
            print(f"[INIT] Note: min_notional for {self.symbol} = 0.0 (default). Step size: {self.step_size}")
        else:
            print(f"[INIT] {self.symbol} min_notional = {self.min_notional_float}, step_size = {self.step_size}")
        self.current_time = None
        self.ob_cooldown = {}
        self.cooldown_period = pd.Timedelta(hours=config['cooldown_period_hours'])

    def update_balance(self, net_pnl, timestamp, description):
        """Updates balance and records the transaction in the ledger."""
        balance_before = self.balance
        self.balance += net_pnl
        balance_after = self.balance
        self.ledger.append({
            'timestamp': timestamp,
            'trade_id': len(self.trades) + 1,
            'account': 'cash',
            'debit': net_pnl if net_pnl < 0 else 0,
            'credit': net_pnl if net_pnl > 0 else 0,
            'balance_before': balance_before,
            'balance_after': balance_after,
            'description': description,
        })
        print(f"[{timestamp}] BALANCE UPDATE | Amount: {net_pnl:.2f} | New Balance: {self.balance:.2f} | Desc: {description}")
        return balance_before, balance_after

    def get_random_spread_pct(self):
        if 'spread_pct_random_range' in self.config:
            min_spread, max_spread = self.config['spread_pct_random_range']
            return np.random.uniform(min_spread, max_spread)
        return self.config.get('spread_pct', 0.0)

    def calculate_risk_amount(self):
        """
        Calculates the amount to risk on a trade based on a hybrid rule:
        - Use a percentage of the balance.
        - Enforce an absolute minimum risk amount (floor).
        - Enforce an absolute maximum risk amount (cap).
        This makes the strategy viable for very small accounts.
        """
        base_risk_pct = self.config.get('risk_pct', 0.01)
        absolute_floor = self.config.get('min_risk_amount', 0.05)
        max_risk = self.config.get('max_risk_per_trade', 0.25)
        
        # Calculate risk based on percentage, but not less than the floor
        dynamic_risk = max(absolute_floor, self.balance * base_risk_pct)
        
        # The final risk amount is capped by the maximum allowed risk per trade
        risk_amount = min(dynamic_risk, max_risk)

        if self.consecutive_losses >= 2:
            risk_amount /= 2
        
        return risk_amount

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

        spread_pct = self.get_random_spread_pct()
        spread = entry_price * spread_pct
        entry_price_after_slippage = entry_price + (entry_price * self.config.get('slippage', 0.0)) if side == 'buy' else entry_price - (entry_price * self.config.get('slippage', 0.0))
        entry_price_after_slippage += spread / 2
        
        notional_value = size * entry_price_after_slippage
        commission_open = notional_value * self.commission
        
        self.position = {
            'side': side, 
            'entry_price': entry_price_after_slippage, 
            'size': size, 
            'sl': sl_price, 
            'tp1': tp1, 
            'tp2': tp2, 
            'notional': notional_value, 
            'open_time': self.current_time, 
            'be_moved': False, 
            'trailing_active': False,
            'commission_open': commission_open,
            'slippage_open': abs(entry_price - entry_price_after_slippage) * size,
        }
        print(f"[{self.current_time}] OPEN {side} | Size: {size:.4f} | Entry: {entry_price_after_slippage:.2f} | SL: {sl_price:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f}")

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
            if (self.position['side'] == 'buy' and (candle['high'] - self.position['entry_price']) >= (abs(self.position['tp1'] - self.position['entry_price']) * 0.5)) or \
               (self.position['side'] == 'sell' and (self.position['entry_price'] - candle['low']) >= (abs(self.position['tp1'] - self.position['entry_price']) * 0.5)):
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
        if not self.position:
            return

        spread_pct = self.get_random_spread_pct()
        spread = exit_price * spread_pct
        exit_price_after_slippage = exit_price - (exit_price * self.config.get('slippage', 0.0)) if self.position['side'] == 'buy' else exit_price + (exit_price * self.config.get('slippage', 0.0))
        exit_price_after_slippage -= spread / 2

        gross_pnl = (exit_price_after_slippage - self.position['entry_price']) * self.position['size'] if self.position['side'] == 'buy' else (self.position['entry_price'] - exit_price_after_slippage) * self.position['size']
        
        commission_open = self.position['commission_open']
        commission_close = (self.position['size'] * exit_price_after_slippage) * self.commission
        total_commission = commission_open + commission_close

        slippage_open = self.position['slippage_open']
        slippage_close = abs(exit_price - exit_price_after_slippage) * self.position['size']
        total_slippage = slippage_open + slippage_close

        net_pnl = gross_pnl - total_commission - total_slippage
        
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        balance_before, balance_after = self.update_balance(net_pnl, self.current_time, f"CLOSE_TRADE_{self.symbol}")

        trade_log = {
            'trade_id': len(self.trades) + 1,
            'symbol': self.symbol,
            'entry_time': self.position['open_time'],
            'exit_time': self.current_time,
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price_after_slippage,
            'qty': self.position['size'],
            'gross_pnl': gross_pnl,
            'commission_open': commission_open,
            'commission_close': commission_close,
            'total_commission': total_commission,
            'slippage_open': slippage_open,
            'slippage_close': slippage_close,
            'total_slippage': total_slippage,
            'net_pnl': net_pnl,
            'balance_before': balance_before,
            'balance_after': balance_after,
            'exit_reason': reason,
        }
        self.trades.append(trade_log)
        print(f"[{self.current_time}] CLOSE {self.position['side']} | Exit: {exit_price_after_slippage:.2f} | Net PnL: {net_pnl:.2f} | Balance: {self.balance:.2f} | Reason: {reason}")
        self.position = None

    def generate_report(self):
        """Generates and prints a performance report and saves a PnL chart."""
        print("\n--- Backtest Report ---")
        if not self.trades:
            print("No trades were executed."); return

        trades_df = pd.DataFrame(self.trades)
        # prefer exit_time if present
        if 'exit_time' in trades_df.columns:
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df.set_index('exit_time', inplace=True)
        elif 'close_time' in trades_df.columns:
            trades_df['close_time'] = pd.to_datetime(trades_df['close_time'])
            trades_df.set_index('close_time', inplace=True)

        total_gross_pnl = trades_df['gross_pnl'].sum()
        total_commission = trades_df['total_commission'].sum()
        total_slippage = trades_df['total_slippage'].sum()
        total_net_pnl = trades_df['net_pnl'].sum()
        
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        losing_sum = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum())
        if losing_sum == 0:
            profit_factor_display = 'inf' if winning_trades['net_pnl'].sum() > 0 else 'N/A'
        else:
            profit_factor_display = f"{winning_trades['net_pnl'].sum() / losing_sum:.2f}"

        cumulative_pnl = trades_df['net_pnl'].cumsum()
        max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max()
        
        equity = (self.initial_capital + cumulative_pnl)
        if len(equity.pct_change().dropna()) > 1:
            sharpe_ratio = equity.pct_change().mean() / equity.pct_change().std() * np.sqrt(252) if equity.pct_change().std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        print(f"Initial Capital: {self.initial_capital:.2f}\nFinal Balance: {self.balance:.2f}")
        print(f"Total Gross PnL: {total_gross_pnl:.2f}")
        print(f"Total Commissions: {total_commission:.2f}")
        print(f"Total Slippage: {total_slippage:.2f}")
        print(f"Total Net PnL: {total_net_pnl:.2f}")

        if not np.isclose(self.initial_capital + total_net_pnl, self.balance):
            print("\n---!!! BALANCE MISMATCH ERROR !!!---")
            print(f"Initial: {self.initial_capital:.2f} + Net PnL: {total_net_pnl:.2f} = {self.initial_capital + total_net_pnl:.2f}")
            print(f"Final Balance: {final_balance:.2f}")
            print("------------------------------------")

        print(f"\nTotal Trades: {total_trades}\nWin Rate: {win_rate:.2f}%\nProfit Factor: {profit_factor_display}")
        print(f"Max Drawdown: {max_drawdown:.2f}\nSharpe Ratio: {sharpe_ratio:.2f}")
        
        plt.figure(figsize=(12, 6)); equity.plot(); plt.title(f'Portfolio Equity Curve - {self.symbol}')
        plt.xlabel('Date'); plt.ylabel('Equity (USDT)'); plt.grid(True); plt.savefig(f'pnl_curve_{self.symbol}.png')
        print(f"\nChart saved to pnl_curve_{self.symbol}.png")
        
        # Save candidates to CSV
        if self.candidates:
            candidates_df = pd.DataFrame(self.candidates)
            candidates_df.to_csv('candidates.csv', index=False)
            print("Candidates saved to candidates.csv")

            # --- Candidate Summary ---
            print("\n--- Candidate Summary ---")
            print(f"Total candidates considered: {len(self.candidates)}")

            # Filter for candidates that were not rejected
            accepted_candidates = candidates_df[candidates_df['reject_reasons'].apply(lambda x: isinstance(x, list) and len(x) == 0)]
            print(f"Total candidates passed all filters: {len(accepted_candidates)}")

            if not accepted_candidates.empty:
                print("\nScore distribution for accepted candidates:")
                print(accepted_candidates['confluence_score'].describe())

                print("\nTop 10 accepted candidates by score:")
                print(accepted_candidates.sort_values(by='confluence_score', ascending=False).head(10)[['timestamp', 'confluence_score', 'planned_position_value']])

            # Show rejection reasons breakdown
            print("\nRejection Reasons Breakdown:")
            rejected_candidates = candidates_df[candidates_df['reject_reasons'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
            if not rejected_candidates.empty:
                rejection_counts = rejected_candidates['reject_reasons'].explode().value_counts()
                print(rejection_counts)
            else:
                print("No candidates were rejected based on the logged reasons.")

    def run(self):
        """Runs the main backtesting loop."""
        print("Running backtest...");
        if self.data_1m.empty or self.data_5m.empty: return
        for i in range(1, len(self.data_5m)):
            self.current_time = self.data_5m.index[i]
            if self.position: self.manage_position(); continue
            
            # Prepare historical data for all timeframes
            hist_1m = self.data_1m[self.data_1m.index <= self.current_time]
            hist_5m = self.data_5m.iloc[:i+1]
            hist_15m = self.data_15m[self.data_15m.index <= self.current_time]
            hist_1h = self.data_1h[self.data_1h.index <= self.current_time]

            # Ensure we have enough data to run indicators
            if len(hist_1h) < 60 or len(hist_15m) < 110 or len(hist_5m) < 70 or len(hist_1m) < 70: continue

            # --- Volatility and Trend Filters ---
            current_atr = self.data_15m.asof(self.current_time)['atr']
            avg_atr = self.data_15m['atr'].rolling(window=50).mean().asof(self.current_time)
            if pd.notna(current_atr) and pd.notna(avg_atr) and current_atr < (avg_atr * 0.5):
                continue # Skip if volatility is too low

            current_adx = adx(hist_15m, period=14).iloc[-1]
            if pd.notna(current_adx) and current_adx < 25:
                continue # Skip if market is not trending
            
            detection_results = {}
            htf_market_bias = detect_htf_market_structure(hist_1h, **self.config['htf_market_structure'])
            detection_results['htf_market_bias'] = htf_market_bias
            
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
                'breaker': detect_breaker_blocks(hist_15m, '15m', **self.config),
                'vwap_hvn': institutional_imbalance_vwap(hist_15m, **self.config),
                'momentum': momentum_volume_filter(hist_5m, '5m', **self.config['momentum_filter']),
                'candle_body_confirm': candle_body_confirmation(hist_5m.iloc[-1], zone, side),
                'session_spread': session_and_spread_filter(),
                'micro_sniper_trigger': micro_sniper_trigger(hist_1m, htf_market_bias.get('side'), **self.config['micro_sniper_trigger'])
            })
            
            confluence_results = compute_confluence_score(detection_results, self.config['weights'])
            composite_score = confluence_results['composite_score']

            # debug: show full confluence details so we understand the score scale
            if self.config.get('debug'):
                print(f"[{self.current_time}] CONFLUENCE RAW -> composite: {composite_score:.4f} votes: {confluence_results.get('votes')} full: {confluence_results}")
            
            # Soft gating (gentle penalties and debug logging)
            penalty_amount = self.config.get('soft_penalty', 0.3)

            if self.config.get('require_htf_score') and htf_market_bias.get('score', 0) < self.config['htf_market_structure']['score_threshold']:
                composite_score -= penalty_amount

            if self.config.get('require_momentum') and detection_results['momentum'].get('score', 0) == 0:
                composite_score -= penalty_amount

            if last_ob and last_ob.get('strength', 0) < self.config['order_blocks']['strength_threshold']:
                composite_score -= penalty_amount

            if last_fvg and last_fvg.get('strength', 0) < self.config['fair_value_gaps']['strength_threshold']:
                composite_score -= penalty_amount

            # debug print for every candidate
            if self.config.get('debug'):
                print(f"[{self.current_time}] DEBUG composite_score: {composite_score:.3f} | htf_score: {htf_market_bias.get('score', None)} | momentum: {detection_results['momentum'].get('score', None)} | ob_strength: {last_ob.get('strength') if last_ob else None}")

            # Candidate logging
            reject_reasons = []

            # Dynamic entry threshold for sniper trades
            current_entry_threshold = self.entry_threshold
            sniper_result = detection_results.get('micro_sniper_trigger', {})
            htf_side = detection_results.get('htf_market_bias', {}).get('side')

            is_strong_sniper = sniper_result.get('score', 0) > 0.6
            is_aligned_with_htf = htf_side == sniper_result.get('side') and htf_side != 'neutral'

            if is_strong_sniper and is_aligned_with_htf:
                delta = self.config.get('sniper_threshold_delta', 1.0)
                current_entry_threshold -= delta
                print(f"[{self.current_time}] INFO: Sniper override active. Threshold lowered to {current_entry_threshold:.2f}")

            if composite_score < current_entry_threshold:
                reject_reasons.append(f'score_too_low (score: {composite_score:.2f}, threshold: {current_entry_threshold:.2f})')

            # Calculate planned position size and value
            atr_row = self.data_15m.asof(self.current_time)
            atr_15m = atr_row['atr'] if isinstance(atr_row, pd.Series) and pd.notna(atr_row.get('atr', np.nan)) else None
            if atr_15m is None:
                # If ATR is not available, we cannot plan a trade that depends on it.
                print(f"[{self.current_time}] WARN: Could not get 15m ATR for {self.symbol}. Skipping candidate.")
                continue
            
            plan = plan_entry_action(self.symbol, side, composite_score, detection_results, hist_5m.iloc[-1]['close'], atr_15m, self.config['entry_planning'])
            
            if not plan or 'entry_price' not in plan or 'stop_loss' not in plan:
                print(f"[{self.current_time}] ERROR: plan_entry_action returned invalid plan: {plan}")
                continue

            calculated_qty = self.calculate_position_size(plan['entry_price'], plan['stop_loss'])
            planned_position_value = calculated_qty * plan['entry_price']

            min_notional_pass = planned_position_value >= self.min_notional_float
            min_qty_pass = calculated_qty > 0
            
            min_trade_override = self.config.get('min_trade_value_override')
            if not min_notional_pass and min_trade_override and planned_position_value >= min_trade_override:
                min_notional_pass = True
                print(f"[{self.current_time}] INFO: Overriding min_notional check for {self.symbol}. Planned: {planned_position_value:.2f}, Min: {self.min_notional}")


            if not min_notional_pass:
                reject_reasons.append(f'min_notional_fail (req: {self.min_notional}, got: {planned_position_value:.2f})')
            if not min_qty_pass:
                reject_reasons.append('min_qty_fail')
            
            candidate_log = {
                'symbol': self.symbol,
                'timestamp': self.current_time,
                'confluence_score': composite_score,
                'score_breakdown': confluence_results['votes'],
                'reject_reasons': reject_reasons,
                'planned_position_value': planned_position_value,
                'calculated_qty': calculated_qty,
                'min_notional_pass': min_notional_pass,
                'min_qty_pass': min_qty_pass,
            }
            self.candidates.append(candidate_log)

            if not reject_reasons and side != 'neutral':
                # --- Correlation Check ---
                is_correlated = False
                corr_config = self.config.get('correlation_check', {})
                if corr_config.get('enabled', False) and self.all_trades:
                    open_positions_other_symbols = []
                    for trade in self.all_trades:
                        if trade['entry_time'] <= self.current_time and self.current_time < trade['exit_time']:
                            open_positions_other_symbols.append(trade)
                    
                    if open_positions_other_symbols:
                        lookback = corr_config.get('lookback_candles', 100)
                        threshold = corr_config.get('threshold', 0.85)
                        hist_self = self.data_5m[self.data_5m.index <= self.current_time].tail(lookback)['close']

                        for open_trade in open_positions_other_symbols:
                            other_symbol = open_trade['symbol']
                            if other_symbol == self.symbol or other_symbol not in self.all_data: continue

                            hist_other = self.all_data[other_symbol]['data_5m']
                            hist_other = hist_other[hist_other.index <= self.current_time].tail(lookback)['close']
                            
                            aligned_self, aligned_other = hist_self.align(hist_other, join='inner')

                            if len(aligned_self) > lookback * 0.8: # Ensure sufficient overlapping data
                                correlation = aligned_self.pct_change().corr(aligned_other.pct_change())
                                if pd.notna(correlation) and abs(correlation) > threshold:
                                    is_correlated = True
                                    reason = f'high_correlation_with_{other_symbol} ({correlation:.2f})'
                                    reject_reasons.append(reason)
                                    self.candidates[-1]['reject_reasons'] = reject_reasons # Update last candidate
                                    print(f"[{self.current_time}] CANCELED {self.symbol} trade due to: {reason}")
                                    break 
                
                if not is_correlated:
                    if last_ob:
                        ob_key = (last_ob['origin_idx'], tuple(last_ob['zone']))
                        if ob_key in self.ob_cooldown and (self.current_time - self.ob_cooldown[ob_key] < self.cooldown_period):
                            continue
                    self.execute_trade(side, plan['entry_price'], plan['stop_loss'], plan['targets'][0], plan['targets'][1], last_ob)

        print("Backtest finished.")
        if self.position:
            # Close at the worst price of the last candle to be conservative
            last_candle = self.data_5m.iloc[-1]
            exit_price = last_candle['low'] if self.position['side'] == 'buy' else last_candle['high']
            self.current_time = self.data_5m.index[-1]
            print(f"[{self.current_time}] EOD - forcing close of open position at worst price: {exit_price}")
            self.close_trade(exit_price, 'EOD_WORST_PRICE')
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
    if 'exit_time' in trades_df.columns:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df.sort_values(by='exit_time', inplace=True)
    
    # Save all trades to CSV
    trades_df.to_csv('all_trade_details.csv', index=False)
    print("All trade details saved to all_trade_details.csv")

    # --- Account Summary ---
    summary_list = []
    for symbol, symbol_trades in trades_df.groupby('symbol'):
        total_trades = len(symbol_trades)
        if total_trades == 0: continue
        
        winning_trades = symbol_trades[symbol_trades['net_pnl'] > 0]
        losing_trades = symbol_trades[symbol_trades['net_pnl'] <= 0]
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = symbol_trades['net_pnl'].sum()
        
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        sum_wins = winning_trades['net_pnl'].sum()
        sum_losses = abs(losing_trades['net_pnl'].sum())
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else 'inf'

        summary_list.append({
            'Symbol': symbol,
            'Total Trades': total_trades,
            'Win Rate (%)': f"{win_rate:.2f}",
            'Total PnL': f"{total_pnl:.2f}",
            'Avg Win': f"{avg_win:.2f}",
            'Avg Loss': f"{avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.2f}" if isinstance(profit_factor, (int, float)) else profit_factor
        })

    # Consolidated summary
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['net_pnl'] > 0]
    losing_trades = trades_df[trades_df['net_pnl'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_net_pnl = trades_df['net_pnl'].sum()

    avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
    
    sum_wins = winning_trades['net_pnl'].sum()
    sum_losses = abs(losing_trades['net_pnl'].sum())
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 'inf'

    summary_list.append({
        'Symbol': 'Total',
        'Total Trades': total_trades,
        'Win Rate (%)': f"{win_rate:.2f}",
        'Total PnL': f"{total_net_pnl:.2f}",
        'Avg Win': f"{avg_win:.2f}",
        'Avg Loss': f"{avg_loss:.2f}",
        'Profit Factor': f"{profit_factor:.2f}" if isinstance(profit_factor, (int, float)) else profit_factor
    })
    
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv('account_summary.csv', index=False)
    print("Account summary saved to account_summary.csv")
    
    # --- Consolidated PnL Curve ---
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
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
    print(f"Total PnL across all symbols: {total_net_pnl:.2f}")
    print(f"Total Trades across all symbols: {total_trades}")

    if not np.isclose(initial_capital + total_net_pnl, final_balance):
            print("\n---!!! CONSOLIDATED BALANCE MISMATCH ERROR !!!---")
            print(f"Initial: {initial_capital:.2f} + Net PnL: {total_net_pnl:.2f} = {initial_capital + total_net_pnl:.2f}")
            print(f"Final Balance: {final_balance:.2f}")
            print("------------------------------------")


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
    symbols_to_test = ["BTCUSDT", "ETHUSDT"]
    start_date_str = "2025-08-30"
    end_date_str = "2025-09-06"
    initial_capital = 10.0
    
    print("Backtesting framework starting...")

    # Pre-load all data for all symbols
    all_data = {}
    for symbol in symbols_to_test:
        print(f"Loading data for {symbol}...")
        data_for_symbol = {f'data_{tf}': fetch_and_cache_data(symbol, tf, start_date_str, end_date_str, config.get('tld', 'com')) for tf in ['1m', '5m', '15m', '1h']}
        if any(df.empty for df in data_for_symbol.values()):
            print(f"Could not load full data for {symbol}, it will be skipped.")
        all_data[symbol] = data_for_symbol
    
    final_balance = initial_capital
    all_trades = []

    for symbol_to_test in symbols_to_test:
        if symbol_to_test not in all_data or any(df.empty for df in all_data[symbol_to_test].values()):
            continue

        print(f"\n--- Running backtest for {symbol_to_test} ---")
        backtester = Backtester(
            symbol=symbol_to_test, 
            start_date=start_date_str, 
            end_date=end_date_str, 
            initial_capital=final_balance, 
            **all_data[symbol_to_test], 
            config=config,
            all_trades=all_trades,
            all_data=all_data
        )
        final_balance = backtester.run()
        all_trades.extend(backtester.trades)
    
    generate_consolidated_report(all_trades, initial_capital, final_balance)
