import os
import json
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np
import plotly.graph_objects as go
from binance.client import Client
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("backtester")

CONFIG = {
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "BNBUSDT"], "TIMEFRAME": "15m",
    "INITIAL_CAPITAL": 10000.0, "BINANCE_FEE": 0.0004, "RISK_PCT_LARGE": 0.02,
    "RISK_SMALL_BALANCE_THRESHOLD": 50.0, "RISK_SMALL_FIXED_USDT": 0.5,
    "MIN_NOTIONAL_USDT": 5.0, "EMA_LEN": 200, "EMASIGNAL_BACKCANDLES": 6,
    "BB_LENGTH_CUSTOM": 20, "BB_STD_CUSTOM": 2.5, "RSI_LEN": 2,
    "ORDER_ENTRY_TIMEOUT": 5, "MAX_TRADE_AGE_BARS": 10, "ORDER_LIMIT_OFFSET_PCT": 0.0,
    "USE_LIMIT_ENTRY": True, "TP1_RR": 1.0, "TP2_RR": 2.0, "TP1_CLOSE_PCT": 0.5,
    "TP2_CLOSE_PCT": 0.5, "SL_BUFFER_PCT": 0.02, "SL_TP_ATR_MULT": 2.5, "ATR_LENGTH": 14,
}

def setup_config(config_file="config.json"):
    global CONFIG
    if os.path.exists(config_file):
        with open(config_file, 'r') as f: CONFIG.update(json.load(f))
    else:
        with open(config_file, 'w') as f: json.dump(CONFIG, f, indent=4)
    return CONFIG

def fetch_historical_data(config, data_file="historical_data.parquet"):
    if os.path.exists(data_file):
        log.info(f"Loading historical data from {data_file}...")
        return pd.read_parquet(data_file)
    
    log.info("Historical data file not found. Starting download...")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        log.error("Binance API credentials not found in environment variables.")
        return None
    
    client = Client(api_key, api_secret)
    
    try:
        days_input = input("How many days of historical data to download? ")
        days = int(days_input)
    except (ValueError, EOFError):
        log.error("Invalid input. Please enter a number.")
        return None

    all_klines_df = []
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%d %b, %Y")

    log.info(f"Downloading {days} days of data from {start_str} for symbols: {config['SYMBOLS']}")
    for symbol in config['SYMBOLS']:
        log.info(f"Fetching data for {symbol}...")
        klines = client.get_historical_klines(symbol, config['TIMEFRAME'], start_str)
        df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore'])
        df['symbol'] = symbol
        all_klines_df.append(df)
    
    final_df = pd.concat(all_klines_df)
    for col in ['open', 'high', 'low', 'close', 'volume']: final_df[col] = pd.to_numeric(final_df[col])
    final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms', utc=True)
    final_df.to_parquet(data_file)
    log.info(f"Data saved to {data_file}")
    return final_df

def ema(values, n): return pd.Series(values).ewm(span=n, adjust=False).mean()
def rsi(series, n):
    delta = pd.Series(series).diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=n - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=n - 1, adjust=False).mean()
    rs = gain / loss; rs.replace([np.inf, -np.inf], np.nan, inplace=True); rs.fillna(method='ffill', inplace=True)
    rsi_val = 100 - (100 / (1 + rs)); rsi_val.fillna(50, inplace=True)
    return rsi_val
def bollinger_bands(series, n, std_dev):
    ma = pd.Series(series).rolling(n).mean(); std = pd.Series(series).rolling(n).std()
    return (ma + (std * std_dev)), (ma - (std * std_dev))
def addemasignal(close, ema, backcandles):
    close_s = pd.Series(close); ema_s = pd.Series(ema)
    above_ema = (close_s > ema_s).astype(int); below_ema = (close_s < ema_s).astype(int)
    all_above = above_ema.rolling(window=backcandles).sum() == backcandles
    all_below = below_ema.rolling(window=backcandles).sum() == backcandles
    signal = pd.Series(0, index=close_s.index); signal[all_above] = 2; signal[all_below] = 1
    return signal.values
def addorderslimit(close, emasignal, bbl, bbu):
    ordersignal = np.zeros_like(close)
    long_cond = (emasignal == 2) & (close <= bbl); short_cond = (emasignal == 1) & (close >= bbu)
    ordersignal[long_cond] = bbl[long_cond]; ordersignal[short_cond] = bbu[short_cond]
    return ordersignal

class EmaBBStrategy(Strategy):
    ema_len = 200
    emasignal_backcandles = 6
    bb_length = 20
    bb_std = 2.5
    rsi_len = 2
    max_trade_age_bars = 10
    sl_buffer_pct = 0.02
    tp1_rr = 1.0
    
    def init(self):
        self.ema200 = self.I(ema, self.data.Close, self.ema_len)
        self.rsi2 = self.I(rsi, self.data.Close, self.rsi_len)
        bbu, bbl = self.I(bollinger_bands, self.data.Close, self.bb_length, self.bb_std, plot=False)
        self.bbu = self.I(lambda: bbu, name="BBU"); self.bbl = self.I(lambda: bbl, name="BBL")
        self.emasignal = self.I(addemasignal, self.data.Close, self.ema200, self.emasignal_backcandles, plot=False)
        self.ordersignal = self.I(addorderslimit, self.data.Close, self.emasignal, self.bbl, self.bbu, plot=False)

    def next(self):
        if self.position:
            if (len(self.data) - self.trades[-1].entry_bar) >= self.max_trade_age_bars or \
               (self.position.is_long and self.rsi2[-1] >= 50) or \
               (self.position.is_short and self.rsi2[-1] <= 50):
                self.position.close()
        if self.ordersignal[-1] > 0 and not self.position:
            sl_basis = self.data.Low[-self.emasignal_backcandles:].min() if self.emasignal[-1] == 2 else self.data.High[-self.emasignal_backcandles:].max()
            stop_price = sl_basis * (1 - self.sl_buffer_pct) if self.emasignal[-1] == 2 else sl_basis * (1 + self.sl_buffer_pct)
            tp_price = self.data.Close[-1] + (self.data.Close[-1] - stop_price) * self.tp1_rr if self.emasignal[-1] == 2 else self.data.Close[-1] - (stop_price - self.data.Close[-1]) * self.tp1_rr
            if self.emasignal[-1] == 2: self.buy(sl=stop_price, tp=tp_price)
            else: self.sell(sl=stop_price, tp=tp_price)

def generate_consolidated_report(all_stats, all_trades, portfolio_equity, config, filename="consolidated_report.html"):
    log.info(f"Generating consolidated HTML report to {filename}...")
    if all_trades.empty:
        log.warning("No trades were made. Cannot generate a report.")
        html_content = "<html><body><h1>Backtest Report</h1><p>No trades were executed in this backtest.</p></body></html>"
        with open(filename, 'w') as f: f.write(html_content)
        return

    # --- KPI Calculations ---
    initial_capital = config['INITIAL_CAPITAL']
    total_pnl = all_trades['PnL'].sum()
    final_equity = initial_capital + total_pnl
    total_return_pct = (total_pnl / initial_capital) * 100
    
    winning_trades = all_trades[all_trades['PnL'] > 0]
    losing_trades = all_trades[all_trades['PnL'] <= 0]
    win_rate = (len(winning_trades) / len(all_trades)) * 100 if not all_trades.empty else 0
    
    gross_profit = winning_trades['PnL'].sum()
    gross_loss = abs(losing_trades['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_win = winning_trades['PnL'].mean() if not winning_trades.empty else 0
    avg_loss = abs(losing_trades['PnL'].mean()) if not losing_trades.empty else 0
    risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # --- Drawdown Calculation ---
    running_max = portfolio_equity.cummax()
    drawdown = (running_max - portfolio_equity) / running_max.replace(0, np.nan)
    max_drawdown_pct = drawdown.max() * 100 if not drawdown.empty else 0

    # --- HTML Assembly ---
    kpi_grid = f"""
        <div class="kpi-box"><div class="kpi-title">Total Return</div><div class="kpi-value {'positive' if total_return_pct >= 0 else 'negative'}">{total_return_pct:.2f}%</div></div>
        <div class="kpi-box"><div class="kpi-title">Total PnL</div><div class="kpi-value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:,.2f}</div></div>
        <div class="kpi-box"><div class="kpi-title">Max Drawdown</div><div class="kpi-value negative">{max_drawdown_pct:.2f}%</div></div>
        <div class="kpi-box"><div class="kpi-title">Win Rate</div><div class="kpi-value {'positive' if win_rate > 50 else 'negative'}">{win_rate:.2f}%</div></div>
        <div class="kpi-box"><div class="kpi-title">Profit Factor</div><div class="kpi-value {'positive' if profit_factor > 1 else 'negative'}">{profit_factor:.2f}</div></div>
        <div class="kpi-box"><div class="kpi-title">Avg Win / Loss</div><div class="kpi-value {'positive' if risk_reward_ratio > 1 else 'negative'}">{risk_reward_ratio:.2f}</div></div>
        <div class="kpi-box"><div class="kpi-title">Total Trades</div><div class="kpi-value">{len(all_trades)}</div></div>
    """
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=portfolio_equity.index, y=portfolio_equity, mode='lines', name='Portfolio Equity'))
    fig_equity.update_layout(title_text='Portfolio Equity Curve', template='plotly_dark')
    equity_chart_html = fig_equity.to_html(full_html=False, include_plotlyjs='cdn')
    
    params_html = "<h3>Parameters Used</h3><table class='trade-table'>"
    for key, value in config.items(): params_html += f"<tr><td>{key}</td><td>{str(value)}</td></tr>"
    params_html += "</table>"
    
    html = f"""
    <html><head><title>Consolidated Backtest Report</title>
    <style>
        :root {{ --bg-color: #1e1e1e; --primary-text: #d4d4d4; --secondary-text: #8c8c8c; --card-bg: #2a2a2a; --border-color: #444; --positive: #28a745; --negative: #dc3545; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: var(--bg-color); color: var(--primary-text); }}
        h1, h2, h3 {{ text-align: center; color: var(--primary-text); font-weight: 300; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .kpi-box {{ background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 15px; text-align: center; }}
        .kpi-title {{ font-weight: 500; font-size: 0.9em; color: var(--secondary-text); margin-bottom: 8px; }}
        .kpi-value {{ font-size: 1.6em; font-weight: 700; }}
        .positive {{ color: var(--positive); }} .negative {{ color: var(--negative); }}
        .chart-container, .table-container {{ background-color: var(--card-bg); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color); margin-bottom: 20px; }}
        .trade-table {{ width: 100%; border-collapse: collapse; }}
        .trade-table th, .trade-table td {{ border-bottom: 1px solid var(--border-color); padding: 12px 15px; text-align: left; }}
        .trade-table th {{ background-color: #333; }}
    </style>
    </head><body>
    <h1>Consolidated Backtest Report</h1>
    <h2>Key Performance Indicators</h2><div class="kpi-grid">{kpi_grid}</div>
    <div class="chart-container">{equity_chart_html}</div>
    <div class="table-container"><h3>Configuration</h3>{params_html}</div>
    <div class="table-container"><h3>All Trades</h3>{all_trades.to_html(classes='trade-table')}</div>
    </body></html>
    """
    with open(filename, "w") as f: f.write(html)
    log.info(f"Consolidated report generated: {filename}")

if __name__ == '__main__':
    config = setup_config()
    try:
        capital_input = input(f"Enter initial capital (or press Enter for default ${config['INITIAL_CAPITAL']}): ")
        if capital_input: config['INITIAL_CAPITAL'] = float(capital_input)
    except (ValueError, EOFError):
        log.warning("Invalid input. Using default capital.")
    
    data_df = fetch_historical_data(config)
    
    if data_df is not None:
        all_stats, all_trades, equity_curves = [], [], {}
        
        for symbol in config['SYMBOLS']:
            print(f"--- Backtesting for {symbol} ---")
            symbol_data = data_df[data_df['symbol'] == symbol].copy()
            if symbol_data.empty: continue
            
            symbol_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            symbol_data.set_index(pd.to_datetime(symbol_data['open_time']), inplace=True)

            bt = Backtest(symbol_data, EmaBBStrategy, cash=config['INITIAL_CAPITAL'], commission=config['BINANCE_FEE'])
            
            stats = bt.run(
                ema_len=config['EMA_LEN'], emasignal_backcandles=config['EMASIGNAL_BACKCANDLES'],
                bb_length=config['BB_LENGTH_CUSTOM'], bb_std=config['BB_STD_CUSTOM'],
                rsi_len=config['RSI_LEN'], max_trade_age_bars=config['MAX_TRADE_AGE_BARS'],
                sl_buffer_pct=config['SL_BUFFER_PCT'], tp1_rr=config['TP1_RR']
            )
            
            all_stats.append(stats)
            trades = stats['_trades']
            if not trades.empty:
                trades['Symbol'] = symbol
                all_trades.append(trades)
            equity_curves[symbol] = stats['_equity_curve']['Equity']
            
        if all_trades:
            portfolio_equity = pd.DataFrame(equity_curves).ffill().sum(axis=1) - (config['INITIAL_CAPITAL'] * (len(config['SYMBOLS']) - 1))
            generate_consolidated_report(all_stats, pd.concat(all_trades), portfolio_equity, config)
        else:
            print("No trades were made across all symbols.")
