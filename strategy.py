import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import time

# --- Helper Functions ---

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR).
    """
    if df.empty:
        return pd.Series(dtype=float)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def swing_highs_lows(df: pd.DataFrame, lookback: int = 20):
    """
    Finds swing highs and lows using scipy's find_peaks.
    A swing high is a peak higher than its neighbors.
    A swing low is a trough lower than its neighbors.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    high_peaks_indices, _ = find_peaks(df['high'], distance=lookback)
    swing_highs = df.iloc[high_peaks_indices]
    low_peaks_indices, _ = find_peaks(-df['low'], distance=lookback)
    swing_lows = df.iloc[low_peaks_indices]
    return swing_highs, swing_lows

# --- Strategy Modules ---
# Each module is designed to return a dictionary with the following keys:
# 'side': 'buy', 'sell', or 'neutral'
# 'score': A float from 0.0 to 1.0 indicating the strength of the signal.
# 'meta': A dictionary containing any other relevant information from the module.

def detect_htf_market_structure(candles_1h: pd.DataFrame, lookback: int = 50, atr_period: int = 50, atr_mult: float = 1.0):
    """
    Detects HTF market structure and bias.
    """
    df = candles_1h.copy()
    df['atr'] = atr(df, period=atr_period)
    swing_highs, swing_lows = swing_highs_lows(df, lookback=lookback)

    side, score, bias_strength = 'neutral', 0.0, 0
    last_swing_high, last_swing_low = None, None
    if not swing_highs.empty: last_swing_high = swing_highs.index[-1]
    if not swing_lows.empty: last_swing_low = swing_lows.index[-1]

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        is_bullish = (swing_highs.index[-1] > swing_highs.index[-2]) and (swing_lows.index[-1] > swing_lows.index[-2]) and (swing_highs['high'].iloc[-1] > swing_highs['high'].iloc[-2]) and (swing_lows['low'].iloc[-1] > swing_lows['low'].iloc[-2])
        is_bearish = (swing_highs.index[-1] > swing_highs.index[-2]) and (swing_lows.index[-1] > swing_lows.index[-2]) and (swing_highs['high'].iloc[-1] < swing_highs['high'].iloc[-2]) and (swing_lows['low'].iloc[-1] < swing_lows['low'].iloc[-2])

        if is_bullish:
            impulsive_legs = 0
            for i in range(1, len(swing_lows)):
                if swing_lows['low'].iloc[i] > swing_lows['low'].iloc[i-1]:
                    next_highs = swing_highs[swing_highs.index > swing_lows.index[i]]
                    if not next_highs.empty:
                        leg_size = next_highs['high'].iloc[0] - swing_lows['low'].iloc[i]
                        if leg_size > df['atr'].loc[swing_lows.index[i]] * atr_mult:
                            impulsive_legs += 1
            side, score, bias_strength = 'buy', min(1.0, impulsive_legs / 3.0), score
        elif is_bearish:
            impulsive_legs = 0
            for i in range(1, len(swing_highs)):
                if swing_highs['high'].iloc[i] < swing_highs['high'].iloc[i-1]:
                    next_lows = swing_lows[swing_lows.index > swing_highs.index[i]]
                    if not next_lows.empty:
                        leg_size = swing_highs['high'].iloc[i] - next_lows['low'].iloc[0]
                        if leg_size > df['atr'].loc[swing_highs.index[i]] * atr_mult:
                            impulsive_legs += 1
            side, score, bias_strength = 'sell', min(1.0, impulsive_legs / 3.0), score

    return {'side': side, 'score': score, 'meta': {'last_swing_high': last_swing_high, 'last_swing_low': last_swing_low, 'bias_strength': bias_strength}}

def find_order_blocks(candles: pd.DataFrame, tf: str, atr_period: int = 14, impulse_mult: float = 1.2, max_width_mult: float = 4.0):
    """
    Detects recent bullish/bearish order blocks.
    """
    df = candles.copy(); df['atr'] = atr(df, period=atr_period); df['range'] = df['high'] - df['low']
    order_blocks = []
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] < df['open'].iloc[i-1] and df['range'].iloc[i] > impulse_mult * df['atr'].iloc[i] and df['close'].iloc[i] > df['open'].iloc[i]:
            ob_zone = [df['open'].iloc[i-1], df['low'].iloc[i-1]]
            if (ob_zone[0] - ob_zone[1]) > 0 and (ob_zone[0] - ob_zone[1]) < max_width_mult * df['atr'].iloc[i-1]:
                order_blocks.append({'type': 'bull', 'zone': ob_zone, 'origin_idx': i-1, 'strength': 1.0})
        elif df['close'].iloc[i-1] > df['open'].iloc[i-1] and df['range'].iloc[i] > impulse_mult * df['atr'].iloc[i] and df['close'].iloc[i] < df['open'].iloc[i]:
            ob_zone = [df['high'].iloc[i-1], df['open'].iloc[i-1]]
            if (ob_zone[0] - ob_zone[1]) > 0 and (ob_zone[0] - ob_zone[1]) < max_width_mult * df['atr'].iloc[i-1]:
                order_blocks.append({'type': 'bear', 'zone': ob_zone, 'origin_idx': i-1, 'strength': 1.0})
    return order_blocks

def find_fair_value_gaps(candles: pd.DataFrame, tf: str):
    """
    Detects Fair Value Gaps (FVGs) or imbalances using a standard 3-candle pattern.
    """
    df, fvgs = candles.copy(), []
    for i in range(len(df) - 2):
        if df['low'].iloc[i+2] > df['high'].iloc[i]:
            fvgs.append({'side': 'buy', 'zone': [df['high'].iloc[i], df['low'].iloc[i+2]], 'origin_idx': i + 1, 'strength': 1.0})
        elif df['high'].iloc[i+2] < df['low'].iloc[i]:
             fvgs.append({'side': 'sell', 'zone': [df['low'].iloc[i], df['high'].iloc[i+2]], 'origin_idx': i + 1, 'strength': 1.0})
    return fvgs

def detect_liquidity_sweep(candles_5m: pd.DataFrame, lookback: int = 20, atr_period: int = 15, atr_mult: float = 0.5):
    """
    Detects a quick wick beyond a swing high/low with a fast rejection.
    NOTE: This is a placeholder and needs to be implemented with more advanced logic.
    """
    return {'side': 'neutral', 'score': 0.0, 'meta': {}}

def compute_ote(impulsive_leg_candles: pd.DataFrame):
    """
    Computes OTE (Optimal Trade Entry) zone of the last impulsive leg.
    NOTE: This is a placeholder and needs to be implemented.
    """
    return {'in_ote': False, 'retrace_pct': 0.0, 'ote_zone': [0,0], 'score': 0.0}

def detect_breaker_blocks(candles: pd.DataFrame, tf: str):
    """
    Finds breaker/mitigation blocks.
    NOTE: This is a placeholder and needs to be implemented.
    """
    return {'type': 'neutral', 'zone': [0,0], 'score': 0.0}

def institutional_imbalance_vwap(candles: pd.DataFrame, volume_profile=None):
    """
    Detects confluence with VWAP or high-volume nodes.
    Volume profile is not available from basic k-line data. This is a simplified version.
    """
    df = candles.copy()
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    else:
        df['vwap'] = df['close'].expanding().mean()
    return {'vwap_confluence': False, 'hvn_distance': 0.0, 'score': 0.0}

def momentum_volume_filter(candles: pd.DataFrame, tf: str):
    """
    Confirms directional momentum via consecutive ranged candles and volume.
    """
    df = candles.copy(); df['body_size'] = abs(df['close'] - df['open'])
    if len(df) < 2: return {'side': 'neutral', 'score': 0.0, 'momentum_strength': 0.0}
    last, prev = df.iloc[-1], df.iloc[-2]
    side, score = 'neutral', 0.0
    if last['close'] > last['open'] and prev['close'] > prev['open'] and last['body_size'] > prev['body_size']:
        side, score = 'buy', 1.0
    elif last['close'] < last['open'] and prev['close'] < prev['open'] and last['body_size'] > prev['body_size']:
        side, score = 'sell', 1.0
    return {'side': side, 'score': score, 'momentum_strength': score}

def candle_body_confirmation(last_candle: pd.Series, zone):
    """
    Requires candle body close confirmation (not just wick).
    """
    if zone is None or last_candle is None: return {'confirmed': False, 'score': 0.0}
    confirmed = last_candle['close'] > zone[0] or last_candle['close'] < zone[1]
    return {'confirmed': confirmed, 'score': 1.0 if confirmed else 0.0}

def session_and_spread_filter(timestamp, spread: float = 0.0):
    """
    Allows/denies signal based on session/time and spread.
    Spread is a placeholder as it's not available in historical data.
    """
    london_open, london_close = time(7, 0), time(16, 0)
    ny_open, ny_close = time(13, 0), time(22, 0)
    ts_time = timestamp.time()
    allowed = (london_open <= ts_time <= london_close) or (ny_open <= ts_time <= ny_close)
    if spread > 0.001: allowed = False
    return {'allowed': allowed, 'score': 1.0 if allowed else 0.0}

def compute_confluence_score(detection_results: dict):
    """
    Computes the composite confluence score from all module results.
    """
    weights = {'htf_market_bias': 1.2, 'order_block': 1.5, 'fvg': 1.3, 'liquidity_sweep': 1.4, 'ote': 1.2, 'breaker': 1.0, 'vwap_hvn': 1.0, 'momentum': 1.0, 'candle_body_confirm': 1.0, 'session_spread': 0.6}
    composite_score, votes = 0, {}
    for module, result in detection_results.items():
        if result and 'score' in result:
            score = result['score'] * weights.get(module, 1.0)
            composite_score += score; votes[module] = score
    return {'composite_score': composite_score, 'votes': votes}

def plan_entry_action(symbol: str, side: str, composite_score: float, zones: dict, price: float, atr_15m: float):
    """
    Plans the entry, stop loss, and take profit based on the strategy rules.
    """
    entry_type = 'market' if composite_score > 3.5 else 'limit'
    entry_price = price
    ob_meta = zones.get('order_block', {}).get('meta')
    if entry_type == 'limit':
        if ob_meta and 'zone' in ob_meta:
            entry_price = (ob_meta['zone'][0] + ob_meta['zone'][1]) / 2.0
        else: entry_type = 'market' 
    stop_loss = 0.0
    if ob_meta and 'zone' in ob_meta:
        buffer = max(0.4 * atr_15m, 0.0005 * price)
        stop_loss = ob_meta['zone'][1] - buffer if side == 'buy' else ob_meta['zone'][0] + buffer
    else:
        sl_distance = 1.2 * atr_15m
        stop_loss = price - sl_distance if side == 'buy' else price + sl_distance

    # --- Minimum SL Guard ---
    min_sl_dist_pct = 0.001 # 0.1% of price
    min_sl_dist = price * min_sl_dist_pct
    if abs(price - stop_loss) < min_sl_dist:
        if side == 'buy':
            stop_loss = price - min_sl_dist
        else:
            stop_loss = price + min_sl_dist
            
    entry_to_sl_dist = abs(entry_price - stop_loss)
    tp1 = entry_price + entry_to_sl_dist if side == 'buy' else entry_price - entry_to_sl_dist
    tp2 = entry_price + 2.5 * entry_to_sl_dist if side == 'buy' else entry_price - 2.5 * entry_to_sl_dist
    return {'entry_type': entry_type, 'entry_price': entry_price, 'stop_loss': stop_loss, 'targets': [tp1, tp2], 'exit_logic': 'Standard SL/TP with trailing'}
