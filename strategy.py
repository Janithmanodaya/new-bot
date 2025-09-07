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

def detect_htf_market_structure(candles_1h: pd.DataFrame, **kwargs):
    """
    Detects HTF market structure and bias.
    """
    atr_mult = kwargs.get('atr_mult', 1.0)
    df = candles_1h.copy()
    df['atr'] = atr(df, period=kwargs.get('atr_period', 50))
    swing_highs, swing_lows = swing_highs_lows(df, lookback=kwargs.get('lookback', 50))

    side, score, bias_strength = 'neutral', 0.0, 0.0
    last_swing_high, last_swing_low = None, None
    if not swing_highs.empty: last_swing_high = swing_highs.index[-1]
    if not swing_lows.empty: last_swing_low = swing_lows.index[-1]

    # need at least two swings of each to compare
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # price comparisons (higher-highs & higher-lows -> bullish)
        hh = swing_highs['high'].iloc[-1] > swing_highs['high'].iloc[-2]
        hl = swing_lows['low'].iloc[-1] > swing_lows['low'].iloc[-2]
        ll = swing_lows['low'].iloc[-1] < swing_lows['low'].iloc[-2]
        lh = swing_highs['high'].iloc[-1] < swing_highs['high'].iloc[-2]

        is_bullish = hh and hl
        is_bearish = ll and lh

        if is_bullish:
            impulsive_legs = 0
            for i in range(1, len(swing_lows)):
                try:
                    atr_val = df['atr'].loc[swing_lows.index[i]]
                except Exception:
                    atr_val = np.nan
                if pd.isna(atr_val):
                    continue
                if swing_lows['low'].iloc[i] > swing_lows['low'].iloc[i-1]:
                    next_highs = swing_highs[swing_highs.index > swing_lows.index[i]]
                    if not next_highs.empty:
                        leg_size = next_highs['high'].iloc[0] - swing_lows['low'].iloc[i]
                        if leg_size > atr_val * atr_mult:
                            impulsive_legs += 1
            score = min(1.0, impulsive_legs / 3.0)
            side = 'buy'
            bias_strength = score

        elif is_bearish:
            impulsive_legs = 0
            for i in range(1, len(swing_highs)):
                try:
                    atr_val = df['atr'].loc[swing_highs.index[i]]
                except Exception:
                    atr_val = np.nan
                if pd.isna(atr_val):
                    continue
                if swing_highs['high'].iloc[i] < swing_highs['high'].iloc[i-1]:
                    next_lows = swing_lows[swing_lows.index > swing_highs.index[i]]
                    if not next_lows.empty:
                        leg_size = swing_highs['high'].iloc[i] - next_lows['low'].iloc[0]
                        if leg_size > atr_val * atr_mult:
                            impulsive_legs += 1
            score = min(1.0, impulsive_legs / 3.0)
            side = 'sell'
            bias_strength = score

    return {'side': side, 'score': score, 'meta': {'last_swing_high': last_swing_high, 'last_swing_low': last_swing_low, 'bias_strength': bias_strength}}

def find_order_blocks(candles: pd.DataFrame, tf: str, **kwargs):
    """
    Detects recent bullish/bearish order blocks.
    """
    atr_period = kwargs.get('atr_period', 14)
    impulse_mult = kwargs.get('impulse_mult', 1.2)
    max_width_mult = kwargs.get('max_width_mult', 4.0)
    
    df = candles.copy(); df['atr'] = atr(df, period=atr_period); df['range'] = df['high'] - df['low']
    order_blocks = []
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] < df['open'].iloc[i-1] and df['range'].iloc[i] > impulse_mult * df['atr'].iloc[i] and df['close'].iloc[i] > df['open'].iloc[i]:
            ob_zone = [df['high'].iloc[i-1], df['low'].iloc[i-1]]
            if (ob_zone[0] - ob_zone[1]) > 0 and (ob_zone[0] - ob_zone[1]) < max_width_mult * df['atr'].iloc[i-1]:
                width = ob_zone[0] - ob_zone[1]
                atr_ref = max(1e-8, df['atr'].iloc[i-1])
                width_score = max(0.0, 1.0 - (width / (max_width_mult * atr_ref)))
                strength = float(min(1.0, max(0.0, width_score)))
                order_blocks.append({'type': 'bull', 'zone': ob_zone, 'origin_idx': i-1, 'strength': strength})
        elif df['close'].iloc[i-1] > df['open'].iloc[i-1] and df['range'].iloc[i] > impulse_mult * df['atr'].iloc[i] and df['close'].iloc[i] < df['open'].iloc[i]:
            ob_zone = [df['high'].iloc[i-1], df['low'].iloc[i-1]]
            if (ob_zone[0] - ob_zone[1]) > 0 and (ob_zone[0] - ob_zone[1]) < max_width_mult * df['atr'].iloc[i-1]:
                width = ob_zone[0] - ob_zone[1]
                atr_ref = max(1e-8, df['atr'].iloc[i-1])
                width_score = max(0.0, 1.0 - (width / (max_width_mult * atr_ref)))
                strength = float(min(1.0, max(0.0, width_score)))
                order_blocks.append({'type': 'bear', 'zone': ob_zone, 'origin_idx': i-1, 'strength': strength})
    return order_blocks

def find_fair_value_gaps(candles: pd.DataFrame, tf: str, **kwargs):
    """
    Detects Fair Value Gaps (FVGs) or imbalances using a standard 3-candle pattern.
    """
    df = candles.copy()
    df['atr'] = atr(df)
    fvgs = []
    for i in range(len(df) - 2):
        if df['low'].iloc[i+2] > df['high'].iloc[i]:
            gap_size = df['low'].iloc[i+2] - df['high'].iloc[i]
            atr_ref = df['atr'].iloc[i+1] if pd.notna(df['atr'].iloc[i+1]) and df['atr'].iloc[i+1] > 0 else gap_size
            strength = float(min(1.0, gap_size / max(1e-9, atr_ref)))
            fvgs.append({'side': 'buy', 'zone': [df['high'].iloc[i], df['low'].iloc[i+2]], 'origin_idx': i + 1, 'strength': strength})
        elif df['high'].iloc[i+2] < df['low'].iloc[i]:
            gap_size = df['low'].iloc[i] - df['high'].iloc[i+2]
            atr_ref = df['atr'].iloc[i+1] if pd.notna(df['atr'].iloc[i+1]) and df['atr'].iloc[i+1] > 0 else gap_size
            strength = float(min(1.0, gap_size / max(1e-9, atr_ref)))
            fvgs.append({'side': 'sell', 'zone': [df['low'].iloc[i], df['high'].iloc[i+2]], 'origin_idx': i + 1, 'strength': strength})
    return fvgs

def detect_liquidity_sweep(candles_5m: pd.DataFrame, **kwargs):
    lookback = kwargs.get('lookback', 20)
    atr_period = kwargs.get('atr_period', 15)
    atr_mult = kwargs.get('atr_mult', 0.5)
    
    df = candles_5m.copy()
    if len(df) < 4:
        return {'side': 'neutral', 'score': 0.0, 'meta': {}}
    # compute atr on the 5m candles if not present (uses local atr function)
    local_atr = atr(df, period=atr_period)
    recent_high = df['high'].rolling(lookback).max().iloc[-2]
    recent_low  = df['low'].rolling(lookback).min().iloc[-2]
    atr_val = local_atr.iloc[-2] if pd.notna(local_atr.iloc[-2]) else (df['high'].iloc[-2] - df['low'].iloc[-2])

    # examine last 3 candles for a wick beyond recent extreme then close back inside
    for look in range(3, 0, -1):
        idx = -look
        candle = df.iloc[idx]
        # upward sweep: wick above recent_high by > atr_mult*atr_val
        if candle['high'] - recent_high > atr_mult * atr_val and candle['close'] < recent_high:
            # score proportional to overshoot
            score = min(1.0, (candle['high'] - recent_high) / (2.0 * atr_val))
            return {'side': 'buy', 'score': float(score), 'meta': {'wick_extreme': candle['high'], 'wick_idx': df.index[idx]}}
        # downward sweep: wick below recent_low by > atr_mult*atr_val
        if recent_low - candle['low'] > atr_mult * atr_val and candle['close'] > recent_low:
            score = min(1.0, (recent_low - candle['low']) / (2.0 * atr_val))
            return {'side': 'sell', 'score': float(score), 'meta': {'wick_extreme': candle['low'], 'wick_idx': df.index[idx]}}
    return {'side': 'neutral', 'score': 0.0, 'meta': {}}

def compute_ote(impulsive_leg_candles: pd.DataFrame):
    """
    Approximate OTE: take recent local extreme (max/min) and compute 61.8-79% zone.
    """
    df = impulsive_leg_candles.copy()
    if df.empty or len(df) < 3:
        return {'in_ote': False, 'retrace_pct': 0.0, 'ote_zone': [0, 0], 'score': 0.0}
    high = df['high'].max()
    low = df['low'].min()
    # guess direction by comparing last close to mid
    if df['close'].iloc[-1] >= (high + low) / 2:
        # impulsive up move: OTE zone below the high
        start, end = low, high
        direction = 'up'
    else:
        # impulsive down move: OTE zone above the low
        start, end = high, low
        direction = 'down'

    move = end - start
    if abs(move) < 1e-8:
        return {'in_ote': False, 'retrace_pct': 0.0, 'ote_zone': [0, 0], 'score': 0.0}

    ote_high = end - 0.618 * move if direction == 'up' else end + 0.618 * move
    ote_low  = end - 0.79 * move  if direction == 'up' else end + 0.79 * move
    ote_zone = [max(ote_high, ote_low), min(ote_high, ote_low)]  # [upper, lower]
    price = df['close'].iloc[-1]
    in_ote = (ote_zone[1] <= price <= ote_zone[0])
    # proximity score to center of OTE
    center = (ote_zone[0] + ote_zone[1]) / 2
    half_width = max(1e-9, (ote_zone[0] - ote_zone[1]) / 2)
    proximity = 1.0 - min(1.0, abs(price - center) / half_width)
    score = float(max(0.0, proximity)) if in_ote else 0.0
    retrace_pct = abs((price - end) / move)
    return {'in_ote': bool(in_ote), 'retrace_pct': float(retrace_pct), 'ote_zone': ote_zone, 'score': float(score)}

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

def momentum_volume_filter(candles: pd.DataFrame, tf: str, **kwargs):
    """
    Graded momentum: uses body-size ratio between last two candles.
    """
    consecutive_candles = kwargs.get('consecutive_candles', 2)
    ratio_min = kwargs.get('ratio_min', 0.5)
    score_divisor = kwargs.get('score_divisor', 1.2)
    
    df = candles.copy()
    df['body_size'] = (df['close'] - df['open']).abs()
    if len(df) < consecutive_candles:
        return {'side': 'neutral', 'score': 0.0, 'momentum_strength': 0.0}

    last_candles = df.iloc[-consecutive_candles:]
    
    if all(last_candles['close'] > last_candles['open']):
        side = 'buy'
    elif all(last_candles['close'] < last_candles['open']):
        side = 'sell'
    else:
        return {'side': 'neutral', 'score': 0.0, 'momentum_strength': 0.0}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    ratio = last['body_size'] / max(prev['body_size'], 1e-9)
    score = min(1.0, max(0.0, (ratio - ratio_min) / score_divisor))
    return {'side': side, 'score': float(score), 'momentum_strength': float(score)}

def candle_body_confirmation(last_candle: pd.Series, zone, side: str = None):
    """
    Requires candle body close confirmation (not just wick).
    zone can be any [a, b] pair; we normalize to (upper, lower).
    If side is provided ('buy'/'sell'), confirmation is made stricter for that side.
    """
    if zone is None or last_candle is None:
        return {'confirmed': False, 'score': 0.0}

    upper = max(zone)
    lower = min(zone)
    close = last_candle['close']
    open_ = last_candle['open']

    # For BUY: prefer close above the zone upper bound or bullish close inside zone
    if side == 'buy':
        confirmed = (close > upper) or (close > lower and close > open_)
    # For SELL: prefer close below the zone lower bound or bearish close inside zone
    elif side == 'sell':
        confirmed = (close < lower) or (close < upper and close < open_)
    else:
        # Generic confirmation: close outside the zone (either side) or bullish/bearish body inside
        confirmed = (close > upper) or (close < lower) or (close > open_) or (close < open_)

    return {'confirmed': bool(confirmed), 'score': 1.0 if confirmed else 0.0}

def session_and_spread_filter(timestamp, spread: float = 0.0):
    """
    Allows/denies signal based on session/time and spread.
    Spread is a placeholder as it's not available in historical data.
    """
    allowed = True  # Allow 24/7 trading for crypto
    if spread > 0.001: allowed = False
    return {'allowed': allowed, 'score': 1.0 if allowed else 0.0}

def compute_confluence_score(detection_results: dict, weights: dict):
    """
    Computes the composite confluence score from all module results.
    """
    composite_score, votes = 0, {}
    for module, result in detection_results.items():
        if result and 'score' in result:
            score = result['score'] * weights.get(module, 1.0)
            composite_score += score; votes[module] = score
    return {'composite_score': composite_score, 'votes': votes}

def plan_entry_action(symbol: str, side: str, composite_score: float, zones: dict, price: float, atr_15m: float, config: dict):
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
        buffer = max(config['buffer_atr_mult'] * atr_15m, config['buffer_price_pct'] * price)
        stop_loss = ob_meta['zone'][1] - buffer if side == 'buy' else ob_meta['zone'][0] + buffer
    else:
        sl_distance = 1.2 * atr_15m
        stop_loss = price - sl_distance if side == 'buy' else price + sl_distance

    # --- Minimum SL Guard ---
    min_sl_dist_pct = config['min_sl_dist_pct']
    min_sl_dist = price * min_sl_dist_pct
    if abs(price - stop_loss) < min_sl_dist:
        if side == 'buy':
            stop_loss = price - min_sl_dist
        else:
            stop_loss = price + min_sl_dist
            
    entry_to_sl_dist = abs(entry_price - stop_loss)
    tp1 = entry_price + config['tp1_rr'] * entry_to_sl_dist if side == 'buy' else entry_price - config['tp1_rr'] * entry_to_sl_dist
    tp2 = entry_price + config['tp2_rr'] * entry_to_sl_dist if side == 'buy' else entry_price - config['tp2_rr'] * entry_to_sl_dist
    return {'entry_type': entry_type, 'entry_price': entry_price, 'stop_loss': stop_loss, 'targets': [tp1, tp2], 'exit_logic': 'Standard SL/TP with trailing'}
