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
    highs, lows = swing_highs_lows(df, lookback=kwargs.get('lookback', 50))

    side, score, bias_strength = 'neutral', 0.0, 0.0
    last_swing_high, last_swing_low = None, None
    if not highs.empty: last_swing_high = highs.index[-1]
    if not lows.empty: last_swing_low = lows.index[-1]

    # need at least two swings of each to compare
    if len(highs) >= 2 and len(lows) >= 2:
        # price comparisons (higher-highs & higher-lows -> bullish)
        hh = highs['high'].iloc[-1] > highs['high'].iloc[-2]
        hl = lows['low'].iloc[-1] > lows['low'].iloc[-2]
        ll = lows['low'].iloc[-1] < lows['low'].iloc[-2]
        lh = highs['high'].iloc[-1] < highs['high'].iloc[-2]

        is_bullish = hh and hl
        is_bearish = ll and lh

        if is_bullish:
            impulsive_legs = 0
            for i in range(1, len(lows)):
                try:
                    atr_val = df['atr'].loc[lows.index[i]]
                except Exception:
                    atr_val = np.nan
                if pd.isna(atr_val):
                    continue
                if lows['low'].iloc[i] > lows['low'].iloc[i-1]:
                    next_highs = highs[highs.index > lows.index[i]]
                    if not next_highs.empty:
                        leg_size = next_highs['high'].iloc[0] - lows['low'].iloc[i]
                        if leg_size > atr_val * atr_mult:
                            impulsive_legs += 1
            score = min(1.0, impulsive_legs / 3.0)
            side = 'buy'
            bias_strength = score

        elif is_bearish:
            impulsive_legs = 0
            for i in range(1, len(highs)):
                try:
                    atr_val = df['atr'].loc[highs.index[i]]
                except Exception:
                    atr_val = np.nan
                if pd.isna(atr_val):
                    continue
                if highs['high'].iloc[i] < highs['high'].iloc[i-1]:
                    next_lows = lows[lows.index > highs.index[i]]
                    if not next_lows.empty:
                        leg_size = highs['high'].iloc[i] - next_lows['low'].iloc[0]
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
        if pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] <= 0:
            continue
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
        atr_ref = df['atr'].iloc[i+1]
        if pd.isna(atr_ref) or atr_ref <= 0:
            continue

        if df['low'].iloc[i+2] > df['high'].iloc[i]:
            gap_size = df['low'].iloc[i+2] - df['high'].iloc[i]
            strength = float(min(1.0, gap_size / atr_ref))
            # Consistent zone ordering: [upper, lower]
            fvgs.append({'side': 'buy', 'zone': [df['low'].iloc[i+2], df['high'].iloc[i]], 'origin_idx': i + 1, 'strength': strength})
        elif df['high'].iloc[i+2] < df['low'].iloc[i]:
            gap_size = df['low'].iloc[i] - df['high'].iloc[i+2]
            strength = float(min(1.0, gap_size / atr_ref))
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
    Calculates the Optimal Trade Entry (OTE) zone (0.618-0.79 Fibonacci retracement)
    based on the most recent impulsive move.
    """
    df = impulsive_leg_candles.copy()
    if df.empty or len(df) < 3:
        return {'in_ote': False, 'retrace_pct': 0.0, 'ote_zone': [0, 0], 'score': 0.0}

    impulse_high = df['high'].max()
    impulse_low = df['low'].min()
    price = df['close'].iloc[-1]

    # Determine the direction of the primary impulse
    is_up_impulse = price >= (impulse_high + impulse_low) / 2
    
    move_size = impulse_high - impulse_low
    if move_size < 1e-8:
        return {'in_ote': False, 'retrace_pct': 0.0, 'ote_zone': [0, 0], 'score': 0.0}

    if is_up_impulse:
        ote_high = impulse_high - 0.618 * move_size
        ote_low  = impulse_high - 0.79 * move_size
        retrace_pct = (impulse_high - price) / move_size if move_size > 0 else 0
    else: # Down impulse
        ote_high = impulse_low + 0.79 * move_size
        ote_low  = impulse_low + 0.618 * move_size
        retrace_pct = (price - impulse_low) / move_size if move_size > 0 else 0

    ote_zone = [ote_high, ote_low]
    in_ote = (ote_low <= price <= ote_high)

    # Score based on proximity to the 0.705 level (center of the OTE)
    score = 0.0
    if in_ote:
        center = (ote_high + ote_low) / 2
        half_width = (ote_high - ote_low) / 2
        if half_width > 1e-9:
            proximity = 1.0 - min(1.0, abs(price - center) / half_width)
            score = float(max(0.0, proximity))

    return {'in_ote': bool(in_ote), 'retrace_pct': float(retrace_pct), 'ote_zone': ote_zone, 'score': score}

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

def session_and_spread_filter(spread: float = 0.0):
    """
    Allows/denies signal based on session/time and spread.
    Spread is a placeholder as it's not available in historical data.
    """
    allowed = True  # Allow 24/7 trading for crypto
    if spread > 0.001: allowed = False
    return {'allowed': allowed, 'score': 1.0 if allowed else 0.0}

def micro_sniper_trigger(df_1m: pd.DataFrame, **kwargs):
    """
    Detects a high-probability "sniper" entry based on a micro-timeframe (1m)
    wick rejection combined with a significant volume spike.
    """
    lookback = kwargs.get('lookback', 20)
    wick_atr_mult = kwargs.get('wick_atr_mult', 0.25)
    volume_z_k = kwargs.get('volume_z_k', 1.5)
    atr_period = kwargs.get('atr_period', 14)

    if len(df_1m) < lookback + 2:
        return {'side': 'neutral', 'score': 0.0, 'meta': {}}

    df = df_1m.copy()
    df['atr'] = atr(df, period=atr_period)

    hist = df.iloc[-lookback-1:-1] # Historical data for lookback
    last_candle = df.iloc[-1]      # The current candle to check

    recent_high = hist['high'].max()
    recent_low = hist['low'].min()
    atr_val = hist['atr'].iloc[-1] if pd.notna(hist['atr'].iloc[-1]) else None
    if atr_val is None or atr_val <= 0:
        return {'side': 'neutral', 'score': 0.0, 'meta': {}}
    
    vol_mean = hist['volume'].mean()
    vol_std = hist['volume'].std()

    if vol_std is None or np.isnan(vol_std) or vol_std == 0: # Avoid division by zero
        return {'side': 'neutral', 'score': 0.0, 'meta': {}}

    volume_spike = last_candle['volume'] >= (vol_mean + volume_z_k * vol_std)

    # Bearish signal: high wick rejection + volume spike
    high_wick_overshoot = last_candle['high'] - recent_high
    if high_wick_overshoot > (wick_atr_mult * atr_val) and last_candle['close'] < recent_high and volume_spike:
        side = 'sell'
        wick_score = min(1.0, high_wick_overshoot / atr_val if atr_val > 0 else 1.0)
        volume_z_score = (last_candle['volume'] - vol_mean) / vol_std
        score = wick_score * (volume_z_score / 4.0) # Scale to keep it in a reasonable 0-1 range
        return {'side': side, 'score': float(score), 'meta': {'type': 'wick_rejection_high', 'overshoot': high_wick_overshoot}}

    # Bullish signal: low wick rejection + volume spike
    low_wick_overshoot = recent_low - last_candle['low']
    if low_wick_overshoot > (wick_atr_mult * atr_val) and last_candle['close'] > recent_low and volume_spike:
        side = 'buy'
        wick_score = min(1.0, low_wick_overshoot / atr_val if atr_val > 0 else 1.0)
        volume_z_score = (last_candle['volume'] - vol_mean) / vol_std
        score = wick_score * (volume_z_score / 4.0)
        return {'side': side, 'score': float(score), 'meta': {'type': 'wick_rejection_low', 'overshoot': low_wick_overshoot}}
    
    return {'side': 'neutral', 'score': 0.0, 'meta': {}}

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
    Includes special handling for "sniper" entries with tighter stop losses.
    """
    entry_price = price
    stop_loss = 0.0
    
    # Check if this is a sniper entry
    is_sniper_entry = zones.get('micro_sniper_trigger', {}).get('score', 0) > 0.6

    # Determine Entry Price
    entry_type = 'market' if composite_score > 3.5 else 'limit'
    ob_meta = zones.get('order_block', {}).get('meta')
    if entry_type == 'limit':
        if ob_meta and 'zone' in ob_meta:
            entry_price = (ob_meta['zone'][0] + ob_meta['zone'][1]) / 2.0
        else:
            entry_type = 'market'

    # Determine Stop Loss
    if is_sniper_entry:
        # Tighter SL for sniper entries
        sl_distance = atr_15m * 0.4 # As suggested by user
        stop_loss = entry_price - sl_distance if side == 'buy' else entry_price + sl_distance
    elif ob_meta and 'zone' in ob_meta:
        # Standard OB-based SL
        buffer = max(config['buffer_atr_mult'] * atr_15m, config['buffer_price_pct'] * price)
        stop_loss = ob_meta['zone'][1] - buffer if side == 'buy' else ob_meta['zone'][0] + buffer
    else:
        # Fallback SL based on ATR
        sl_distance = 1.2 * atr_15m
        stop_loss = entry_price - sl_distance if side == 'buy' else entry_price + sl_distance

    # --- SL Guards ---
    # 1. Minimum SL distance to avoid zero or tiny SLs
    min_sl_dist = entry_price * config['min_sl_dist_pct']
    if abs(entry_price - stop_loss) < min_sl_dist:
        stop_loss = entry_price - min_sl_dist if side == 'buy' else entry_price + min_sl_dist

    # 2. Maximum SL distance to cap risk, especially for snipes
    if is_sniper_entry:
        max_sl_dist = entry_price * config['max_sl_pct']
        if abs(entry_price - stop_loss) > max_sl_dist:
            stop_loss = entry_price - max_sl_dist if side == 'buy' else entry_price + max_sl_dist
            
    # Determine Take Profit targets
    entry_to_sl_dist = abs(entry_price - stop_loss)
    if entry_to_sl_dist < 1e-9: # Avoid division by zero if SL is at entry
        return None

    tp1 = entry_price + config['tp1_rr'] * entry_to_sl_dist if side == 'buy' else entry_price - config['tp1_rr'] * entry_to_sl_dist
    tp2 = entry_price + config['tp2_rr'] * entry_to_sl_dist if side == 'buy' else entry_price - config['tp2_rr'] * entry_to_sl_dist
    
    return {'entry_type': entry_type, 'entry_price': entry_price, 'stop_loss': stop_loss, 'targets': [tp1, tp2], 'exit_logic': 'Standard SL/TP with trailing'}
