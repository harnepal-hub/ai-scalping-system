"""
Step 3 — Scalping Strategy Engine
====================================
For each of the Top 5 scored stocks, calculates precise intraday trade
parameters: Entry, Stop-Loss, Take-Profit 1, Take-Profit 2, Quantity,
Max Loss, and Risk-Reward Ratio.

Capital rules:
  • Total Capital    : ₹1,00,000
  • Max positions    : 5 simultaneous
  • Capital per stock: ₹20,000
  • Max risk / trade : ₹1,000 (1% of capital)

ATR-based SL/TP (5-min ATR):
  • SL  = Entry ± 1.5 × ATR  (clamped to 0.3%–0.8% from entry)
  • TP1 = Entry ± 1.0 × ATR  (book 50% qty)
  • TP2 = Entry ± 2.5 × ATR  (book remaining 50% qty)

Entry signal conditions (ALL must be true for LONG):
  1. Price breaks above ORB High (morning) or reclaims VWAP (afternoon)
  2. EMA(8) > EMA(21) on 5-min
  3. RSI(7) between 52 and 75
  4. Current candle volume > 1.5× avg of last 10 candles
  5. Not within 15 min of market close (3:15 PM cutoff)
"""

import sys
import os
import math
import warnings
from datetime import datetime, time as dt_time
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config.config import (
        CAPITAL_PER_STOCK,
        SL_ATR_MULTIPLIER,
        TP1_ATR_MULTIPLIER,
        TP2_ATR_MULTIPLIER,
        MIN_SL_PCT,
        MAX_SL_PCT,
        RSI_LONG_MIN,
        RSI_LONG_MAX,
        RSI_SHORT_MIN,
        RSI_SHORT_MAX,
        VOLUME_SPIKE_FACTOR,
        NO_NEW_TRADES_AFTER,
        LOG_FILE,
    )
except ModuleNotFoundError:
    CAPITAL_PER_STOCK = 20_000
    SL_ATR_MULTIPLIER = 1.5
    TP1_ATR_MULTIPLIER = 1.0
    TP2_ATR_MULTIPLIER = 2.5
    MIN_SL_PCT = 0.003
    MAX_SL_PCT = 0.008
    RSI_LONG_MIN = 52
    RSI_LONG_MAX = 75
    RSI_SHORT_MIN = 25
    RSI_SHORT_MAX = 48
    VOLUME_SPIKE_FACTOR = 1.5
    NO_NEW_TRADES_AFTER = "15:00"
    LOG_FILE = "logs/trading.log"

try:
    from loguru import logger
    logger.add(LOG_FILE, rotation="1 day", retention="30 days", level="INFO")
except Exception:
    import logging
    logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_trade_setup(stock: str,
                          price: float,
                          atr: float,
                          direction: str = "LONG",
                          session: str = "morning") -> dict:
    """
    Generate a complete trade setup for a single stock.

    Parameters
    ----------
    stock : str
        Ticker symbol, e.g. "RELIANCE.NS".
    price : float
        Entry price (breakout candle close price).
    atr : float
        5-min ATR value (in ₹).
    direction : str
        "LONG" or "SHORT".
    session : str
        "morning" or "afternoon".

    Returns
    -------
    dict
        Keys: stock, direction, session, entry, sl, tp1, tp2,
              qty, max_loss, rr_ratio, sl_pct, tp1_pct, tp2_pct.
    """
    direction = direction.upper()
    sign = 1 if direction == "LONG" else -1

    # ── SL / TP raw levels ───────────────────────────────────────────────────
    raw_sl_dist = SL_ATR_MULTIPLIER * atr
    tp1_dist = TP1_ATR_MULTIPLIER * atr
    tp2_dist = TP2_ATR_MULTIPLIER * atr

    # Clamp SL distance to [MIN_SL_PCT, MAX_SL_PCT] of entry price
    min_sl_dist = MIN_SL_PCT * price
    max_sl_dist = MAX_SL_PCT * price
    sl_dist = float(np.clip(raw_sl_dist, min_sl_dist, max_sl_dist))

    if direction == "LONG":
        sl = price - sl_dist
        tp1 = price + tp1_dist
        tp2 = price + tp2_dist
    else:
        sl = price + sl_dist
        tp1 = price - tp1_dist
        tp2 = price - tp2_dist

    # ── Quantity ─────────────────────────────────────────────────────────────
    qty = max(1, math.floor(CAPITAL_PER_STOCK / price))

    # ── Max Loss ─────────────────────────────────────────────────────────────
    max_loss = round(qty * sl_dist, 2)

    # ── Risk-Reward ───────────────────────────────────────────────────────────
    reward = tp2_dist
    rr_ratio = round(reward / sl_dist, 2) if sl_dist > 0 else 0.0

    # ── Percentage distances ─────────────────────────────────────────────────
    sl_pct = round(sl_dist / price * 100, 2)
    tp1_pct = round(tp1_dist / price * 100, 2)
    tp2_pct = round(tp2_dist / price * 100, 2)

    setup = {
        "stock": stock,
        "direction": direction,
        "session": session,
        "entry": round(price, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "qty": qty,
        "max_loss": max_loss,
        "rr_ratio": rr_ratio,
        "sl_pct": sl_pct,
        "tp1_pct": tp1_pct,
        "tp2_pct": tp2_pct,
        "atr": round(atr, 4),
    }

    logger.info(
        f"Trade setup generated: {stock} {direction} "
        f"Entry={price:.2f} SL={sl:.2f} TP1={tp1:.2f} TP2={tp2:.2f} Qty={qty}"
    )
    return setup


def check_entry_signal(stock_data: pd.DataFrame,
                        session: str = "morning",
                        direction: str = "LONG",
                        current_time: Optional[datetime] = None) -> tuple:
    """
    Evaluate whether entry conditions are met for a given stock.

    All 5 conditions must be True for a valid LONG entry:
      1. Price breaks above ORB High (morning) or reclaims VWAP (afternoon)
      2. EMA(8) > EMA(21) on 5-min
      3. RSI(7) in [RSI_LONG_MIN, RSI_LONG_MAX] (or short range for SHORT)
      4. Current candle volume > VOLUME_SPIKE_FACTOR × last-10-candle avg
      5. Current time < NO_NEW_TRADES_AFTER (15:00 IST)

    Parameters
    ----------
    stock_data : pd.DataFrame
        5-min OHLCV DataFrame for the stock.
    session : str
        "morning" or "afternoon".
    direction : str
        "LONG" or "SHORT".
    current_time : datetime, optional
        Override for current time (useful for backtesting). Defaults to now.

    Returns
    -------
    tuple (bool, str)
        (True, "All conditions met") if valid entry, else (False, reason).
    """
    if stock_data is None or len(stock_data) < 15:
        return False, "Insufficient data (need ≥ 15 candles)"

    direction = direction.upper()
    if current_time is None:
        current_time = datetime.now()

    # ── Condition 5: Time check ───────────────────────────────────────────────
    cutoff_h, cutoff_m = map(int, NO_NEW_TRADES_AFTER.split(":"))
    cutoff = current_time.replace(hour=cutoff_h, minute=cutoff_m,
                                   second=0, microsecond=0)
    if current_time >= cutoff:
        return False, f"Past no-new-trades cutoff ({NO_NEW_TRADES_AFTER})"

    close = stock_data["Close"].dropna().values
    high = stock_data["High"].dropna().values
    low = stock_data["Low"].dropna().values
    volume = stock_data["Volume"].dropna().values

    last_close = float(close[-1])

    # ── Condition 2: EMA(8) vs EMA(21) ───────────────────────────────────────
    ema8 = _ema(close, 8)
    ema21 = _ema(close, 21)
    if len(ema8) == 0 or len(ema21) == 0:
        return False, "Cannot compute EMAs"

    if direction == "LONG" and ema8[-1] <= ema21[-1]:
        return False, f"EMA(8) {ema8[-1]:.2f} not above EMA(21) {ema21[-1]:.2f}"
    if direction == "SHORT" and ema8[-1] >= ema21[-1]:
        return False, f"EMA(8) {ema8[-1]:.2f} not below EMA(21) {ema21[-1]:.2f}"

    # ── Condition 3: RSI(7) ────────────────────────────────────────────────────
    rsi = _compute_rsi(close, period=7)
    last_rsi = float(rsi[-1])

    if direction == "LONG":
        if not (RSI_LONG_MIN <= last_rsi <= RSI_LONG_MAX):
            return False, f"RSI(7) {last_rsi:.1f} not in LONG range [{RSI_LONG_MIN}, {RSI_LONG_MAX}]"
    else:
        if not (RSI_SHORT_MIN <= last_rsi <= RSI_SHORT_MAX):
            return False, f"RSI(7) {last_rsi:.1f} not in SHORT range [{RSI_SHORT_MIN}, {RSI_SHORT_MAX}]"

    # ── Condition 4: Volume spike ──────────────────────────────────────────────
    if len(volume) >= 11:
        avg_vol_10 = float(np.mean(volume[-11:-1]))
        current_vol = float(volume[-1])
        if avg_vol_10 > 0 and current_vol < VOLUME_SPIKE_FACTOR * avg_vol_10:
            return False, (
                f"Volume {current_vol:,.0f} < "
                f"{VOLUME_SPIKE_FACTOR}× avg {avg_vol_10:,.0f}"
            )

    # ── Condition 1: ORB / VWAP breakout ─────────────────────────────────────
    try:
        idx = pd.to_datetime(stock_data.index)
        if session == "morning":
            orb_mask = (idx.hour == 9) & (idx.minute <= 30)
            orb_df = stock_data[orb_mask]
            if orb_df.empty:
                orb_df = stock_data.iloc[:3]
            orb_high = float(orb_df["High"].max())
            orb_low = float(orb_df["Low"].min())

            if direction == "LONG" and last_close <= orb_high:
                return False, f"Price {last_close:.2f} not above ORB High {orb_high:.2f}"
            if direction == "SHORT" and last_close >= orb_low:
                return False, f"Price {last_close:.2f} not below ORB Low {orb_low:.2f}"
        else:
            # Afternoon: VWAP reclaim
            typical = (stock_data["High"] + stock_data["Low"] + stock_data["Close"]) / 3
            vol = stock_data["Volume"].replace(0, np.nan)
            vwap = (typical * vol).cumsum() / vol.cumsum()
            last_vwap = float(vwap.dropna().iloc[-1])

            if direction == "LONG" and last_close <= last_vwap:
                return False, f"Price {last_close:.2f} not above VWAP {last_vwap:.2f}"
            if direction == "SHORT" and last_close >= last_vwap:
                return False, f"Price {last_close:.2f} not below VWAP {last_vwap:.2f}"
    except Exception as e:
        logger.debug(f"ORB/VWAP check error: {e}")

    return True, "All conditions met"


def display_trade_card(trade_setup: dict) -> None:
    """
    Print a formatted ASCII trade card to the console.

    Parameters
    ----------
    trade_setup : dict
        Output of generate_trade_setup().
    """
    stock = str(trade_setup.get("stock", "?")).replace(".NS", "")
    direction = str(trade_setup.get("direction", "LONG"))
    session = str(trade_setup.get("session", "MORNING")).upper()
    entry = float(trade_setup.get("entry", 0))
    sl = float(trade_setup.get("sl", 0))
    tp1 = float(trade_setup.get("tp1", 0))
    tp2 = float(trade_setup.get("tp2", 0))
    qty = int(trade_setup.get("qty", 0))
    max_loss = float(trade_setup.get("max_loss", 0))
    rr = float(trade_setup.get("rr_ratio", 0))

    sl_pct = trade_setup.get("sl_pct", abs(sl - entry) / entry * 100 if entry else 0)
    tp1_pct = trade_setup.get("tp1_pct", abs(tp1 - entry) / entry * 100 if entry else 0)
    tp2_pct = trade_setup.get("tp2_pct", abs(tp2 - entry) / entry * 100 if entry else 0)

    sl_sign = "-" if direction == "LONG" else "+"
    tp_sign = "+" if direction == "LONG" else "-"

    def _row(label: str, value: str) -> str:
        """Build a padded card row of fixed inner width 44."""
        content = f"  {label:<16}: {value}"
        padding = 44 - len(content)
        return f"\u2551{content}{' ' * max(padding, 0)}\u2551"

    title_text = f"  TRADE SETUP -- {stock} [{direction}]"
    inner_w = 44
    title_padded = f"{title_text:<{inner_w}}"

    print(f"\n\u2554{'=' * inner_w}\u2557")
    print(f"\u2551{title_padded}\u2551")
    print(f"\u2560{'=' * inner_w}\u2563")
    print(_row("Entry Price",   f"Rs {entry:>10,.2f}"))
    print(_row("Stop Loss",     f"Rs {sl:>10,.2f}  ({sl_sign}{sl_pct:.2f}%)"))
    print(_row("Take Profit 1", f"Rs {tp1:>10,.2f}  ({tp_sign}{tp1_pct:.2f}%)"))
    print(_row("Take Profit 2", f"Rs {tp2:>10,.2f}  ({tp_sign}{tp2_pct:.2f}%)"))
    print(_row("Quantity",      f"{qty} shares"))
    print(_row("Max Loss",      f"Rs {max_loss:>10,.2f}"))
    print(_row("Risk-Reward",   f"1 : {rr:.2f}"))
    print(_row("Session",       session))
    print(f"\u255a{'=' * inner_w}\u255d\n")


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute the *period*-bar Average True Range from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (5-min bars).
    period : int
        ATR period. Default 14.

    Returns
    -------
    float
        Last ATR value in ₹.
    """
    if len(df) < period + 1:
        # Fallback: use simple high-low range
        hl_range = (df["High"] - df["Low"]).dropna()
        return float(hl_range.mean()) if not hl_range.empty else 0.0

    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    return float(np.mean(tr[-period:]))


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers (duplicated for module independence)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(prices.astype(float))
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(len(delta), np.nan)
    avg_loss = np.full(len(delta), np.nan)

    if len(delta) < period:
        return np.full(len(prices), 50.0)

    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])

    for i in range(period, len(delta)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    full_rsi = np.full(len(prices), 50.0)
    full_rsi[1:] = rsi
    return full_rsi


def _ema(data: np.ndarray, span: int) -> np.ndarray:
    if len(data) == 0:
        return np.array([])
    s = pd.Series(data.astype(float))
    return s.ewm(span=span, adjust=False).mean().values


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_setups = [
        ("RELIANCE.NS", 2450.00, 9.75, "LONG", "morning"),
        ("HDFCBANK.NS", 1680.00, 6.50, "LONG", "morning"),
        ("TCS.NS",      3890.00, 14.20, "LONG", "afternoon"),
        ("INFY.NS",     1590.00, 5.80, "SHORT", "morning"),
        ("SBIN.NS",     810.00, 3.20, "LONG", "afternoon"),
    ]

    print("Running Step 3 — Strategy Engine …\n")
    for sym, price, atr, direction, session in sample_setups:
        setup = generate_trade_setup(sym, price, atr, direction, session)
        display_trade_card(setup)
