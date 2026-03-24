"""
Step 2 — AI Scoring Engine
============================
Scores each stock in the filtered universe across 7 weighted parameters and
returns the Top 5 stocks for the Morning session and Top 5 for the Afternoon
session, each with a composite score and directional bias (LONG / SHORT).

Scoring parameters & weights:
  1. Volume Surge Ratio   (20%) — current 15-min vol vs 20-day avg 15-min vol
  2. ATR Volatility Score (15%) — 14-period ATR on 5-min chart
  3. Momentum Score       (20%) — RSI(7) + MACD crossover bonus
  4. VWAP Position Score  (15%) — price position relative to VWAP
  5. ORB Score            (15%) — Open Range Breakout (9:15–9:30)
  6. News Sentiment Score (10%) — keyword-based NLP on yfinance news
  7. Trend Score          ( 5%) — EMA(8) vs EMA(21) crossover
"""

import sys
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config.config import WEIGHTS, USE_ANGEL_ONE, LOG_FILE
except ModuleNotFoundError:
    WEIGHTS = {
        'volume_surge': 0.20,
        'atr_volatility': 0.15,
        'momentum': 0.20,
        'vwap_position': 0.15,
        'orb_score': 0.15,
        'sentiment': 0.10,
        'trend': 0.05,
    }
    USE_ANGEL_ONE = False
    LOG_FILE = "logs/trading.log"

try:
    from loguru import logger
    logger.add(LOG_FILE, rotation="1 day", retention="30 days", level="INFO")
except Exception:
    import logging
    logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta library not installed. Install with: pip install ta")

BULLISH_KEYWORDS = [
    "beat", "growth", "profit", "record", "surge", "gain", "upgrade",
    "buy", "outperform", "strong", "positive", "rally", "jump", "rise",
    "dividend", "expansion", "win", "award", "contract", "order",
]
BEARISH_KEYWORDS = [
    "miss", "loss", "decline", "cut", "downgrade", "sell", "underperform",
    "weak", "negative", "fall", "drop", "concern", "risk", "fraud",
    "penalty", "fine", "lawsuit", "layoff", "debt",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def calculate_all_scores(stock_list: list,
                         intraday_data: Optional[dict] = None,
                         session: str = "morning") -> pd.DataFrame:
    """
    Compute all 7 scoring parameters for each stock and return a DataFrame.

    Parameters
    ----------
    stock_list : list of str
        Filtered universe of ticker symbols (e.g. ["RELIANCE.NS", ...]).
    intraday_data : dict, optional
        Pre-fetched {symbol: OHLCV DataFrame} mapping.  If None, the function
        will attempt to fetch 5-min data via yfinance.
    session : str
        "morning" or "afternoon" — affects ORB and VWAP scoring windows.

    Returns
    -------
    pd.DataFrame
        One row per stock with columns for each score, composite_score,
        direction, and rank.
    """
    if intraday_data is None:
        intraday_data = _fetch_intraday(stock_list)

    rows = []
    for sym in stock_list:
        df = intraday_data.get(sym)
        if df is None or df.empty:
            logger.debug(f"{sym}: no intraday data — skipped in scoring.")
            continue

        row = _score_single_stock(sym, df, session)
        rows.append(row)

    if not rows:
        logger.warning("No stocks could be scored.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["composite_score"] = (
        result["volume_surge_score"] * WEIGHTS["volume_surge"]
        + result["atr_volatility_score"] * WEIGHTS["atr_volatility"]
        + result["momentum_score"] * WEIGHTS["momentum"]
        + result["vwap_position_score"] * WEIGHTS["vwap_position"]
        + result["orb_score"] * WEIGHTS["orb_score"]
        + result["sentiment_score"] * WEIGHTS["sentiment"]
        + result["trend_score"] * WEIGHTS["trend"]
    ).round(4)

    result["direction"] = result.apply(_determine_direction, axis=1)
    result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result


def get_top5(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the top 5 stocks by composite score.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of calculate_all_scores().

    Returns
    -------
    pd.DataFrame
        Top 5 rows with symbol, composite_score, direction, and all sub-scores.
    """
    if scored_df.empty:
        return scored_df
    return scored_df.head(5).copy()


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """
    Linearly normalise *value* to the [0, 1] range.

    Parameters
    ----------
    value : float
        Raw metric value.
    min_val : float
        The minimum of the expected range (maps to score 0).
    max_val : float
        The maximum of the expected range (maps to score 1).

    Returns
    -------
    float
        Clamped score in [0.0, 1.0].
    """
    if max_val == min_val:
        return 0.5
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def display_score_table(scored_df: pd.DataFrame, session: str = "morning") -> None:
    """
    Print a formatted score table for the given session's top picks.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of calculate_all_scores().
    session : str
        "morning" or "afternoon" — used only in the header label.
    """
    if scored_df.empty:
        print("No scored stocks to display.")
        return

    top5 = get_top5(scored_df)
    session_label = session.upper()

    print("\n" + "═" * 110)
    print(f"  AI SCORING ENGINE — TOP 5 PICKS ({session_label} SESSION)")
    print("═" * 110)
    header = (
        f"  {'#':>2}  {'Symbol':<15} {'Dir':>5}  "
        f"{'VolSurge':>8} {'ATR':>5} {'Mom':>5} {'VWAP':>5} "
        f"{'ORB':>5} {'News':>5} {'Trend':>5}  {'COMPOSITE':>9}"
    )
    print(header)
    print("  " + "─" * 106)

    for _, row in top5.iterrows():
        sym = str(row["symbol"]).replace(".NS", "")
        direction = str(row.get("direction", "─"))
        vs = f"{row['volume_surge_score']:.2f}"
        atr = f"{row['atr_volatility_score']:.2f}"
        mom = f"{row['momentum_score']:.2f}"
        vwap = f"{row['vwap_position_score']:.2f}"
        orb = f"{row['orb_score']:.2f}"
        news = f"{row['sentiment_score']:.2f}"
        trend = f"{row['trend_score']:.2f}"
        comp = f"{row['composite_score']:.4f}"
        rank = int(row["rank"])

        dir_icon = "▲" if direction == "LONG" else "▼"
        print(
            f"  {rank:>2}  {sym:<15} {dir_icon} {direction:<4}  "
            f"{vs:>8} {atr:>5} {mom:>5} {vwap:>5} "
            f"{orb:>5} {news:>5} {trend:>5}  {comp:>9}"
        )

    print("═" * 110 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-stock scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_single_stock(sym: str, df: pd.DataFrame, session: str) -> dict:
    """Compute all 7 sub-scores for a single stock."""
    row = {"symbol": sym}

    try:
        row["volume_surge_score"] = _volume_surge_score(df)
    except Exception as e:
        logger.debug(f"{sym} volume_surge error: {e}")
        row["volume_surge_score"] = 0.5

    try:
        row["atr_volatility_score"] = _atr_volatility_score(df)
    except Exception as e:
        logger.debug(f"{sym} atr_volatility error: {e}")
        row["atr_volatility_score"] = 0.5

    try:
        row["momentum_score"] = _momentum_score(df)
    except Exception as e:
        logger.debug(f"{sym} momentum error: {e}")
        row["momentum_score"] = 0.5

    try:
        row["vwap_position_score"] = _vwap_position_score(df)
    except Exception as e:
        logger.debug(f"{sym} vwap error: {e}")
        row["vwap_position_score"] = 0.5

    try:
        row["orb_score"] = _orb_score(df, session)
    except Exception as e:
        logger.debug(f"{sym} orb error: {e}")
        row["orb_score"] = 0.5

    try:
        row["sentiment_score"] = _news_sentiment_score(sym)
    except Exception as e:
        logger.debug(f"{sym} sentiment error: {e}")
        row["sentiment_score"] = 0.5

    try:
        row["trend_score"] = _trend_score(df)
    except Exception as e:
        logger.debug(f"{sym} trend error: {e}")
        row["trend_score"] = 0.5

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Individual scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def _volume_surge_score(df: pd.DataFrame) -> float:
    """
    Volume Surge Ratio (weight 20%).

    Compares the most recent 15-minute cumulative volume to the 20-day average
    of the same window.  Score is clamped to [0, 1].
    """
    vol = df["Volume"].dropna()
    if len(vol) < 3:
        return 0.5

    # Use the last 3 bars as a proxy for ~15 min volume on 5-min data
    recent_vol = float(vol.iloc[-3:].sum())
    avg_vol = float(vol.mean()) * 3  # scale average to same 15-min window
    if avg_vol == 0:
        return 0.5

    ratio = recent_vol / avg_vol
    # Ratio of 2× → score 1.0; ratio of 0 → score 0
    return normalize_score(ratio, min_val=0.5, max_val=3.0)


def _atr_volatility_score(df: pd.DataFrame) -> float:
    """
    ATR Volatility Score (weight 15%).

    Calculates 14-period ATR on the 5-min chart.  Score is high when
    ATR > 0.4% of price (good scalping volatility).
    """
    if len(df) < 15:
        return 0.5

    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    # True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    atr = float(np.mean(tr[-14:]))
    last_price = float(close[-1])

    if last_price == 0:
        return 0.5

    atr_pct = atr / last_price
    # Score peaks when ATR > 0.4%; above 1.5% is too volatile
    return normalize_score(atr_pct, min_val=0.001, max_val=0.015)


def _momentum_score(df: pd.DataFrame) -> float:
    """
    Momentum Score (weight 20%).

    RSI(7) on 5-min chart: score peaks at RSI 60-70 (bull) or 30-40 (bear).
    Adds a MACD crossover bonus of 0.1 when MACD line crosses signal line upward.
    Final score is clamped to [0, 1].
    """
    close = df["Close"].dropna()
    if len(close) < 14:
        return 0.5

    # RSI(7)
    rsi = _compute_rsi(close.values, period=7)
    last_rsi = float(rsi[-1])

    # Score RSI: best between 60-70 (bullish) and 30-40 (bearish)
    if 60 <= last_rsi <= 70:
        rsi_score = 1.0
    elif 52 <= last_rsi < 60:
        rsi_score = normalize_score(last_rsi, 52, 60)
    elif 70 < last_rsi <= 80:
        rsi_score = normalize_score(80 - last_rsi, 0, 10)
    elif 30 <= last_rsi <= 40:
        rsi_score = 0.8
    elif 25 <= last_rsi < 30:
        rsi_score = normalize_score(last_rsi, 25, 30)
    else:
        rsi_score = 0.3

    # MACD crossover bonus
    macd_bonus = 0.0
    if len(close) >= 35:
        ema12 = _ema(close.values, 12)
        ema26 = _ema(close.values, 26)
        macd_line = ema12 - ema26
        signal = _ema(macd_line, 9)
        if len(macd_line) >= 2 and len(signal) >= 2:
            if macd_line[-1] > signal[-1] and macd_line[-2] <= signal[-2]:
                macd_bonus = 0.15  # Bullish crossover
            elif macd_line[-1] < signal[-1] and macd_line[-2] >= signal[-2]:
                macd_bonus = 0.05  # Bearish crossover (partial credit)

    return float(np.clip(rsi_score + macd_bonus, 0.0, 1.0))


def _vwap_position_score(df: pd.DataFrame) -> float:
    """
    VWAP Position Score (weight 15%).

    Computes VWAP from the available intraday data.  Price above VWAP is
    bullish.  The further above VWAP, the higher the score (up to +2%).
    """
    if len(df) < 3:
        return 0.5

    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan)
    cumulative_tp_vol = (typical * vol).cumsum()
    cumulative_vol = vol.cumsum()
    vwap = cumulative_tp_vol / cumulative_vol

    last_close = float(df["Close"].dropna().iloc[-1])
    last_vwap = float(vwap.dropna().iloc[-1])

    if last_vwap == 0:
        return 0.5

    distance_pct = (last_close - last_vwap) / last_vwap
    # Score: +2% above VWAP → 1.0; -2% below → 0.0
    return normalize_score(distance_pct, min_val=-0.02, max_val=0.02)


def _orb_score(df: pd.DataFrame, session: str) -> float:
    """
    Open Range Breakout Score (weight 15%).

    Checks if the current price is near or breaking the ORB High/Low defined
    by the first 15 minutes of trading (9:15–9:30 for morning session).

    For afternoon session, recalculates an afternoon ORB (13:30–13:45).
    """
    if df.empty or len(df) < 4:
        return 0.5

    # Identify ORB window by time-of-day index (if datetime index available)
    try:
        idx = pd.to_datetime(df.index)
        if session == "morning":
            orb_mask = (idx.hour == 9) & (idx.minute <= 30)
        else:
            orb_mask = (idx.hour == 13) & (idx.minute <= 45)

        orb_df = df[orb_mask]
        if orb_df.empty:
            # Fallback: use first 3 bars as ORB
            orb_df = df.iloc[:3]
    except Exception:
        orb_df = df.iloc[:3]

    orb_high = float(orb_df["High"].max())
    orb_low = float(orb_df["Low"].min())
    last_close = float(df["Close"].dropna().iloc[-1])

    if orb_high == orb_low:
        return 0.5

    orb_range = orb_high - orb_low

    if last_close > orb_high:
        # Breakout above ORB
        excess = (last_close - orb_high) / orb_range
        return float(np.clip(0.8 + excess * 0.2, 0.0, 1.0))
    elif last_close < orb_low:
        # Breakdown below ORB
        deficit = (orb_low - last_close) / orb_range
        return float(np.clip(0.2 - deficit * 0.2, 0.0, 1.0))
    else:
        # Inside ORB — neutral, bias toward top
        position = (last_close - orb_low) / orb_range
        return normalize_score(position, 0.0, 1.0) * 0.6 + 0.2


def _news_sentiment_score(sym: str) -> float:
    """
    News Sentiment Score (weight 10%).

    Fetches recent news headlines from yfinance and applies a simple
    keyword-based NLP classifier.  Returns a score in [0, 1].
    0 = very bearish, 0.5 = neutral, 1 = very bullish.
    """
    if not YFINANCE_AVAILABLE:
        return 0.5

    try:
        ticker = yf.Ticker(sym)
        news = ticker.news or []
        if not news:
            return 0.5

        bull_count = 0
        bear_count = 0

        for item in news[:10]:  # Check up to 10 recent headlines
            headline = ""
            # Handle both old and new yfinance news formats
            if isinstance(item, dict):
                headline = (
                    item.get("title", "")
                    + " "
                    + item.get("summary", "")
                ).lower()
            else:
                headline = str(item).lower()

            for word in BULLISH_KEYWORDS:
                if word in headline:
                    bull_count += 1
            for word in BEARISH_KEYWORDS:
                if word in headline:
                    bear_count += 1

        total = bull_count + bear_count
        if total == 0:
            return 0.5

        return normalize_score(bull_count, 0, total)

    except Exception as e:
        logger.debug(f"News fetch failed for {sym}: {e}")
        return 0.5


def _trend_score(df: pd.DataFrame) -> float:
    """
    Trend Score (weight 5%).

    Checks EMA(8) vs EMA(21) on the 5-min chart.
    Bullish crossover (EMA8 > EMA21) → score approaching 1.0.
    Bearish crossover → score approaching 0.0.
    """
    close = df["Close"].dropna().values
    if len(close) < 22:
        return 0.5

    ema8 = _ema(close, 8)
    ema21 = _ema(close, 21)

    if len(ema8) == 0 or len(ema21) == 0:
        return 0.5

    last_ema8 = float(ema8[-1])
    last_ema21 = float(ema21[-1])

    if last_ema21 == 0:
        return 0.5

    spread_pct = (last_ema8 - last_ema21) / last_ema21
    # Normalise: +1% spread → 1.0; -1% spread → 0.0
    return normalize_score(spread_pct, min_val=-0.01, max_val=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Wilder RSI for a 1-D price array."""
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
    """Compute Exponential Moving Average using pandas EWM under the hood."""
    if len(data) == 0:
        return np.array([])
    s = pd.Series(data.astype(float))
    return s.ewm(span=span, adjust=False).mean().values


def _determine_direction(row: pd.Series) -> str:
    """
    Determine trade direction (LONG / SHORT) from the composite sub-scores.

    A stock is classified as LONG if the majority of directional signals are
    bullish (score > 0.5), otherwise SHORT.
    """
    directional_cols = [
        "momentum_score", "vwap_position_score",
        "orb_score", "trend_score",
    ]
    scores = [row.get(col, 0.5) for col in directional_cols]
    bullish_count = sum(1 for s in scores if s > 0.5)
    return "LONG" if bullish_count >= 2 else "SHORT"


def _fetch_intraday(symbols: list) -> dict:
    """Fetch 5-min intraday data for *symbols* using yfinance."""
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not available.")
        return {}

    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="5d", interval="5m")
            if not df.empty:
                data[sym] = df
        except Exception as e:
            logger.debug(f"{sym}: intraday fetch failed — {e}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sample_stocks = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
        "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
    ]

    print("Running Step 2 — AI Scoring Engine …\n")
    scored = calculate_all_scores(sample_stocks, session="morning")
    if not scored.empty:
        display_score_table(scored, session="morning")
        top5 = get_top5(scored)
        print("Top 5 stocks:")
        print(top5[["rank", "symbol", "direction", "composite_score"]])
    else:
        print("No scores computed (check data availability).")
