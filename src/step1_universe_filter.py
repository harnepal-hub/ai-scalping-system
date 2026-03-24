"""
Step 1 — Stock Universe Filter
================================
Maintains a base universe of ~150 NSE stocks (NIFTY 50 + NIFTY NEXT 50 +
top liquid mid-caps) and applies pre-market filter rules to produce a
shortlist of ~80-100 tradable stocks for the scoring engine.

Pre-market filter rules (run at 9:10 AM and 1:15 PM):
  • Remove F&O ban list stocks
  • Remove stocks with price < ₹100
  • Remove stocks with avg daily volume < 5,00,000 shares
  • Remove stocks that have hit upper/lower circuit (>9% move from prev close)

Data source: yfinance (primary / Colab mode)
            Angel One SmartAPI (secondary / live mode — requires USE_ANGEL_ONE=True)
"""

import sys
import os
import math
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Allow running as a standalone script or as part of the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config.config import (
        ALL_SYMBOLS,
        NIFTY50_SYMBOLS,
        NIFTY_NEXT50_SYMBOLS,
        FNO_BAN_LIST,
        MIN_PRICE,
        MIN_AVG_VOLUME,
        CIRCUIT_LIMIT_PCT,
        USE_ANGEL_ONE,
        LOG_FILE,
    )
except ModuleNotFoundError:
    # Fallback defaults when running outside the package
    ALL_SYMBOLS = []
    NIFTY50_SYMBOLS = []
    NIFTY_NEXT50_SYMBOLS = []
    FNO_BAN_LIST = []
    MIN_PRICE = 100
    MIN_AVG_VOLUME = 500_000
    CIRCUIT_LIMIT_PCT = 0.09
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
    logger.warning("yfinance not installed. Install with: pip install yfinance")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_universe() -> list:
    """
    Return the full base universe of tradable NSE symbols.

    Returns
    -------
    list of str
        Yahoo Finance ticker symbols ending in '.NS'.
    """
    return list(ALL_SYMBOLS)


def get_premarket_data(symbols: Optional[list] = None,
                       period: str = "5d",
                       interval: str = "5m") -> dict:
    """
    Fetch OHLCV data for the given symbols using yfinance.

    Parameters
    ----------
    symbols : list of str, optional
        List of ticker symbols (e.g. ["RELIANCE.NS", "TCS.NS"]).
        Defaults to the full universe from get_universe().
    period : str
        yfinance period string, e.g. "5d", "1mo".
    interval : str
        yfinance interval string, e.g. "5m", "1m", "1d".

    Returns
    -------
    dict
        Mapping symbol → pd.DataFrame (OHLCV columns).
        Symbols that fail to download are omitted.
    """
    if symbols is None:
        symbols = get_universe()

    data = {}

    if not YFINANCE_AVAILABLE:
        logger.error("yfinance is not available. Cannot fetch pre-market data.")
        return data

    logger.info(f"Fetching {interval} data for {len(symbols)} symbols via yfinance …")

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.debug(f"{sym}: empty data returned, skipping.")
                continue
            df.index = pd.to_datetime(df.index)
            data[sym] = df
        except Exception as exc:
            logger.warning(f"{sym}: download failed — {exc}")

    logger.info(f"Successfully fetched data for {len(data)}/{len(symbols)} symbols.")
    return data


def apply_premarket_filters(market_data: dict,
                             fno_ban: Optional[list] = None) -> pd.DataFrame:
    """
    Apply pre-market filter rules and return a DataFrame of tradable stocks.

    Filter rules applied in order:
      1. Remove F&O ban list stocks
      2. Remove stocks with last price < MIN_PRICE (₹100)
      3. Remove stocks with 20-day average volume < MIN_AVG_VOLUME (5,00,000)
      4. Remove stocks that moved > CIRCUIT_LIMIT_PCT (9%) from previous close

    Parameters
    ----------
    market_data : dict
        Output of get_premarket_data(); symbol → OHLCV DataFrame.
    fno_ban : list of str, optional
        Additional F&O ban symbols to exclude. Merged with config list.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, last_price, avg_volume, pct_change, passes_filter, reason
    """
    if fno_ban is None:
        fno_ban = []
    ban_set = set(FNO_BAN_LIST) | set(fno_ban)

    rows = []

    for sym, df in market_data.items():
        reason = "OK"
        passes = True

        # ── Rule 1: F&O ban ──────────────────────────────────────────────────
        if sym in ban_set:
            reason = "F&O ban"
            passes = False
            rows.append(_make_row(sym, df, passes, reason))
            continue

        # ── Compute summary stats ─────────────────────────────────────────────
        last_price = _get_last_price(df)
        avg_vol = _get_avg_daily_volume(df)
        pct_chg = _get_pct_change_from_prev_close(df)

        # ── Rule 2: Minimum price ─────────────────────────────────────────────
        if last_price is not None and last_price < MIN_PRICE:
            reason = f"Price ₹{last_price:.2f} < ₹{MIN_PRICE}"
            passes = False

        # ── Rule 3: Minimum average volume ───────────────────────────────────
        elif avg_vol is not None and avg_vol < MIN_AVG_VOLUME:
            reason = f"Avg vol {avg_vol:,.0f} < {MIN_AVG_VOLUME:,}"
            passes = False

        # ── Rule 4: Circuit breaker ───────────────────────────────────────────
        elif pct_chg is not None and abs(pct_chg) > CIRCUIT_LIMIT_PCT:
            direction = "upper" if pct_chg > 0 else "lower"
            reason = f"Circuit ({direction}): {pct_chg*100:.1f}%"
            passes = False

        rows.append({
            "symbol": sym,
            "last_price": round(last_price, 2) if last_price is not None else None,
            "avg_volume": round(avg_vol, 0) if avg_vol is not None else None,
            "pct_change": round(pct_chg * 100, 2) if pct_chg is not None else None,
            "passes_filter": passes,
            "reason": reason,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    passed = result[result["passes_filter"]].copy()
    logger.info(
        f"Pre-market filter: {len(passed)}/{len(market_data)} stocks passed "
        f"({len(market_data) - len(passed)} removed)."
    )
    return result


def get_filtered_universe(symbols: Optional[list] = None,
                           fno_ban: Optional[list] = None,
                           verbose: bool = True) -> list:
    """
    High-level convenience function that runs the complete pre-market pipeline.

    Fetches daily data (period="20d", interval="1d") for volume averaging,
    applies all filters, and returns the list of tradable symbols.

    Parameters
    ----------
    symbols : list of str, optional
        Symbols to consider. Defaults to full universe.
    fno_ban : list of str, optional
        Additional F&O ban symbols.
    verbose : bool
        Print summary table to stdout if True.

    Returns
    -------
    list of str
        Filtered symbols ready for the scoring engine.
    """
    if symbols is None:
        symbols = get_universe()

    market_data = get_premarket_data(symbols, period="20d", interval="1d")
    filter_df = apply_premarket_filters(market_data, fno_ban=fno_ban)

    if filter_df.empty:
        logger.warning("No data available — returning full universe as fallback.")
        return symbols

    passed_df = filter_df[filter_df["passes_filter"]]

    if verbose:
        _print_filter_summary(filter_df)

    return passed_df["symbol"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_last_price(df: pd.DataFrame) -> Optional[float]:
    """Return the most recent closing price from a daily or intraday OHLCV df."""
    try:
        return float(df["Close"].dropna().iloc[-1])
    except (IndexError, KeyError):
        return None


def _get_avg_daily_volume(df: pd.DataFrame) -> Optional[float]:
    """Return the average daily volume over the available history."""
    try:
        return float(df["Volume"].dropna().mean())
    except (IndexError, KeyError):
        return None


def _get_pct_change_from_prev_close(df: pd.DataFrame) -> Optional[float]:
    """
    Return the percentage change of the last close vs the second-to-last close.
    Used to detect circuit-limit breaches.
    """
    try:
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        prev = float(closes.iloc[-2])
        last = float(closes.iloc[-1])
        if prev == 0:
            return None
        return (last - prev) / prev
    except (IndexError, KeyError):
        return None


def _make_row(sym: str, df: pd.DataFrame,
              passes: bool, reason: str) -> dict:
    """Build a result row dict for the filter DataFrame."""
    return {
        "symbol": sym,
        "last_price": _get_last_price(df),
        "avg_volume": _get_avg_daily_volume(df),
        "pct_change": None,
        "passes_filter": passes,
        "reason": reason,
    }


def _print_filter_summary(filter_df: pd.DataFrame) -> None:
    """Print a formatted pre-market filter summary table."""
    total = len(filter_df)
    passed = filter_df["passes_filter"].sum()
    removed = total - passed

    print("\n" + "═" * 72)
    print("  PRE-MARKET FILTER SUMMARY")
    print("═" * 72)
    print(f"  Total universe   : {total:>4} stocks")
    print(f"  Passed filter    : {passed:>4} stocks  ✅")
    print(f"  Removed          : {removed:>4} stocks  ❌")
    print("─" * 72)

    passed_df = filter_df[filter_df["passes_filter"]].copy()
    if not passed_df.empty:
        print("\n  TRADABLE STOCKS:")
        print(f"  {'Symbol':<20} {'Price (₹)':>10} {'Avg Vol':>12} {'Δ%':>7}")
        print("  " + "─" * 55)
        for _, row in passed_df.iterrows():
            sym = row["symbol"].replace(".NS", "")
            price = f"₹{row['last_price']:,.2f}" if row["last_price"] else "N/A"
            vol = f"{row['avg_volume']:>12,.0f}" if row["avg_volume"] else "N/A"
            pct = f"{row['pct_change']:>+.2f}%" if row["pct_change"] is not None else "N/A"
            print(f"  {sym:<20} {price:>10} {vol:>12} {pct:>7}")

    removed_df = filter_df[~filter_df["passes_filter"]]
    if not removed_df.empty:
        print(f"\n  REMOVED STOCKS ({len(removed_df)}):")
        for _, row in removed_df.iterrows():
            sym = row["symbol"].replace(".NS", "")
            print(f"    ✗ {sym:<18} — {row['reason']}")

    print("═" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Step 1 — Stock Universe Filter …\n")
    tradable = get_filtered_universe(verbose=True)
    print(f"\nFinal tradable list ({len(tradable)} stocks):")
    print(tradable)
