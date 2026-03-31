"""
Step 4 — Paper Trade Simulator & P&L Tracker
=============================================
Simulates intraday scalping trades with full Angel One fee calculation,
trade lifecycle management (OPEN → CLOSED), and daily P&L reporting.

Fee structure (Angel One intraday):
  Brokerage      : ₹20 per leg (flat)
  STT            : 0.025% on sell side
  Transaction    : 0.00345% of turnover (NSE)
  SEBI charges   : ₹10 per crore of turnover
  Stamp duty     : 0.003% on buy side
  GST            : 18% on (brokerage + transaction)

Trade states:
  OPEN   → Active trade
  CLOSED → Fully closed (TP / SL / EOD square-off)
"""

import sys
import os
import csv
import math
import warnings
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── IST timezone helper ────────────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))


def _today_ist() -> date:
    """Return the current date in IST (UTC+5:30)."""
    return datetime.now(tz=_IST).date()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from config.config import (
        CAPITAL,
        CAPITAL_PER_STOCK,
        MAX_DAILY_LOSS,
        BROKERAGE_PER_LEG,
        STT_RATE,
        TRANSACTION_CHARGE_RATE,
        SEBI_CHARGE_RATE,
        STAMP_DUTY_RATE,
        GST_RATE,
        LOG_FILE,
    )
except ModuleNotFoundError:
    CAPITAL = 100_000
    CAPITAL_PER_STOCK = 20_000
    MAX_DAILY_LOSS = 2_000
    BROKERAGE_PER_LEG = 20
    STT_RATE = 0.00025
    TRANSACTION_CHARGE_RATE = 0.0000345
    SEBI_CHARGE_RATE = 10 / 10_000_000
    STAMP_DUTY_RATE = 0.00003
    GST_RATE = 0.18
    LOG_FILE = "logs/trading.log"

try:
    from loguru import logger
    logger.add(LOG_FILE, rotation="1 day", retention="30 days", level="INFO")
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ── CSV column order ───────────────────────────────────────────────────────────
TRADE_CSV_COLUMNS = [
    "date", "session", "stock", "symbol_token", "direction",
    "entry_price", "exit_price", "qty",
    "sl", "tp", "exit_reason",
    "gross_pnl", "fees", "net_pnl",
    "entry_time", "exit_time",
    "composite_score", "atr", "rr_ratio", "capital_after",
]

# ── Trade states ───────────────────────────────────────────────────────────────
STATE_OPEN = "OPEN"
STATE_CLOSED = "CLOSED"


# ─────────────────────────────────────────────────────────────────────────────
# Fee Calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_charges(buy_price: float, sell_price: float, qty: int) -> float:
    """
    Calculate total transaction charges for an intraday trade on Angel One.

    Parameters
    ----------
    buy_price : float
        Price at which shares were bought.
    sell_price : float
        Price at which shares were sold.
    qty : int
        Number of shares.

    Returns
    -------
    float
        Total charges in ₹ (rounded to 2 decimal places).
    """
    turnover = (buy_price + sell_price) * qty
    brokerage = BROKERAGE_PER_LEG + BROKERAGE_PER_LEG          # ₹20 per leg
    stt = STT_RATE * sell_price * qty                           # sell side only
    transaction_charges = TRANSACTION_CHARGE_RATE * turnover
    sebi_charges = SEBI_CHARGE_RATE * turnover
    stamp_duty = STAMP_DUTY_RATE * buy_price * qty              # buy side only
    gst = GST_RATE * (brokerage + transaction_charges)
    total = (brokerage + stt + transaction_charges
             + sebi_charges + stamp_duty + gst)
    return round(total, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PaperTrader Class
# ─────────────────────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Simulates intraday paper trades with realistic fee calculation and
    end-of-day square-off. The full position exits at the single TP level.

    Parameters
    ----------
    capital : float
        Starting capital in ₹. Defaults to config.CAPITAL (₹1,00,000).
    """

    def __init__(self, capital: float = CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.open_trades: dict = {}    # stock → trade dict
        self.closed_trades: list = []  # list of completed trade dicts
        self._daily_loss = 0.0

    # ── Trade Lifecycle ───────────────────────────────────────────────────────

    def open_trade(self,
                   stock: str,
                   direction: str,
                   entry: float,
                   sl: float,
                   tp: float,
                   qty: int,
                   session: str = "morning",
                   composite_score: float = 0.0,
                   atr: float = 0.0,
                   rr_ratio: float = 0.0,
                   symbol_token: str = "") -> bool:
        """
        Open a new paper trade position.

        Parameters
        ----------
        stock : str
            Ticker symbol.
        direction : str
            "LONG" or "SHORT".
        entry : float
            Entry price in ₹.
        sl : float
            Stop-loss price in ₹.
        tp : float
            Take-Profit price in ₹ (full position exits here).
        qty : int
            Total number of shares to trade.
        session : str
            "morning" or "afternoon".
        composite_score : float
            AI composite score (for logging).
        atr : float
            ATR used in setup.
        rr_ratio : float
            Risk-Reward ratio.
        symbol_token : str
            Angel One symbol token (optional).

        Returns
        -------
        bool
            True if trade opened successfully.
        """
        if stock in self.open_trades:
            logger.warning(f"Trade for {stock} already open. Skipping.")
            return False

        if self._daily_loss >= MAX_DAILY_LOSS:
            logger.warning(
                f"Daily loss limit ₹{MAX_DAILY_LOSS:,} reached. "
                "No new trades for today."
            )
            return False

        trade = {
            "stock": stock,
            "symbol_token": symbol_token,
            "direction": direction.upper(),
            "entry": entry,
            "sl": sl,
            "original_sl": sl,
            "tp": tp,
            "qty": qty,
            "session": session,
            "state": STATE_OPEN,
            "entry_time": datetime.now(),
            "date": _today_ist(),
            "composite_score": composite_score,
            "atr": atr,
            "rr_ratio": rr_ratio,
        }

        self.open_trades[stock] = trade
        logger.info(
            f"[OPEN] {stock} {direction} @ ₹{entry:.2f}  "
            f"SL={sl:.2f}  TP={tp:.2f}  Qty={qty}"
        )
        return True

    def update_prices(self, stock: str, current_price: float) -> Optional[str]:
        """
        Update the price for an open trade and trigger TP / SL if hit.

        Call this once per 1-min or 5-min candle close.

        Parameters
        ----------
        stock : str
            Ticker symbol.
        current_price : float
            Latest market price.

        Returns
        -------
        str or None
            "TP" or "SL" if a level was hit, else None.
        """
        if stock not in self.open_trades:
            return None

        trade = self.open_trades[stock]
        direction = trade["direction"]

        # Check TP (full exit)
        if _hit_target(direction, current_price, trade["tp"], "LONG_TP"):
            return self.close_trade(stock, current_price, "TP")
        # Check SL
        if _hit_stop(direction, current_price, trade["sl"]):
            return self.close_trade(stock, current_price, "SL")

        return None

    def close_trade(self, stock: str, exit_price: float,
                    reason: str = "EOD") -> Optional[str]:
        """
        Fully close an open trade and record the result.

        Parameters
        ----------
        stock : str
            Ticker symbol.
        exit_price : float
            Exit price in ₹.
        reason : str
            Reason code: "TP", "SL", "EOD", "MANUAL".

        Returns
        -------
        str
            The reason code if trade was closed, else None.
        """
        if stock not in self.open_trades:
            logger.warning(f"No open trade found for {stock}.")
            return None

        trade = self.open_trades.pop(stock)
        direction = trade["direction"]
        entry = trade["entry"]
        qty = trade["qty"]

        # ── Gross P&L: full qty exits at exit_price ───────────────────────────
        if direction == "LONG":
            gross = (exit_price - entry) * qty
        else:
            gross = (entry - exit_price) * qty
        total_gross = round(gross, 2)

        # ── Fees ──────────────────────────────────────────────────────────────
        if direction == "LONG":
            buy_p, sell_p = entry, exit_price
        else:
            buy_p, sell_p = exit_price, entry
        total_fees = calculate_charges(buy_p, sell_p, qty)

        net_pnl = round(total_gross - total_fees, 2)

        self.capital = round(self.capital + net_pnl, 2)
        if net_pnl < 0:
            self._daily_loss += abs(net_pnl)

        # ── Build closed trade record ─────────────────────────────────────────
        closed = {
            "date": str(trade["date"]),
            "session": trade["session"],
            "stock": stock,
            "symbol_token": trade["symbol_token"],
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "qty": qty,
            "sl": trade["original_sl"],
            "tp": trade["tp"],
            "exit_reason": reason,
            "gross_pnl": total_gross,
            "fees": total_fees,
            "net_pnl": net_pnl,
            "entry_time": str(trade["entry_time"]),
            "exit_time": str(datetime.now()),
            "composite_score": trade["composite_score"],
            "atr": trade["atr"],
            "rr_ratio": trade["rr_ratio"],
            "capital_after": self.capital,
        }
        self.closed_trades.append(closed)

        result_icon = "✅" if net_pnl >= 0 else "❌"
        logger.info(
            f"[CLOSED] {stock} {direction}  Exit={exit_price:.2f}  "
            f"Reason={reason}  Gross=₹{total_gross:+.2f}  "
            f"Fees=₹{total_fees:.2f}  Net=₹{net_pnl:+.2f} {result_icon}"
        )
        return reason

    def eod_squareoff(self) -> None:
        """
        Square off all remaining open trades at the last known price.
        Called at 15:15 IST (market close).
        """
        if not self.open_trades:
            return

        stocks = list(self.open_trades.keys())
        for stock in stocks:
            trade = self.open_trades[stock]
            # Use TP as proxy for last price if no live price available
            last_price = trade.get("last_price", trade["tp"])
            self.close_trade(stock, last_price, "EOD")
            logger.info(f"[EOD] {stock} squared off at ₹{last_price:.2f}")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_daily_summary(self, trade_date: Optional[str] = None) -> dict:
        """
        Compute and return the daily P&L summary.

        Parameters
        ----------
        trade_date : str, optional
            Filter trades for this date (YYYY-MM-DD). Defaults to today.

        Returns
        -------
        dict
            Summary with trade count, wins, losses, gross/fees/net P&L,
            capital, win rate, and individual trade rows.
        """
        if trade_date is None:
            trade_date = str(_today_ist())

        day_trades = [
            t for t in self.closed_trades if t["date"] == trade_date
        ]

        if not day_trades:
            return {
                "date": trade_date,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "gross_pnl": 0.0,
                "total_fees": 0.0,
                "net_pnl": 0.0,
                "capital": self.capital,
                "daily_return_pct": 0.0,
                "trades": [],
            }

        wins = [t for t in day_trades if t["net_pnl"] >= 0]
        losses = [t for t in day_trades if t["net_pnl"] < 0]
        gross = sum(t["gross_pnl"] for t in day_trades)
        fees = sum(t["fees"] for t in day_trades)
        net = sum(t["net_pnl"] for t in day_trades)
        daily_return = (net / self.initial_capital) * 100

        return {
            "date": trade_date,
            "total_trades": len(day_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(day_trades) * 100 if day_trades else 0.0,
            "gross_pnl": round(gross, 2),
            "total_fees": round(fees, 2),
            "net_pnl": round(net, 2),
            "capital": self.capital,
            "daily_return_pct": round(daily_return, 3),
            "trades": day_trades,
        }

    def print_daily_report(self, trade_date: Optional[str] = None) -> None:
        """
        Print a formatted daily P&L report to the console.

        Parameters
        ----------
        trade_date : str, optional
            Date to report (YYYY-MM-DD). Defaults to today.
        """
        summary = self.get_daily_summary(trade_date)
        date_label = summary["date"]

        print("\n" + "═" * 95)
        print(f"  📅 PAPER TRADE REPORT — {date_label}")
        print("═" * 95)

        if not summary["trades"]:
            print("  No trades recorded for this date.")
            print("═" * 95 + "\n")
            return

        # Header
        print(
            f"  {'SESSION':<10} │ {'STOCK':<10} │ {'DIR':<5} │ "
            f"{'ENTRY':>7} │ {'EXIT':>7} │ {'QTY':>4} │ "
            f"{'GROSS':>8} │ {'FEES':>6} │ {'NET P&L':>8} │ RESULT"
        )
        print("  " + "─" * 91)

        for t in summary["trades"]:
            sym = str(t["stock"]).replace(".NS", "")
            direction = t["direction"]
            gross = t["gross_pnl"]
            net = t["net_pnl"]
            result = "✅ WIN" if net >= 0 else "❌ LOSS"
            print(
                f"  {t['session'].upper():<10} │ {sym:<10} │ {direction:<5} │ "
                f"₹{t['entry_price']:>6.1f} │ ₹{t['exit_price']:>6.1f} │ "
                f"{t['qty']:>4} │ "
                f"{'+₹' if gross >= 0 else '-₹'}{abs(gross):>6.0f} │ "
                f"-₹{t['fees']:>5.0f} │ "
                f"{'+₹' if net >= 0 else '-₹'}{abs(net):>7.0f} │ {result}"
            )

        print("  " + "─" * 91)
        print(
            f"  TOTAL TRADES: {summary['total_trades']}  │  "
            f"WINS: {summary['wins']}  │  "
            f"LOSSES: {summary['losses']}  │  "
            f"WIN RATE: {summary['win_rate']:.1f}%"
        )
        net_sign = "+" if summary["net_pnl"] >= 0 else ""
        gross_sign = "+" if summary["gross_pnl"] >= 0 else ""
        print(
            f"  GROSS P&L: {gross_sign}₹{summary['gross_pnl']:.2f}  │  "
            f"TOTAL FEES: -₹{summary['total_fees']:.2f}  │  "
            f"NET P&L: {net_sign}₹{summary['net_pnl']:.2f}"
        )
        ret_sign = "+" if summary["daily_return_pct"] >= 0 else ""
        print(
            f"  CAPITAL: ₹{summary['capital']:,.2f}  │  "
            f"DAILY RETURN: {ret_sign}{summary['daily_return_pct']:.3f}%"
        )
        print("═" * 95 + "\n")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_csv(self, filepath: str) -> None:
        """
        Append today's new closed trades to the CSV, avoiding duplicates.
        Existing rows are preserved; only trades not already present are written.

        Parameters
        ----------
        filepath : str
            Destination file path (e.g. "data/trades.csv").
        """
        if not self.closed_trades:
            logger.info("No closed trades to save.")
            return

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(self.closed_trades)[TRADE_CSV_COLUMNS]

        if Path(filepath).exists():
            existing_df = pd.read_csv(filepath)
            # Normalise date strings to prevent type mismatch duplicates
            existing_df["date"] = pd.to_datetime(
                existing_df["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            new_df["date"] = pd.to_datetime(
                new_df["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            # Deduplicate on natural key: date + stock + entry_time
            key_cols = ["date", "stock", "entry_time"]
            merged = new_df.merge(
                existing_df[key_cols], on=key_cols, how="left", indicator=True
            )
            truly_new = new_df[merged["_merge"] == "left_only"]
            if truly_new.empty:
                logger.info("No new trades to append (all already saved).")
                return
            combined = pd.concat([existing_df, truly_new], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(filepath, index=False)
        logger.info(
            f"Saved {len(combined)} total trades to {filepath} "
            f"({len(new_df)} new today)"
        )

    def load_history(self, filepath: str) -> None:
        """
        Load trade history from a CSV file into closed_trades.

        Parameters
        ----------
        filepath : str
            Path to the CSV file (must match TRADE_CSV_COLUMNS schema).
        """
        if not Path(filepath).exists():
            logger.warning(f"Trade history file not found: {filepath}")
            return

        df = pd.read_csv(filepath)
        # Migrate old tp1/tp2 columns to single tp column, with safety fallback
        if "tp" not in df.columns or df["tp"].isna().all():
            if "tp2" in df.columns:
                df["tp"] = df["tp2"]
            elif "tp1" in df.columns:
                df["tp"] = df["tp1"]
            else:
                df["tp"] = df.get("entry_price", 0)
        # Coerce tp to numeric to prevent string comparison issues
        df["tp"] = pd.to_numeric(df["tp"], errors="coerce").fillna(0)
        # Bug 1: normalise date column to plain YYYY-MM-DD strings
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        # Align columns
        for col in TRADE_CSV_COLUMNS:
            if col not in df.columns:
                df[col] = None

        self.closed_trades = df[TRADE_CSV_COLUMNS].to_dict("records")
        logger.info(f"Loaded {len(self.closed_trades)} trades from {filepath}")

        # Update capital to last recorded value
        if self.closed_trades:
            last_capital = self.closed_trades[-1].get("capital_after")
            if last_capital and not pd.isna(last_capital):
                self.capital = float(last_capital)

    # ── Internal helpers ──────────────────────────────────────────────────────
    # (reserved for future use)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hit_target(direction: str, price: float,
                target: float, side: str) -> bool:
    if direction == "LONG":
        return price >= target
    else:
        return price <= target


def _hit_stop(direction: str, price: float, stop: float) -> bool:
    if direction == "LONG":
        return price <= stop
    else:
        return price >= stop


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point / quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Step 4 — Paper Trade Simulator …\n")

    trader = PaperTrader(capital=100_000)

    # Simulate 3 trades
    trader.open_trade("RELIANCE.NS", "LONG",  2450.0, 2435.25, 2459.50, 8,  "morning")
    trader.open_trade("HDFCBANK.NS", "LONG",  1680.0, 1666.60, 1686.50, 11, "morning")
    trader.open_trade("TCS.NS",      "LONG",  3890.0, 3867.50, 3904.00, 5,  "afternoon")

    # Simulate price movements
    trader.update_prices("RELIANCE.NS", 2460.00)   # TP hit
    trader.update_prices("HDFCBANK.NS", 1664.00)   # SL hit
    trader.update_prices("TCS.NS",      3904.00)   # TP hit

    trader.print_daily_report()

    print(f"Fee example: buy=₹2450 sell=₹2478 qty=8 → ₹{calculate_charges(2450, 2478, 8)}")
