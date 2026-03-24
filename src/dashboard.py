"""
Streamlit Dashboard — AI Scalping System P&L Tracker
=====================================================
Reads the paper trade CSV history and displays:
  • Daily P&L bar chart (green/red)
  • Cumulative capital curve
  • Key performance metrics (Win Rate, Profit Factor, Sharpe, Max Drawdown)
  • Today's open trades table
  • All-time trade log with filters

Run with:
    streamlit run src/dashboard.py
"""

import sys
import os
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── Default paths ──────────────────────────────────────────────────────────────
DEFAULT_TRADE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_trades.csv"
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_trade_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess trade CSV data."""
    if not Path(filepath).exists():
        return pd.DataFrame()
    df = pd.read_csv(filepath, parse_dates=["date", "entry_time", "exit_time"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0)
    df["gross_pnl"] = pd.to_numeric(df["gross_pnl"], errors="coerce").fillna(0)
    df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0)
    df["capital_after"] = pd.to_numeric(df["capital_after"], errors="coerce")
    df["result"] = df["net_pnl"].apply(lambda x: "WIN" if x >= 0 else "LOSS")
    return df


def compute_metrics(df: pd.DataFrame, initial_capital: float = 100_000) -> dict:
    """Compute key performance metrics from trade history."""
    if df.empty:
        return {}

    total_trades = len(df)
    wins = df[df["net_pnl"] >= 0]
    losses = df[df["net_pnl"] < 0]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0

    gross_profit = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Daily P&L
    daily = df.groupby("date")["net_pnl"].sum().reset_index()
    daily["cumulative"] = daily["net_pnl"].cumsum() + initial_capital

    # Sharpe Ratio (annualised, assuming 250 trading days)
    if len(daily) > 1:
        daily_returns = daily["net_pnl"] / initial_capital
        sharpe = (daily_returns.mean() / daily_returns.std()) * (250 ** 0.5) if daily_returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = daily["cumulative"].cummax()
    drawdown = (daily["cumulative"] - peak) / peak * 100
    max_drawdown = float(drawdown.min())

    return {
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "net_pnl": round(df["net_pnl"].sum(), 2),
        "total_fees": round(df["fees"].sum(), 2),
        "best_trade": round(df["net_pnl"].max(), 2),
        "worst_trade": round(df["net_pnl"].min(), 2),
        "daily": daily,
        "current_capital": round(
            df["capital_after"].dropna().iloc[-1] if not df["capital_after"].dropna().empty
            else initial_capital + df["net_pnl"].sum(), 2
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def run_dashboard():
    """Main Streamlit dashboard entry point."""
    st.set_page_config(
        page_title="AI Scalping System — P&L Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 AI Intraday Scalping System — P&L Dashboard")
    st.markdown("*NSE/BSE Paper Trading Tracker · Angel One SmartAPI*")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    csv_path = st.sidebar.text_input(
        "Trade CSV Path",
        value=DEFAULT_TRADE_CSV,
        help="Path to the paper trade CSV file.",
    )
    initial_capital = st.sidebar.number_input(
        "Initial Capital (₹)", value=100_000, step=10_000
    )
    session_filter = st.sidebar.multiselect(
        "Session Filter",
        options=["morning", "afternoon"],
        default=["morning", "afternoon"],
    )

    # ── Load Data ──────────────────────────────────────────────────────────────
    df = load_trade_data(csv_path)

    if df.empty:
        st.warning(
            f"⚠️ No trade data found at `{csv_path}`. "
            "Run the paper trader to generate trades, or check the file path."
        )
        st.info(
            "**Quick Start:**  \n"
            "1. Run `python src/step4_paper_trader.py` to simulate trades.  \n"
            "2. Trades will be saved to `data/trades.csv`.  \n"
            "3. Update the CSV path in the sidebar."
        )
        return

    if session_filter:
        df = df[df["session"].isin(session_filter)]

    metrics = compute_metrics(df, initial_capital)

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Key Performance Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Trades", metrics.get("total_trades", 0))
    col2.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
    col3.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    col4.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    col5.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
    net = metrics.get("net_pnl", 0)
    col6.metric("Net P&L", f"₹{net:,.2f}", delta=f"₹{net:+,.2f}")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📅 Daily P&L")
        if PLOTLY_AVAILABLE and "daily" in metrics:
            daily = metrics["daily"]
            colors = ["#00c853" if v >= 0 else "#d32f2f" for v in daily["net_pnl"]]
            fig = go.Figure(go.Bar(
                x=daily["date"].astype(str),
                y=daily["net_pnl"],
                marker_color=colors,
                text=[f"₹{v:+,.0f}" for v in daily["net_pnl"]],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Net P&L (₹)",
                showlegend=False,
                height=350,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Install plotly for charts: `pip install plotly`")

    with col_right:
        st.markdown("#### 📈 Cumulative Capital Curve")
        if PLOTLY_AVAILABLE and "daily" in metrics:
            daily = metrics["daily"]
            fig2 = go.Figure(go.Scatter(
                x=daily["date"].astype(str),
                y=daily["cumulative"],
                mode="lines+markers",
                line=dict(color="#1976d2", width=2),
                fill="tozeroy",
                fillcolor="rgba(25, 118, 210, 0.1)",
            ))
            fig2.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital",
            )
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="Capital (₹)",
                height=350,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Today's Summary ────────────────────────────────────────────────────────
    today = date.today()
    today_df = df[df["date"] == today]
    st.markdown(f"#### 🗓️ Today's Trades ({today})")
    if today_df.empty:
        st.info("No trades recorded for today.")
    else:
        display_cols = [
            "session", "stock", "direction", "entry_price", "exit_price",
            "qty", "gross_pnl", "fees", "net_pnl", "exit_reason", "result",
        ]
        st.dataframe(
            today_df[[c for c in display_cols if c in today_df.columns]]
            .rename(columns={
                "session": "Session", "stock": "Stock",
                "direction": "Dir", "entry_price": "Entry",
                "exit_price": "Exit", "qty": "Qty",
                "gross_pnl": "Gross P&L", "fees": "Fees",
                "net_pnl": "Net P&L", "exit_reason": "Reason",
                "result": "Result",
            }),
            use_container_width=True,
        )

    st.markdown("---")

    # ── All-Time Trade Log ─────────────────────────────────────────────────────
    st.markdown("#### 📋 All-Time Trade Log")

    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    all_stocks = sorted(df["stock"].str.replace(".NS", "", regex=False).unique().tolist())
    sel_stocks = fcol1.multiselect("Stock Filter", options=all_stocks, default=[])
    sel_result = fcol2.multiselect("Result Filter", options=["WIN", "LOSS"], default=[])
    sel_dir = fcol3.multiselect("Direction", options=["LONG", "SHORT"], default=[])

    log_df = df.copy()
    if sel_stocks:
        log_df = log_df[log_df["stock"].str.replace(".NS", "", regex=False).isin(sel_stocks)]
    if sel_result:
        log_df = log_df[log_df["result"].isin(sel_result)]
    if sel_dir:
        log_df = log_df[log_df["direction"].isin(sel_dir)]

    st.dataframe(
        log_df[[c for c in [
            "date", "session", "stock", "direction",
            "entry_price", "exit_price", "qty",
            "gross_pnl", "fees", "net_pnl",
            "exit_reason", "result", "rr_ratio", "composite_score",
        ] if c in log_df.columns]].sort_values("date", ascending=False),
        use_container_width=True,
    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "⚠️ *This is a paper trading simulation only. Not financial advice. "
        "Past paper performance does not guarantee future live results.*"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_dashboard()
    else:
        print("Streamlit not installed. Run: pip install streamlit")
        print("Then: streamlit run src/dashboard.py")
