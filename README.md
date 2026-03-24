# 🤖 AI Intraday Scalping System — NSE/BSE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harnepal-hub/ai-scalping-system/blob/main/notebooks/AI_Scalping_System_Colab.ipynb)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-paper%20trading-orange)

> ⚠️ **PAPER TRADING ONLY** — This system is for educational and research purposes. It does **not** execute real trades. Always consult a SEBI-registered advisor before investing.

---

## 📋 Project Overview

An AI-powered intraday scalping framework for NSE/BSE Indian stock markets that:

- **Identifies** the top 5 momentum stocks for Morning (9:15–10:45) and Afternoon (13:30–15:15) sessions
- **Scores** each stock across 7 weighted parameters using technical and sentiment analysis
- **Generates** precise trade setups (Entry, TP, Stop-Loss) using ATR-based calculations
- **Simulates** paper trades with full Angel One fee structure, lifecycle management, and daily P&L reporting
- **Visualises** performance via a Streamlit dashboard

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────┐
                    │   AI SCALPING SYSTEM — DATA FLOW    │
                    └─────────────────────────────────────┘

  yfinance / Angel One API
         │
         ▼
  ┌─────────────────────┐
  │  Step 1: Universe   │  ← ~150 NSE stocks (NIFTY 50 + NIFTY NEXT 50)
  │  Filter             │  ← Price, Volume, Circuit breaker, F&O ban filters
  └──────────┬──────────┘
             │ ~80-100 tradable stocks
             ▼
  ┌─────────────────────┐
  │  Step 2: AI Scoring │  ← 7 weighted parameters
  │  Engine             │  ← Morning Top 5 + Afternoon Top 5
  └──────────┬──────────┘
             │ Top 5 stocks + direction (LONG/SHORT)
             ▼
  ┌─────────────────────┐
  │  Step 3: Strategy   │  ← ATR-based Entry / SL / TP
  │  Engine             │  ← Trade card generation
  └──────────┬──────────┘
             │ Trade setups
             ▼
  ┌─────────────────────┐
  │  Step 4: Paper      │  ← Full trade lifecycle simulation
  │  Trader             │  ← Fee calculation, P&L tracking, CSV logging
  └──────────┬──────────┘
             │ Trade history CSV
             ▼
  ┌─────────────────────┐
  │  Dashboard          │  ← Streamlit P&L dashboard
  │  (Streamlit)        │  ← Charts, metrics, trade log
  └─────────────────────┘
```

---

## 🚀 Quick Start (Google Colab)

1. Click the **Open in Colab** badge above
2. Run all cells in order (Runtime → Run All)
3. For Angel One live mode, add your API credentials to Colab secrets

**Local setup:**

```bash
git clone https://github.com/harnepal-hub/ai-scalping-system.git
cd ai-scalping-system
pip install -r requirements.txt
python src/step4_paper_trader.py        # smoke test
streamlit run src/dashboard.py          # launch dashboard
```

---

## 📁 Project Structure

```
ai-scalping-system/
├── README.md
├── requirements.txt
├── config/
│   └── config.py                       ← All trading constants & universe lists
├── notebooks/
│   └── AI_Scalping_System_Colab.ipynb  ← Main Colab notebook (all-in-one)
├── src/
│   ├── __init__.py
│   ├── step1_universe_filter.py        ← Stock universe + pre-market scan
│   ├── step2_scoring_engine.py         ← AI scoring engine (7 parameters)
│   ├── step3_strategy_engine.py        ← Scalping strategy (Entry/TP/SL)
│   ├── step4_paper_trader.py           ← Paper trade simulator + fee calc
│   └── dashboard.py                    ← Streamlit P&L dashboard
├── data/
│   └── sample_trades.csv               ← 10 sample paper trades
└── logs/
    └── .gitkeep
```

---

## 🔧 Configuration

All constants live in `config/config.py`:

| Setting | Default | Description |
|---|---|---|
| `CAPITAL` | ₹1,00,000 | Starting capital |
| `MAX_POSITIONS` | 5 | Max simultaneous trades |
| `CAPITAL_PER_STOCK` | ₹20,000 | Capital allocated per stock |
| `MAX_RISK_PER_TRADE` | ₹1,000 | 1% risk per trade |
| `MAX_DAILY_LOSS` | ₹2,000 | Daily circuit breaker |
| `USE_ANGEL_ONE` | `False` | Switch to Angel One live data |

**Angel One API Keys** (never commit to git):

```bash
# Colab: add via Colab Secrets (left sidebar → 🔑 icon)
ANGEL_ONE_API_KEY=your_key
ANGEL_ONE_CLIENT_ID=your_client_id
ANGEL_ONE_PASSWORD=your_password
ANGEL_ONE_TOTP_SECRET=your_totp_secret

# Local: create a .env file
cp .env.example .env
# Edit .env with your credentials
```

---

## 📊 Scoring Parameters

| # | Parameter | Weight | Logic |
|---|---|---|---|
| 1 | Volume Surge Ratio | 20% | 15-min vol vs 20-day avg 15-min vol |
| 2 | ATR Volatility Score | 15% | 14-period ATR on 5-min chart; score > 0.4% of price |
| 3 | Momentum Score | 20% | RSI(7): peaks at 60-70 (bull) / 30-40 (bear) + MACD bonus |
| 4 | VWAP Position Score | 15% | Distance above/below VWAP |
| 5 | ORB Score | 15% | Open Range Breakout (9:15–9:30 first 15 min) |
| 6 | News Sentiment | 10% | Keyword-based NLP on yfinance news headlines |
| 7 | Trend Score | 5% | EMA(8) vs EMA(21) crossover on 5-min |

---

## 💰 Fee Structure (Angel One Intraday)

| Charge | Rate | Applied On |
|---|---|---|
| Brokerage | ₹20 per leg | Both buy & sell |
| STT | 0.025% | Sell side only |
| NSE Transaction | 0.00345% | Total turnover |
| SEBI Charges | ₹10/crore | Total turnover |
| Stamp Duty | 0.003% | Buy side only |
| GST | 18% | Brokerage + Transaction |

---

## 🛡️ Risk Management

| Rule | Setting |
|---|---|
| Capital per trade | ₹20,000 (20% of total) |
| Max risk per trade | ₹1,000 (1% of capital) |
| Daily loss limit | ₹2,000 (2% of capital) |
| SL distance | 0.4%–1.0% from entry (ATR-clamped) |
| TP exit | Full position at 1× ATR |
| No new trades | After 15:00 IST |
| EOD square-off | 15:15 IST |

---

## 📈 Strategy Summary

### Entry Conditions (ALL must be true for LONG)
1. Price breaks above ORB High (morning) or reclaims VWAP (afternoon)
2. EMA(8) > EMA(21) on 5-min chart
3. RSI(7) between 52 and 75
4. Current candle volume > 1.5× average of last 10 candles
5. Entry time < 15:00 IST

### Trade Lifecycle
```
OPEN → price hits TP → CLOSED (WIN)
     → price hits SL → CLOSED (LOSS)
     → 15:15 IST     → EOD SQUARE-OFF
```

---

## 🗺️ Roadmap

- [x] Step 1 — Stock Universe Filter
- [x] Step 2 — AI Scoring Engine (7 parameters)
- [x] Step 3 — Scalping Strategy Engine
- [x] Step 4 — Paper Trade Simulator + P&L Dashboard
- [ ] Step 5 — ML Model (Random Forest / XGBoost) trained on paper trade history
- [ ] Step 6 — Angel One Live Execution (after 3-month paper validation)
- [ ] Step 7 — Telegram / WhatsApp trade alerts
- [ ] Step 8 — Multi-account support & position sizing optimizer

---

## ⚙️ Module Reference

### `src/step1_universe_filter.py`
- `get_universe()` → full list of ~150 NSE symbols
- `get_premarket_data(symbols, period, interval)` → OHLCV data dict
- `apply_premarket_filters(market_data)` → filter result DataFrame
- `get_filtered_universe(symbols, fno_ban, verbose)` → tradable symbol list

### `src/step2_scoring_engine.py`
- `calculate_all_scores(stock_list, intraday_data, session)` → scored DataFrame
- `get_top5(scored_df)` → top 5 rows
- `normalize_score(value, min_val, max_val)` → [0, 1] normalisation
- `display_score_table(scored_df, session)` → formatted console table

### `src/step3_strategy_engine.py`
- `generate_trade_setup(stock, price, atr, direction, session)` → trade dict
- `check_entry_signal(stock_data, session, direction, current_time)` → (bool, reason)
- `display_trade_card(trade_setup)` → ASCII trade card
- `calculate_atr(df, period)` → ATR value in ₹

### `src/step4_paper_trader.py`
- `calculate_charges(buy_price, sell_price, qty)` → total fees in ₹
- `PaperTrader.open_trade(...)` → opens a position
- `PaperTrader.update_prices(stock, price)` → triggers TP/SL if hit
- `PaperTrader.close_trade(stock, exit_price, reason)` → closes position
- `PaperTrader.eod_squareoff()` → closes all open trades at 15:15
- `PaperTrader.get_daily_summary(date)` → daily P&L dict
- `PaperTrader.print_daily_report(date)` → formatted console report
- `PaperTrader.save_to_csv(filepath)` → persist trade history
- `PaperTrader.load_history(filepath)` → load past trades

### `src/dashboard.py`
- Streamlit app — run with `streamlit run src/dashboard.py`

---

## 📜 Disclaimer

This project is for **educational and research purposes only**.

- All trades simulated here are **paper trades** (no real money involved)
- Past paper trading performance is **not indicative** of future live results
- The authors are **not SEBI-registered advisors**
- **Never** invest money you cannot afford to lose
- Always do your own due diligence before trading real capital