"""
Trading Configuration for AI Scalping System
NSE/BSE Indian Stock Markets — Angel One SmartAPI + yfinance
"""

# ── Capital Management ────────────────────────────────────────────────────────
CAPITAL = 100_000                            # ₹1,00,000 starting capital
MAX_POSITIONS = 5                            # Maximum simultaneous open trades
CAPITAL_PER_STOCK = CAPITAL / MAX_POSITIONS  # ₹20,000 per stock
MAX_RISK_PER_TRADE = 0.01 * CAPITAL          # 1% → ₹1,000
MAX_DAILY_LOSS = 0.02 * CAPITAL              # 2% → ₹2,000 daily circuit breaker

# ── Session Timings (IST) ─────────────────────────────────────────────────────
MORNING_START = "09:15"
MORNING_END = "10:45"
AFTERNOON_START = "13:30"
AFTERNOON_END = "15:15"
EOD_SQUAREOFF = "15:15"
NO_NEW_TRADES_AFTER = "15:00"

# ── ATR Multipliers for SL / TP ───────────────────────────────────────────────
SL_ATR_MULTIPLIER = 2.0   # wider SL to survive intraday noise
TP_ATR_MULTIPLIER = 1.0   # single TP — full position exits here
MIN_SL_PCT = 0.004         # Minimum SL distance: 0.4% from entry
MAX_SL_PCT = 0.010         # Maximum SL distance: 1.0% from entry

MIN_COMPOSITE_SCORE = 0.60

# ── Angel One Brokerage & Statutory Charges ───────────────────────────────────
BROKERAGE_PER_LEG = 20                       # ₹20 flat per leg (intraday)
STT_RATE = 0.00025                           # 0.025% on sell side
TRANSACTION_CHARGE_RATE = 0.0000345          # NSE transaction charge
SEBI_CHARGE_RATE = 10 / 10_000_000          # ₹10 per crore of turnover
STAMP_DUTY_RATE = 0.00003                    # 0.003% on buy side
GST_RATE = 0.18                              # 18% GST on brokerage + transaction

# ── Scoring Weights (must sum to 1.0) ────────────────────────────────────────
WEIGHTS = {
    'volume_surge': 0.20,
    'atr_volatility': 0.15,
    'momentum': 0.20,
    'vwap_position': 0.15,
    'orb_score': 0.15,
    'sentiment': 0.10,
    'trend': 0.05,
}

# ── Data Source ───────────────────────────────────────────────────────────────
USE_ANGEL_ONE = False   # Set True to switch from yfinance to Angel One live data

# ── Pre-market Filter Thresholds ─────────────────────────────────────────────
MIN_PRICE = 100                  # Minimum stock price (₹)
MIN_AVG_VOLUME = 500_000         # Minimum 20-day average daily volume
CIRCUIT_LIMIT_PCT = 0.09         # Exclude stocks with >9% move from prev close

# ── RSI Bounds for Entry ──────────────────────────────────────────────────────
RSI_LONG_MIN = 52
RSI_LONG_MAX = 75
RSI_SHORT_MIN = 25
RSI_SHORT_MAX = 48

# ── Volume Confirmation ───────────────────────────────────────────────────────
VOLUME_SPIKE_FACTOR = 1.5   # Current candle volume must be >1.5× last-10 avg

# ── Stock Universe (NIFTY 50) ─────────────────────────────────────────────────
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "TECHM.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "BAJAJFINSV.NS", "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "M&M.NS", "SBILIFE.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "LTIM.NS",
]

# ── Stock Universe (NIFTY NEXT 50) ────────────────────────────────────────────
NIFTY_NEXT50_SYMBOLS = [
    "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BANDHANBNK.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS",
    "FEDERALBNK.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "INDHOTEL.NS", "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS",
    "LUPIN.NS", "MCDOWELL-N.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "NMDC.NS", "OFSS.NS", "PAGEIND.NS", "PEL.NS", "PIDILITIND.NS",
    "PIIND.NS", "PNB.NS", "RECLTD.NS", "SAIL.NS", "SIEMENS.NS",
    "SRF.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS",
    "UNIONBANK.NS", "UNITDSPR.NS", "UPL.NS", "VEDL.NS", "VOLTAS.NS",
    "WHIRLPOOL.NS", "ZYDUSLIFE.NS",
]

ALL_SYMBOLS = NIFTY50_SYMBOLS + NIFTY_NEXT50_SYMBOLS

# ── F&O Ban List (updated daily — hardcoded as fallback) ──────────────────────
FNO_BAN_LIST = [
    # Add NSE F&O ban list symbols here (updated daily)
    # e.g. "HINDCOPPER.NS", "IDEA.NS"
]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "logs/trading.log"
LOG_ROTATION = "1 day"
LOG_RETENTION = "30 days"
