#@title Agentic Mk1 — TimesFM Edition v2 (enriched 32-day signal)

import json, os, time, gc, warnings, logging
import pandas as pd
import numpy as np
try:
    import timesfm
    _TIMESFM_AVAILABLE = True
except ModuleNotFoundError:
    _TIMESFM_AVAILABLE = False
    print("WARNING: timesfm not installed — forecast signal disabled")
from datetime import datetime
from pydantic import BaseModel
from typing import Literal, List, Dict, Optional
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# API imports
from fastapi import FastAPI, Request

os.environ["HF_HOME"]               = "./.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./.cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
load_dotenv()

# =========================================================
# COMPETITION SCHEMAS
# =========================================================

class PricePoint(BaseModel):
    date: str
    price: float

class TradingRequest(BaseModel):
    date: str
    price: Dict[str, float]
    news: Dict[str, List[str]]
    symbol: List[str]
    momentum: Dict[str, str]
    history_price: Dict[str, List[PricePoint]]
    ten_k: Optional[Dict[str, List[str]]] = None
    ten_q: Optional[Dict[str, List[str]]] = None

class CompetitionResponse(BaseModel):
    recommended_action: Literal["BUY", "HOLD", "SELL"]

class MarketAnalysis(BaseModel):
    date: str
    news_bias: Literal["bullish", "bearish", "mixed"]
    macro_view: Literal["bull_market", "bear_market"]
    volatility: Literal["low", "normal", "high"]
    momentum_signal: Literal["bullish", "bearish", "mixed"]
    summary: str

class RiskAssessment(BaseModel):
    position_state: Literal["flat", "long", "short"]
    risk_level: Literal["low", "medium", "high"]
    preferred_action: Literal["buy", "sell", "hold"]
    conviction: Literal["strong", "moderate", "weak"]
    risk_note: str

class TradeDecision(BaseModel):
    date: str
    action: Literal["buy", "sell", "hold"]
    conviction: Literal["strong", "moderate", "weak"]
    reason: str


# =========================================================
# CONFIG
# =========================================================
WINDOW_SIZE        = 50
LAST_TRADES_WINDOW = 10
DATA_FILE          = "/home/work/fin/mulmodel/btc_data.jsonl"
HISTORY_FILE       = "trade_history_multi_model_edited_full_timesfm_v2.jsonl"
TIMESFM_FILE       = "timesfm_forecasts_v2.jsonl"
ASSET_SYMBOL       = "BTC"
INITIAL_BUDGET_USD = 200_000.0
MIN_CONVICTION_BUY = {"strong", "moderate"}   # weak → hold
MIN_HOLD_BARS      = 3
MIN_BUY_USD        = 1_000.0
SLEEP              = 0
FEE_RATE           = 0.001
MEMORY_WINDOW      = 10   # How many past daily records the model sees per request
TA_MIN_ROWS        = 50   # Rows needed for reliable SMA-50 / RSI-14 / MACD signals
MAX_EXPOSURE_PCT   = 0.60
TAKE_PROFIT_PCT    = 0.15
MIN_STOP_LOSS_PCT  = 0.07
CASH_RESERVE_PCT   = 0.25

# Live state for API mode — keyed by symbol so multi-asset requests don't bleed into each other.
# balance: +1.0 = long, 0.0 = flat, -1.0 = short (competition SELL = short)
_live_states: Dict[str, dict] = {}

# Rolling daily history per symbol — accumulates one record per request.
# Each record is a compact dict the agents read as "memory" of past decisions.
_symbol_history: Dict[str, List[dict]] = {}

def _get_state(symbol: str) -> dict:
    if symbol not in _live_states:
        _live_states[symbol] = {
            "balance":        0.0,   # +1 long / 0 flat / -1 short
            "avg_buy_price":  0.0,
            "peak_price":     0.0,
            "bars_since_buy": 9999,
            # Accumulated price rows: list of {"date": str, "prices": float}
            # Grows day by day so TA signals improve as the server runs longer.
            "price_buffer":   [],
        }
    return _live_states[symbol]

def _merge_prices(state: dict, incoming_rows: List[dict]) -> pd.DataFrame:
    """
    Merge the server's accumulated price_buffer with the rows from the request.
    - incoming_rows: list of {"date": str, "prices": float} (already renamed column)
    - Deduplicates by date, keeps the request's value when there's a conflict.
    - Appends any new date from the request into price_buffer for future calls.
    - Returns a sorted DataFrame ready for compute_ta_signals().
    """
    buf = state["price_buffer"]

    # Build a date → price dict from the accumulated buffer
    buf_map: Dict[str, float] = {r["date"]: r["prices"] for r in buf}

    # Overlay with incoming rows (request is authoritative for its dates)
    for row in incoming_rows:
        buf_map[row["date"]] = row["prices"]

    # Persist updated buffer back into state (sorted, capped at 500 rows)
    sorted_rows = sorted(buf_map.items())          # list of (date, price) tuples
    state["price_buffer"] = [{"date": d, "prices": p} for d, p in sorted_rows]
    if len(state["price_buffer"]) > 500:
        state["price_buffer"] = state["price_buffer"][-500:]

    # Return as DataFrame
    df = pd.DataFrame(state["price_buffer"])
    return df.reset_index(drop=True)

def _get_history(symbol: str) -> List[dict]:
    if symbol not in _symbol_history:
        _symbol_history[symbol] = []
    return _symbol_history[symbol]

def _append_history(symbol: str, record: dict) -> None:
    """Append today's record and cap the buffer at MEMORY_WINDOW entries."""
    hist = _get_history(symbol)
    hist.append(record)
    if len(hist) > MEMORY_WINDOW:
        hist.pop(0)

# ── Single model via HF Inference API — no GPU needed ──
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN", "")   # set in Render env vars

# =========================================================
# LOGGER INITIALIZATION
# =========================================================
class TradingLogger:
    def __init__(self, history_file: str):
        self.log_file = history_file.replace(".jsonl", ".log")
        self.logger   = logging.getLogger("trading")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info(f"=== Session started | log -> {self.log_file} ===")

    def day(self, date, price, rsi, regime, rs, pnl_pct, allowed):
        self.logger.debug(
            f"DAY  {date} | price=${price:>10,.0f} | RSI={rsi:5.1f} "
            f"| {regime.upper():<4} | {rs:<30} | pnl={pnl_pct:+6.1f}% "
            f"| allowed={allowed}"
        )

    def trade(self, date, action, conviction, price, portfolio_value,
              pnl_pct, reason, executed):
        if not executed:
            self.logger.debug(f"HOLD {date} | portfolio=${portfolio_value:,.2f}")
            return
        tag = "BUY " if action == "buy" else "SELL"
        self.logger.info(
            f"{tag} {date} | price=${price:>10,.0f} | conviction={conviction:<8} "
            f"| pnl={pnl_pct:+6.1f}% | portfolio=${portfolio_value:,.2f} | {reason}"
        )

    def warn(self, msg): self.logger.warning(msg)
    def err(self,  msg): self.logger.error(msg)
    def info(self, msg): self.logger.info(msg)

# Initialize logger early so API/functions can safely use it globally
log = TradingLogger(HISTORY_FILE)

# =========================================================
# GLOBAL MODEL — HF Inference API (no GPU, no local weights)
# =========================================================
class ModelManager:
    def __init__(self):
        self.client = None
        # Keep these names so the rest of the code is unchanged
        self.gen_market   = "market"
        self.gen_risk     = "risk"
        self.gen_decision = "decision"

    def load(self):
        print(f"\n--- Connecting to HF Inference API: {MODEL_ID} ---")
        self.client = InferenceClient(
            model=MODEL_ID,
            token=HF_TOKEN,
        )
        print("    HF Inference API ready — no GPU needed")

print("=== Initializing AI Models ===")
manager = ModelManager()
manager.load()

print("=== Loading TimesFM model ===")
tfm = None
if _TIMESFM_AVAILABLE:
    try:
        tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        # .compile() was added in newer TimesFM releases; skip gracefully if absent
        if hasattr(tfm, "compile"):
            tfm.compile(
                timesfm.ForecastConfig(
                    max_context=512,
                    max_horizon=128,
                    normalize_inputs=True,
                )
            )
        print("=== TimesFM Loaded ===")
    except Exception as e:
        print(f"WARNING: TimesFM failed to load ({e}) — forecast signal disabled")
else:
    print("WARNING: TimesFM skipped (not installed)")
print("=== Models Ready ===")

# =========================================================
# FASTAPI INSTANCE & ENDPOINT
# =========================================================
app = FastAPI(title="Agentic Mk1 - Trading Bot")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "timesfm": tfm is not None}

@app.post("/trading_action/", response_model=CompetitionResponse)
async def get_trading_decision(body: TradingRequest, request: Request):
    try:
        # Re-parse raw JSON to recover digit-prefixed keys "10k" / "10q" that
        # Pydantic cannot map to Python field names automatically.
        raw = await request.json()
        symbol = body.symbol[0]
        current_price = float(body.price[symbol])

        # 1. Per-symbol live state (fix: was a single global dict shared across symbols)
        state = _get_state(symbol)

        # 2. Parse competition signals — "10k"/"10q" come in as raw keys
        momentum = body.momentum.get(symbol, "neutral")
        ten_k_raw = raw.get("10k") or raw.get("ten_k")
        ten_q_raw = raw.get("10q") or raw.get("ten_q")
        ten_k = (ten_k_raw or {}).get(symbol) if isinstance(ten_k_raw, dict) else None
        ten_q = (ten_q_raw or {}).get(symbol) if isinstance(ten_q_raw, dict) else None
        news = list(body.news.get(symbol, []))

        filings_context = ""
        if ten_k:
            filings_context += f"10-K Filing: {ten_k}\n"
        if ten_q:
            filings_context += f"10-Q Filing: {ten_q}\n"
        news.append(f"Momentum: {momentum}")
        if filings_context:
            news.append(filings_context)

        # 3. Convert history to DataFrame and merge with accumulated price buffer.
        #    The competition sends ~10 rows per request — far too few for SMA-50 / MACD.
        #    _merge_prices() grows a per-symbol buffer across calls so TA quality
        #    improves every day the server runs, reaching full accuracy after ~50 days.
        raw_points = body.history_price.get(symbol, [])
        if len(raw_points) < 1:
            log.warn(f"[{symbol}] history_price is empty; defaulting to HOLD")
            return {"recommended_action": "HOLD"}

        # Convert incoming PricePoint objects → {"date", "prices"} dicts
        incoming_rows = [
            {"date": p.date, "prices": float(p.price)}
            for p in raw_points
        ]
        history_df = _merge_prices(state, incoming_rows)
        buf_len = len(history_df)

        if buf_len < 10:
            log.warn(f"[{symbol}] Only {buf_len} total price rows accumulated — defaulting to HOLD")
            return {"recommended_action": "HOLD"}
        if buf_len < TA_MIN_ROWS:
            log.warn(f"[{symbol}] {buf_len}/{TA_MIN_ROWS} rows — TA signals partially reliable; NaNs will be substituted")

        # 4. Run TA on the full merged history
        ta = compute_ta_signals(history_df.tail(WINDOW_SIZE).reset_index(drop=True))

        # Replace any NaN TA values with safe defaults so they don't corrupt the LLM prompt
        for key, default in [("rsi_14", 50.0), ("volatility_14d", 3.0),
                              ("momentum_10d", 0.0), ("bb_pct", 0.5),
                              ("dist_sma10_pct", 0.0), ("dist_sma20_pct", 0.0),
                              ("dist_sma50_pct", 0.0)]:
            if key in ta and (ta[key] != ta[key]):   # NaN check
                ta[key] = default
                log.warn(f"[{symbol}] TA field '{key}' was NaN — replaced with {default}")

        close_arr = history_df["prices"].astype(float).to_numpy()[-WINDOW_SIZE:]
        timesfm_str, _ = run_timesfm_forecast(close_arr, current_price)

        # 5. Position metrics — use balance as signal: +1 long, 0 flat, -1 short
        #    calc_position_metrics treats balance <= 0 as no position, which is
        #    correct for both flat and short (no trailing-stop logic needed when short).
        if state["balance"] > 0:
            state["peak_price"] = max(state["peak_price"], current_price)
        else:
            state["peak_price"] = 0.0

        pos = calc_position_metrics(
            state["balance"],
            state["avg_buy_price"],
            current_price,
            state["peak_price"],
            ta["volatility_14d"],
        )
        rs = regime_signal(ta, pos)
        allowed = ["buy", "sell", "hold"]

        # 6. Determine position label for risk agent
        if state["balance"] > 0:
            position_label = "long"
        elif state["balance"] < 0:
            position_label = "short"
        else:
            position_label = "flat"

        portfolio_state = {
            "symbol":         symbol,
            "balance":        state["balance"],   # +1 / 0 / -1
            "position_state": position_label,
            "avg_buy_price":  state["avg_buy_price"],
        }

        # 7. Load rolling daily memory for this symbol
        past_records = _get_history(symbol)   # list of last N daily dicts, oldest first

        # 8. Run the 3-Agent Chain — pass past_records as last_trades so the
        #    risk agent sees what was decided on previous days
        mkt  = analyze_market(ta, rs, pos, news, body.date, timesfm_str)
        risk = assess_risk(ta, rs, pos, portfolio_state, past_records, allowed, state["bars_since_buy"])
        decision = decide_action(mkt, risk, ta, rs, pos, allowed, state["bars_since_buy"], body.date, past_records)

        if decision is None:
            # fallback_decision expects Pydantic objects or None — pass None safely
            fb = fallback_decision(ta, rs, pos, allowed, mkt, risk, body.date)
            action = fb["action"].upper()
            reason = fb["reason"]
        else:
            action = decision.action.upper()
            reason = decision.reason

        # 9. Update per-symbol live state
        #    BUY  → long (+1);  SELL → short (-1);  HOLD → unchanged
        if action == "BUY":
            if state["balance"] <= 0:   # was flat or short → go long
                state["balance"]        = 1.0
                state["avg_buy_price"]  = current_price
                state["peak_price"]     = current_price
                state["bars_since_buy"] = 0
            else:
                state["bars_since_buy"] += 1
        elif action == "SELL":
            # Competition SELL = short position (fix: was incorrectly set to 0 / flat)
            state["balance"]        = -1.0
            state["avg_buy_price"]  = 0.0
            state["bars_since_buy"] = 9999
        else:  # HOLD
            if state["balance"] != 0:
                state["bars_since_buy"] += 1

        # 10. Persist today's record into rolling memory so future requests remember it
        _append_history(symbol, {
            "date":             body.date,
            "price":            current_price,
            "action":           action,
            "reason":           reason[:200],
            "position":         position_label,
            "regime":           rs,
            "rsi_14":           ta["rsi_14"],
            "macd_cross":       ta["macd_cross"],
            "macro_trend":      ta["macro_trend"],
            "momentum_10d":     ta["momentum_10d"],
            "unrealized_pnl":   pos["unrealized_pnl_pct"],
        })

        log.info(f"[{symbol}] {body.date} | price={current_price} | action={action} "
                 f"| position={position_label} → {state['balance']:+.0f} | rs={rs} "
                 f"| price_buf={buf_len} | memory_depth={len(past_records)}")
        return {"recommended_action": action}

    except Exception as e:
        log.err(f"Error during API inference: {e}")
        return {"recommended_action": "HOLD"}  # Safe default


# Uvicorn startup is at the bottom of this file, after all functions are defined.

# =========================================================
# PERFORMANCE TRACKER
# =========================================================
class PerformanceTracker:
    def __init__(self, initial_budget: float, start_price: float):
        self.initial      = initial_budget
        self.start_price  = start_price
        self.trades       = []
        self.equity_curve =[]
        self.total_fees   = 0.0
        self._open_price  = None
        self._open_date   = None

    def record_equity(self, date, value):
        self.equity_curve.append((date, value))

    def record_fee(self, fee):
        self.total_fees += fee

    def open_trade(self, date, price):
        self._open_price = price
        self._open_date  = date

    def close_trade(self, date, price):
        if self._open_price is None:
            return
        pnl = (price - self._open_price) / self._open_price * 100
        self.trades.append({
            "open_date": self._open_date, "close_date": date,
            "open_price": self._open_price, "close_price": price,
            "pnl_pct": round(pnl, 2), "win": pnl > 0,
        })
        self._open_price = self._open_date = None

    def max_drawdown(self):
        peak = dd = 0.0
        for _, v in self.equity_curve:
            peak = max(peak, v)
            dd   = max(dd, (peak - v) / peak * 100 if peak else 0)
        return round(dd, 2)

    def sharpe(self):
        if len(self.equity_curve) < 2:
            return 0.0
        vals    = pd.Series([v for _, v in self.equity_curve])
        returns = vals.pct_change().dropna()
        return round((returns.mean() / returns.std()) * (252 ** 0.5), 2) \
               if returns.std() != 0 else 0.0

    def summary(self, final_value, final_price, logger):
        total_return = (final_value - self.initial) / self.initial * 100
        bah          = (final_price - self.start_price) / self.start_price * 100
        alpha        = total_return - bah
        wins   = [t for t in self.trades if t["win"]]
        losses = [t for t in self.trades if not t["win"]]
        avg_w  = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0.0
        avg_l  = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0.0
        pfact  = abs(avg_w * len(wins)) / abs(avg_l * len(losses)) \
                 if losses and avg_l != 0 else float("inf")
        lines =[
            "", "=" * 65,
            "  SIMULATION SUMMARY", "=" * 65,
            f"  Initial capital   : ${self.initial:>15,.2f}",
            f"  Final value       : ${final_value:>15,.2f}",
            f"  Total AI Return   : {total_return:>+14.2f}%",
            f"  Buy & Hold Return : {bah:>+14.2f}%",
            f"  Net Alpha         : {alpha:>+14.2f}%",
            f"  Max drawdown      : {self.max_drawdown():>14.2f}%",
            f"  Sharpe ratio      : {self.sharpe():>14.2f}",
            f"  Total fees paid   : ${self.total_fees:>15,.2f}",
            "-" * 65,
            f"  Total trades      : {len(self.trades):>14}",
            f"  Wins / Losses     : {len(wins):>6} / {len(losses):<6}",
            f"  Win rate          : {len(wins)/len(self.trades)*100 if self.trades else 0:>13.1f}%",
            f"  Avg win           : {avg_w:>+13.2f}%",
            f"  Avg loss          : {avg_l:>+13.2f}%",
            f"  Profit factor     : {pfact:>14.2f}",
            "-" * 65,
        ]
        if self.trades:
            best  = max(self.trades, key=lambda t: t["pnl_pct"])
            worst = min(self.trades, key=lambda t: t["pnl_pct"])
            lines += [
                f"  Best trade        : {best['pnl_pct']:>+13.2f}%"
                f"  ({best['open_date']} -> {best['close_date']})",
                f"  Worst trade       : {worst['pnl_pct']:>+13.2f}%"
                f"  ({worst['open_date']} -> {worst['close_date']})",
            ]
        lines.append("=" * 65)
        block = "\n".join(lines)
        print(block)
        logger.info(block)


# =========================================================
# TECHNICAL ANALYSIS
# =========================================================
def compute_ta_signals(df: pd.DataFrame) -> dict:
    close = df["prices"].astype(float)
    n     = len(close)
    price = close.iloc[-1]

    sma10 = close.rolling(min(10, n)).mean().iloc[-1]
    sma20 = close.rolling(min(20, n)).mean().iloc[-1]
    sma50 = close.rolling(min(50, n)).mean().iloc[-1]

    dist10 = (price - sma10) / sma10 * 100
    dist20 = (price - sma20) / sma20 * 100
    dist50 = (price - sma50) / sma50 * 100

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1]

    bb_mid = close.rolling(20).mean().iloc[-1]
    bb_std = close.rolling(20).std().iloc[-1]
    bb_pct = (price - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = (ema12 - ema26).iloc[-1]
    sig   = (ema12 - ema26).ewm(span=9).mean().iloc[-1]

    vol14 = close.pct_change().dropna().tail(14).std() * 100
    if pd.isna(vol14): vol14 = 3.0

    mom10 = (price - close.iloc[-min(11, n)]) / close.iloc[-min(11, n)] * 100

    return {
        "dist_sma10_pct": round(dist10, 2),
        "dist_sma20_pct": round(dist20, 2),
        "dist_sma50_pct": round(dist50, 2),
        "macro_trend":    "bear" if dist50 < 0 else "bull",
        "rsi_14":         round(rsi, 1),
        "rsi_zone":       "oversold" if rsi < 35 else ("overbought" if rsi > 65 else "neutral"),
        "bb_pct":         round(bb_pct, 2),
        "bb_zone":        "near_bottom" if bb_pct < 0.2 else ("near_top" if bb_pct > 0.8 else "mid"),
        "macd_cross":     "bullish" if macd > sig else "bearish",
        "volatility_14d": round(vol14, 2),
        "momentum_10d":   round(mom10, 2),
    }


# =========================================================
# POSITION METRICS
# =========================================================
def calc_position_metrics(btc_bal, avg_buy, current, highest, vol14) -> dict:
    if btc_bal <= 0 or avg_buy <= 0:
        return {
            "avg_buy_price": 0.0, "unrealized_pnl_pct": 0.0,
            "highest_price_since_buy": 0.0, "drop_from_peak_pct": 0.0,
            "dynamic_sl_threshold": 0.0,
            "take_profit_triggered": False, "dynamic_stop_triggered": False,
        }
    pnl_pct        = (current - avg_buy) / avg_buy * 100
    drop_from_peak = (highest - current) / highest * 100 if highest > 0 else 0
    dyn_sl         = min(max(MIN_STOP_LOSS_PCT * 100, 2.0 * vol14), 25.0)
    return {
        "avg_buy_price":           round(avg_buy, 2),
        "unrealized_pnl_pct":      round(pnl_pct, 2),
        "highest_price_since_buy": round(highest, 2),
        "drop_from_peak_pct":      round(drop_from_peak, 2),
        "dynamic_sl_threshold":    round(dyn_sl, 2),
        "take_profit_triggered":   bool(pnl_pct >= TAKE_PROFIT_PCT * 100),
        "dynamic_stop_triggered":  bool(drop_from_peak >= dyn_sl),
    }


# =========================================================
# REGIME SIGNAL
# =========================================================
def regime_signal(ta: dict, pos: dict = None) -> str:
    below50    = ta["dist_sma50_pct"] < 0
    neg_mom    = ta["momentum_10d"] < -5
    not_os     = ta["rsi_14"] > 40
    oversold   = ta["rsi_zone"] == "oversold"
    overbought = ta["rsi_zone"] == "overbought"
    macd_bull  = ta["macd_cross"] == "bullish"
    macd_bear  = ta["macd_cross"] == "bearish"

    if pos:
        if pos.get("dynamic_stop_triggered"):
            return "dynamic_stop_loss_exit"
        if pos.get("take_profit_triggered") and (overbought or macd_bear):
            return "take_profit_exit"

    if oversold and macd_bull:
        return "bear_oversold_bounce"

    if ta["macro_trend"] == "bear":
        if neg_mom and not_os:
            return "strong_bear_avoid_buy"
        return "macro_bear_avoid_buy"

    if overbought and macd_bear:
        return "overbought_momentum_fading"

    if not below50 and macd_bull:
        return "bull_momentum_buy_ok"

    return "neutral"


# =========================================================
# ALLOWED ACTIONS
# =========================================================
def compute_allowed_actions(cash, btc, bars, pos, price) -> list:
    allowed   = ["hold"]
    # Fix: Calculate reserve dynamically based on available cash so bots don't permanently lock out
    spendable = cash * (1.0 - CASH_RESERVE_PCT)

    if spendable >= MIN_BUY_USD and btc == 0:
        allowed.append("buy")
    if btc > 0 and bars >= MIN_HOLD_BARS:
        allowed.append("sell")
    if btc > 0 and pos.get("dynamic_stop_triggered") and "sell" not in allowed:
        allowed.append("sell")
    return allowed


# =========================================================
# FORCED EXITS
# =========================================================
def enforce_exits(decision, ta, rs, pos, allowed, bars, date) -> dict:
    if rs == "dynamic_stop_loss_exit" and "sell" in allowed:
        return {"date": date, "action": "sell", "conviction": "strong",
                "reason": f"[FORCED] Trailing stop: -{pos['drop_from_peak_pct']:.1f}% from peak."}
    if rs == "take_profit_exit" and "sell" in allowed:
        return {"date": date, "action": "sell", "conviction": "strong",
                "reason": f"[FORCED] Take-profit: {pos['unrealized_pnl_pct']:+.1f}%."}
    return decision


# =========================================================
# TIMESFM FORECAST — v2: enriched 32-day signal
# =========================================================
def run_timesfm_forecast(
    close_prices: np.ndarray, current_price: float
) -> tuple[str, list[float] | None]:
    if tfm is None:
        return "TimesFM forecast unavailable: model not loaded", None
    try:
        point_forecast, _ = tfm.forecast(
            horizon=32,
            inputs=[close_prices.astype(np.float32)],
        )
        forecast_list =[round(float(v), 2) for v in point_forecast[0]]

        prices = np.array(forecast_list)
        days   = np.arange(1, 33)

        slope, intercept = np.polyfit(days, prices, 1)
        fitted = slope * days + intercept
        ss_res = np.sum((prices - fitted) ** 2)
        ss_tot = np.sum((prices - prices.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        if r2 >= 0.80:
            trend_label = "strong uptrend" if slope > 0 else "strong downtrend"
        elif r2 >= 0.50:
            trend_label = "moderate uptrend" if slope > 0 else "moderate downtrend"
        else:
            trend_label = "choppy/flat"

        # Protection against zero division
        safe_price = current_price if current_price > 0 else 1e-9
        avg_ret   = (prices.mean() - safe_price) / safe_price * 100
        early_ret = (prices[0:10].mean()  - safe_price) / safe_price * 100
        mid_ret   = (prices[10:20].mean() - safe_price) / safe_price * 100
        late_ret  = (prices[20:32].mean() - safe_price) / safe_price * 100

        if late_ret > mid_ret > early_ret and avg_ret > 0:
            shape_label = "accelerating"
        elif late_ret < mid_ret < early_ret and avg_ret < 0:
            shape_label = "decelerating decline"
        elif early_ret > mid_ret and mid_ret < late_ret:
            shape_label = "dip-then-recover"
        elif early_ret < mid_ret and mid_ret > late_ret:
            shape_label = "peak-then-fade"
        else:
            shape_label = "mixed"

        bull_days  = int(np.sum(prices > current_price))
        bull_ratio = bull_days / 32 * 100

        peak_idx   = int(np.argmax(prices))
        trough_idx = int(np.argmin(prices))
        peak_ret   = (prices[peak_idx]   - safe_price) / safe_price * 100
        trough_ret = (prices[trough_idx] - safe_price) / safe_price * 100

        timesfm_str = (
            f"TimesFM 32-day forecast:\n"
            f"  Trend: {slope:+.1f}$/day (r\u00b2={r2:.2f}, {trend_label})\n"
            f"  Avg return: {avg_ret:+.1f}% across 32 days\n"
            f"  Shape: early {early_ret:+.1f}% \u2192 mid {mid_ret:+.1f}% \u2192 late {late_ret:+.1f}% ({shape_label})\n"
            f"  Bullish days: {bull_days}/32 ({bull_ratio:.1f}%)\n"
            f"  Peak: {peak_ret:+.1f}% on day {peak_idx+1} | Trough: {trough_ret:+.1f}% on day {trough_idx+1}"
        )
        return timesfm_str, forecast_list

    except Exception as e:
        return f"TimesFM forecast unavailable: {e}", None


# =========================================================
# PROMPTS
# =========================================================
MARKET_ANALYST_PROMPT = """You are a market analyst analyzing stationary, relative metrics.
You receive percentage-based distance from SMAs, a regime_signal, and recent news.
Interpret these stationary features — do not recalculate raw prices.

regime_signal meanings:
- dynamic_stop_loss_exit     : price dropped by 2x volatility from local peak.
- take_profit_exit           : position up heavily AND overbought/momentum fading.
- macro_bear_avoid_buy       : price is below the 50-day moving average.
- overbought_momentum_fading : RSI overbought AND MACD turning bearish.
- strong_bear_avoid_buy      : bear trend, negative momentum, RSI not oversold.
- bear_oversold_bounce       : bear trend but oversold + MACD turning bullish.
- bull_momentum_buy_ok       : bull trend with bullish MACD crossover.
- neutral                    : mixed signals.

A timesfm_forecast field contains a rich multi-line quantitative signal:
- Trend $/day + r²: linear regression slope across all 32 forecast days; r²≥0.80 = strong conviction.
- Avg return: mean of all 32 predicted prices vs current; directional consensus.
- Shape (early→mid→late): segment returns reveal trajectory type.
  Accelerating = momentum building; peak-then-fade = near-term strength only; dip-then-recover = patience required.
- Bullish days: fraction of 32 days predicted above current; high % = persistent bid.
- Peak/Trough days: timing of extremes helps gauge entry/exit windows.

High r² + high bullish-days + accelerating shape = strong model conviction for a trend.
Low r² or choppy label = model uncertainty; defer more to regime_signal and RSI.

Write a 1-2 sentence summary referencing specific indicator values."""

RISK_MANAGER_PROMPT = """You are a portfolio risk manager. Protect capital but seek asymmetric entries.

A field called last_trades contains your memory of the last N days: date, price, action taken,
regime signal, RSI, and unrealized PnL at that time. Use this to avoid repeating losing patterns,
detect if we have been holding a losing position too long, and understand recent momentum context.

Rules:
- Never recommend an action outside the provided allowed_actions list.
- If dynamic_stop_triggered is true, preferred_action MUST be sell.
- bear_oversold_bounce signal favors BUY (high reward/risk).
- bull_momentum_buy_ok signal favors BUY.
- macro_bear_avoid_buy signal favors HOLD unless RSI < 35.
- If last_trades shows repeated HOLD with declining unrealized_pnl, consider sell.
- If last_trades shows we just bought (action=BUY in last 1-2 days), bias toward hold.

Write a 1-2 sentence risk_note."""

DECISION_PROMPT = """You are the final trade decision layer.

A field called last_trades contains your memory of the last N days: date, price, action taken,
regime signal, RSI, MACD, and unrealized PnL. Use this to understand what was decided recently
and whether the current signals represent a change or continuation of trend.

Rules (strict priority order):
1. NEVER choose an action outside the allowed_actions list.
2. regime=dynamic_stop_loss_exit AND sell allowed → action=sell, conviction=strong.
3. regime=take_profit_exit AND sell allowed → action=sell, conviction=strong.
4. regime=bull_momentum_buy_ok AND buy allowed → action=buy, conviction=moderate.
5. regime=bear_oversold_bounce AND buy allowed → action=buy, conviction=moderate.
6. regime=macro_bear_avoid_buy AND rsi_14 < 35 AND buy allowed → action=buy, conviction=weak.
7. regime=macro_bear_avoid_buy AND rsi_14 >= 35 → action=hold.
8. weak buy conviction → output hold. All buys and sells are full position, no partial trades.
9. If last_trades shows the same regime signal for 3+ days with no action change, increase conviction.

Write a 1-2 sentence reason."""


# =========================================================
# UTILS
# =========================================================
def call_structured(generator, schema_cls, system_prompt, user_content, max_new_tokens=300):
    """Call HF Inference API and parse structured JSON response into Pydantic schema."""
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    full_system = (
        f"{system_prompt}\n\n"
        f"You MUST respond with valid JSON only — no explanation, no markdown, no backticks.\n"
        f"Your response must match this exact JSON schema:\n{schema_json}"
    )
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": user_content},
    ]
    response = manager.client.chat.completions.create(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.1,
    )
    raw_text = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        return schema_cls.model_validate_json(raw_text)
    except Exception as e:
        log.err(f"LLM parsing failed: {e} | raw: {raw_text[:200]}")
        raise ValueError("Incomplete or malformed JSON output.")

def normalize_date(val):
    return str(val).split("T")[0]


# =========================================================
# BUSINESS LOGIC
# =========================================================
def apply_trade(action, conviction, price, cash, btc, avg_buy, total_cost):
    executed = False
    fee_paid = 0.0

    if action == "buy" and conviction not in MIN_CONVICTION_BUY:
        action = "hold"

    if action == "buy" and cash >= MIN_BUY_USD:
        # Fix: Dynamic spendable calculation
        spendable = cash * (1.0 - CASH_RESERVE_PCT)
        if spendable >= MIN_BUY_USD:
            fee        = spendable * FEE_RATE
            btc_bought = (spendable - fee) / price
            cash      -= spendable
            total_cost += spendable
            btc        = btc_bought
            avg_buy    = price
            fee_paid   = fee
            executed   = True

    elif action == "sell" and btc > 0:
        gross      = btc * price
        fee        = gross * FEE_RATE
        cash      += gross - fee
        fee_paid   = fee
        btc        = 0.0
        avg_buy    = 0.0
        total_cost = 0.0
        executed   = True

    return cash, btc, cash + btc * price, executed, avg_buy, total_cost, fee_paid


# =========================================================
# 3-AGENT CHAIN
# =========================================================
def analyze_market(ta, rs, pos, news, date, timesfm_str: str = "") -> MarketAnalysis | None:
    try:
        result = call_structured(
            manager.gen_market, MarketAnalysis, MARKET_ANALYST_PROMPT,
            json.dumps({"date": date, "ta_signals": ta, "regime_signal": rs,
                        "pos_metrics": pos, "news_snippets": news,
                        "timesfm_forecast": timesfm_str}, indent=2),
            max_new_tokens=300,
        )
        result.date = date
        return result
    except Exception as e:
        log.warn(f"analyze_market failed: {e}")
        return None

def assess_risk(ta, rs, pos, portfolio, last_trades, allowed, bars) -> RiskAssessment | None:
    try:
        result = call_structured(
            manager.gen_risk, RiskAssessment, RISK_MANAGER_PROMPT,
            json.dumps({"portfolio_state": portfolio, "ta_signals": ta,
                        "regime_signal": rs, "pos_metrics": pos,
                        "last_trades": last_trades, "allowed_actions": allowed,
                        "can_buy": "buy" in allowed, "can_sell": "sell" in allowed,
                        "bars_since_buy": bars}, indent=2),
            max_new_tokens=250,
        )
        result.position_state = (
            "long"  if portfolio.get("balance", 0) > 0 else
            "short" if portfolio.get("balance", 0) < 0 else
            "flat"
        )
        if result.preferred_action not in allowed:
            result.preferred_action = "hold"
        return result
    except Exception as e:
        log.warn(f"assess_risk failed: {e}")
        return None

def decide_action(mkt, risk, ta, rs, pos, allowed, bars, date, past_records=None) -> TradeDecision | None:
    try:
        result = call_structured(
            manager.gen_decision, TradeDecision, DECISION_PROMPT,
            json.dumps({"date": date, "allowed_actions": allowed,
                        "bars_since_buy": bars, "ta_signals": ta,
                        "regime_signal": rs, "pos_metrics": pos,
                        "market_view": mkt.model_dump() if mkt else None,
                        "risk_view":   risk.model_dump() if risk else None,
                        "last_trades": past_records or []}, indent=2),
            max_new_tokens=200,
        )
        result.date = date
        if result.action not in allowed:
            result.action = "hold"
        return result
    except Exception as e:
        log.warn(f"decide_action failed: {e}")
        return None

def fallback_decision(ta, rs, pos, allowed, mkt, risk, date):
    action = "hold"
    if pos.get("dynamic_stop_triggered") and "sell" in allowed:
        action = "sell"
    elif pos.get("take_profit_triggered") and "sell" in allowed and ta["rsi_zone"] == "overbought":
        action = "sell"
    elif "sell" in allowed and ta["rsi_zone"] == "overbought" and ta["macd_cross"] == "bearish":
        action = "sell"
    elif "buy" in allowed and ta["rsi_14"] < 35:
        action = "buy"
    elif "buy" in allowed and rs in {"bull_momentum_buy_ok", "bear_oversold_bounce"}:
        action = "buy"
    # Fix: mkt and risk are Pydantic model instances — use attribute access, not .get()
    reason = " ".join([
        (mkt.summary    if mkt  else ""),
        (risk.risk_note if risk else ""),
    ]).strip() or "Rule-based fallback."
    return {"date": date, "action": action, "conviction": "moderate", "reason": reason[:240]}


# =========================================================
# SIMULATION LOOP (Runs only if script executed directly)
# =========================================================
if __name__ == "__main__":
    import uvicorn
    if os.getenv("RUN_LIVE_API", "true").lower() == "true":
        print("Starting Agentic Trading API...")
        uvicorn.run(app, host="0.0.0.0", port=62237, log_level="info")
    else:
        if not os.path.exists(DATA_FILE):
            print(f"Data file not found: {DATA_FILE}. Exiting simulation.")
            exit(0)

        df = pd.read_json(DATA_FILE, lines=True)
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = df["date"].astype(str)

        cash           = INITIAL_BUDGET_USD
        btc            = 0.0
        avg_buy        = 0.0
        total_cost     = 0.0
        bars_since_buy = 9999
        peak_price     = 0.0
        trade_history  = []
        pos_history    = []

        start_price = float(df.iloc[WINDOW_SIZE - 1]["prices"])
        tracker     = PerformanceTracker(INITIAL_BUDGET_USD, start_price)

        log.info(f"Model    : {MODEL_ID}")
        log.info(f"Data     : {DATA_FILE} | rows={len(df)} | start_price=${start_price:,.2f}")
        log.info(f"Config   : TP={TAKE_PROFIT_PCT*100:.0f}%  MinSL={MIN_STOP_LOSS_PCT*100:.0f}%  "
                 f"MaxExposure={MAX_EXPOSURE_PCT*100:.0f}%  CashReserve={CASH_RESERVE_PCT*100:.0f}%")

        open(HISTORY_FILE,  "w", encoding="utf-8").close()
        open(TIMESFM_FILE,  "w", encoding="utf-8").close()

        for i in range(WINDOW_SIZE - 1, len(df)):
            window       = df.iloc[i - WINDOW_SIZE + 1 : i + 1].copy()
            date         = normalize_date(window.iloc[-1]["date"])
            price        = float(window.iloc[-1]["prices"])

            peak_price   = max(peak_price, price) if btc > 0 else 0.0

            ta  = compute_ta_signals(window)
            pos = calc_position_metrics(btc, avg_buy, price, peak_price, ta["volatility_14d"])
            rs  = regime_signal(ta, pos)
            news = (window["news"].astype(str).str[:300].tail(5).tolist()
                    if "news" in window.columns else [])

            # --- TimesFM forecast ---
            close_arr                = window["prices"].astype(float).to_numpy()
            timesfm_str, tfm_prices  = run_timesfm_forecast(close_arr, price)
            log.info(f"TimesFM | {timesfm_str}")
            with open(TIMESFM_FILE, "a", encoding="utf-8") as tf:
                tf.write(json.dumps({
                    "date":              date,
                    "current_price":     price,
                    "forecast_horizon":  32,
                    "predicted_prices":  tfm_prices,
                    "pred_7d":           tfm_prices[6]  if tfm_prices else None,
                    "pred_32d":          tfm_prices[31] if tfm_prices else None,
                    "ret_7d_pct":        round((tfm_prices[6]  - price) / price * 100, 4) if tfm_prices else None,
                    "ret_32d_pct":       round((tfm_prices[31] - price) / price * 100, 4) if tfm_prices else None,
                    "timesfm_signal":    timesfm_str,
                }) + "\n")

            allowed = compute_allowed_actions(cash, btc, bars_since_buy, pos, price)
            portfolio_state = {
                "cash_balance":           round(cash, 2),
                "balance":                round(btc, 8),
                "portfolio_value":        round(cash + btc * price, 2),
                "bars_since_buy":         bars_since_buy,
                "unrealized_pnl_pct":     pos["unrealized_pnl_pct"],
                "drop_from_peak_pct":     pos["drop_from_peak_pct"],
                "dynamic_stop_triggered": pos["dynamic_stop_triggered"],
            }

            log.day(date, price, ta["rsi_14"], ta["macro_trend"], rs,
                    pos["unrealized_pnl_pct"], allowed)

            mkt  = analyze_market(ta, rs, pos, news, date, timesfm_str)
            risk = assess_risk(ta, rs, pos, portfolio_state,
                               pos_history[-LAST_TRADES_WINDOW:], allowed, bars_since_buy)

            mkt_dict  = mkt.model_dump()  if mkt  else None
            risk_dict = risk.model_dump() if risk else None

            if mkt and risk:
                decision_obj = decide_action(mkt, risk, ta, rs, pos, allowed, bars_since_buy, date)
                decision = decision_obj.model_dump() if decision_obj else \
                           fallback_decision(ta, rs, pos, allowed, mkt, risk, date)
            else:
                decision = fallback_decision(ta, rs, pos, allowed, mkt, risk, date)

            decision = enforce_exits(decision, ta, rs, pos, allowed, bars_since_buy, date)

            (cash, btc, pv, executed,
             avg_buy, total_cost, fee) = apply_trade(
                decision["action"], decision["conviction"], price,
                cash, btc, avg_buy, total_cost,
            )

            if executed:
                tracker.record_fee(fee)
                if decision["action"] == "buy":
                    tracker.open_trade(date, price)
                    if peak_price == 0.0:
                        peak_price = price
                elif decision["action"] == "sell":
                    tracker.close_trade(date, price)
            tracker.record_equity(date, pv)

            if decision["action"] == "buy" and executed:
                bars_since_buy = 0
            elif decision["action"] == "sell" and executed and btc < 1e-8:
                bars_since_buy = 9999
            else:
                bars_since_buy += 1

            log.trade(date, decision["action"], decision["conviction"],
                      price, pv, pos["unrealized_pnl_pct"], decision["reason"], executed)

            row = {
                "date": date, "action": decision["action"],
                "conviction": decision["conviction"], "prices": price,
                "cash_balance": round(cash, 2), "btc_balance": round(btc, 8),
                "portfolio_value": round(pv, 2), "trade_executed": executed,
                "reason": decision["reason"], "allowed_actions": allowed,
                "market_view": mkt_dict, "risk_view": risk_dict,
                "ta_signals": ta, "pos_metrics": pos, "regime_signal": rs,
            }
            trade_history.append(row)
            if executed and decision["action"] in {"buy", "sell"}:
                pos_history.append(row)
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            time.sleep(SLEEP)

        # FINAL SUMMARY
        final_price = float(df.iloc[-1]["prices"])
        tracker.summary(cash + btc * final_price, final_price, log)