# Agentic Mk1 — TimesFM Trading API
### FinMMEval CLEF 2026 | Task 3: Financial Decision Making

A daily news-driven trading agent that returns **BUY / HOLD / SELL** decisions using a 3-agent LLM chain powered by Qwen2.5-7B and Google TimesFM 2.5.

---

## Architecture

```
Daily Request (news + price + history)
           │
           ▼
┌──────────────────────────────────────────┐
│           Price Buffer (500 rows)         │  ← grows every day
│     Technical Analysis (RSI, MACD, BB)   │  ← fully reliable after day 50
│           TimesFM 32-day Forecast        │  ← Google foundation model
└──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Agent 1           │  Market Analyst
│   Qwen2.5-7B 4-bit  │  → news_bias, macro_view, momentum_signal
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Agent 2           │  Risk Manager
│   Qwen2.5-7B 4-bit  │  → position_state, risk_level, preferred_action
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Agent 3           │  Decision Layer
│   Qwen2.5-7B 4-bit  │  → BUY / HOLD / SELL
└─────────────────────┘
           │
           ▼
    enforce_exits()      ← hard override for stop loss / take profit
           │
           ▼
  {"recommended_action": "BUY"}
```

---

## Models

| Model | Role | Size | Precision |
|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 3-Agent Chain | 7B | 4-bit NF4 |
| google/timesfm-2.5-200m-pytorch | Price Forecasting | 200M | float32 |

---

## Key Features

- **Per-symbol memory** — independent state tracking for each asset (BTC, TSLA, etc.)
- **Rolling price buffer** — accumulates up to 500 days of price history across calls, improving TA accuracy over time
- **10-day decision memory** — agents see the last 10 days of decisions, regime signals, and PnL
- **Forced exits** — dynamic trailing stop loss and take profit override LLM decisions for capital protection
- **Structured output** — outlines + Pydantic schemas guarantee valid JSON from the LLM every time
- **Graceful degradation** — TimesFM failure falls back to TA-only mode; all other signals remain active

---

## Technical Signals

| Signal | Description | Rows needed |
|---|---|---|
| RSI-14 | Momentum oscillator | 15 |
| MACD | Trend/momentum crossover | 35 |
| Bollinger Bands | Volatility bands | 20 |
| SMA-10/20/50 | Moving average distances | 50 |
| TimesFM 32-day | Neural price forecast | 10+ |

---

## Regime Signals

| Signal | Meaning | Action bias |
|---|---|---|
| `bull_momentum_buy_ok` | Bull trend + bullish MACD | BUY |
| `bear_oversold_bounce` | Oversold + MACD turning | BUY |
| `dynamic_stop_loss_exit` | Drop > 2× volatility from peak | SELL (forced) |
| `take_profit_exit` | +10% gain + overbought | SELL (forced) |
| `macro_bear_avoid_buy` | Price below SMA-50 | HOLD |
| `strong_bear_avoid_buy` | Bear + negative momentum | HOLD/SELL |
| `overbought_momentum_fading` | RSI overbought + MACD bearish | SELL |
| `neutral` | Mixed signals | HOLD |

---

## API Reference

### `GET /health`
```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "timesfm": true
}
```

### `POST /trading_action/`

**Request:**
```json
{
  "date": "2025-01-15",
  "price": {"BTC": 98400.0},
  "news": {"BTC": ["Bitcoin ETF inflows remain strong"]},
  "symbol": ["BTC"],
  "momentum": {"BTC": "bullish"},
  "10k": null,
  "10q": null,
  "history_price": {
    "BTC": [
      {"date": "2025-01-05", "price": 94200.0},
      {"date": "2025-01-14", "price": 97600.0}
    ]
  }
}
```

**Response:**
```json
{"recommended_action": "BUY"}
```

Valid actions: `BUY` | `HOLD` | `SELL`

---

## Setup

### Requirements
- Python 3.11+
- NVIDIA GPU with 16GB+ VRAM (24GB recommended)
- CUDA 12.x

### Install
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 --no-deps
pip install huggingface_hub==1.12.0
pip install accelerate bitsandbytes outlines fastapi uvicorn \
            pydantic pandas numpy python-dotenv httpx requests \
            tokenizers safetensors tqdm regex filelock packaging PyYAML
pip install "git+https://github.com/google-research/timesfm.git" --no-deps
```

### Run
```bash
echo "RUN_LIVE_API=true" > .env
python API.py
```

Server starts on `http://0.0.0.0:62237`

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `TAKE_PROFIT_PCT` | 0.10 | Exit long at +10% gain |
| `MIN_STOP_LOSS_PCT` | 0.05 | Minimum trailing stop at -5% |
| `CASH_RESERVE_PCT` | 0.25 | Keep 25% cash reserved |
| `MEMORY_WINDOW` | 10 | Days of decision history per symbol |
| `TA_MIN_ROWS` | 50 | Rows for full TA reliability |
| `WINDOW_SIZE` | 50 | Rolling window for TA computation |

---

## Performance Notes

- **Days 1–14:** RSI reliable, MACD/BB partially substituted with neutral defaults
- **Days 15–35:** MACD becomes reliable
- **Days 36–50:** All signals fully reliable including SMA-50
- **Day 50+:** Full accuracy, all TA signals computed from real data

The system is designed to be conservative early (when TA data is sparse) and become more decisive as the price buffer grows.

---

## Competition

Submitted to: [FinMMEval CLEF 2026 — Task 3: Financial Decision Making](https://huggingface.co/spaces/TheFinAI/Agent-Market-Arena)

Dataset: [MBZUAI/finmmeval-lab-clef2026](https://huggingface.co/datasets/MBZUAI/finmmeval-lab-clef2026)

Evaluation metrics: Cumulative Return (CR), Sharpe Ratio (SR), Maximum Drawdown (MD), Daily Volatility (DV), Annualized Volatility (AV)
