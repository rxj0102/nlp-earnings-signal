# NLP Earnings Signal — FinBERT Return-Predictive Alpha

[![CI](https://github.com/rxj0102/nlp-earnings-signal/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/nlp-earnings-signal/actions/workflows/ci.yml)

A systematic NLP signal that extracts **tone change** from S&P 500 earnings call transcripts using FinBERT, and constructs a long-short equity strategy (pipeline and methodology; see the status section below for what the committed outputs do and do not show).

---

## Methodology

### 1. Data Acquisition
- **Source:** live scrape of Motley Fool transcripts where available, otherwise a synthetic demo dataset (100 tickers x 12 quarters)
- **Format:** `{ticker, date, quarter, text}` stored as JSON
- **Scraping:** BeautifulSoup + requests targeting Motley Fool transcripts; falls back to a curated synthetic dataset preserving realistic sentiment autocorrelation

### 2. NLP Pipeline — FinBERT
- Model: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) — domain-adapted BERT trained on financial text
- Each transcript is split into paragraphs and classified as **positive / negative / neutral**
- Call-level aggregation:
  ```
  positive_ratio = #positive_paragraphs / total
  net_tone       = positive_ratio − negative_ratio
  tone_change    = net_tone(t) − net_tone(t−1)   ← key alpha signal
  ```
- `tone_change` captures **management sentiment momentum**, filtering out persistent tone bias

### 3. Signal Construction
- Forward returns computed at **1, 5, 10, 20 calendar days** post-earnings
- **IC (Spearman rank correlation)** measures how well `tone_change` ranks future returns
- Decile analysis tests monotonicity: top decile should earn the highest returns
- **Baseline comparison:** simple positive/negative word-count ratio — quantifies FinBERT's incremental value

### 4. Long-Short Backtest
- **Universe:** ~100 S&P 500 stocks with quarterly earnings coverage
- **Rebalancing:** monthly
- **Long:** top 3 deciles of `tone_change` (improving sentiment)
- **Short:** bottom 3 deciles (deteriorating sentiment)
- **Transaction cost:** 10 bps one-way
- **OOS split:** train 2018–2020 | test 2021–2023

### 5. Integration with Momentum
- Incremental IC analysis: `tone_change` is residualized on a `net_tone`-based proxy (not price momentum; see limitations)

---

## Status of results (read this first)

The committed outputs in `outputs/` come from the **demo configuration**: synthetic transcripts (`scraper/fetch_transcripts.py` generates 100 tickers x 12 quarters), the word-count mock scorer, and whatever returns were cached at the time. They are **not evidence of predictive power**, and the earlier README tables (IC 0.03-0.08, Sharpe 0.5-1.1) were target ranges, not measured results. They have been removed.

What the committed outputs actually show (mock scorer, ~1,100 observations):

| Signal | Horizon | IC | t-stat |
|--------|---------|----|--------|
| `tone_change` | 5d | -0.045 | -1.47 |
| `tone_change` | 20d | -0.019 | -0.60 |
| word-count baseline | 20d | -0.020 | -0.64 |

No horizon is statistically distinguishable from zero, and the 2021-2023 test window of the long-short backtest is negative (about 22 monthly observations in total, far too few to conclude anything).

Known limitations:
- Transcripts are synthetic unless the live Motley Fool scrape succeeds (it needs >= 50 transcripts).
- Return fetching now **raises** if prices cannot be downloaded. Simulated returns (which inject a signal by construction, so any IC on them is circular) require `--allow-synthetic` / `allow_synthetic=True`.
- `momentum_proxy` in the incremental-IC step is a rolling mean of `net_tone`, **not price momentum**, so that step does not test incremental value over price momentum.
- Entry is the close on the call date; calls after the close or before the open make the reaction window imprecise.
- Genuine results require `python run_pipeline.py --real` on real transcripts (not included) plus a clean price download.

---

## Project Structure

```
nlp-earnings-signal/
├── scraper/
│   └── fetch_transcripts.py    # Motley Fool scraper + synthetic fallback
├── nlp/
│   └── finbert_scorer.py       # FinBERT inference + tone_change computation
├── signals/
│   └── ic_analysis.py          # IC, decile analysis, FinBERT vs. baseline
├── backtest/
│   └── longshort_backtest.py   # L/S strategy, metrics, incremental IC
├── notebooks/
│   └── nlp_signal_walkthrough.ipynb
├── outputs/
│   ├── ic_by_horizon.csv
│   ├── decile_returns.png
│   ├── strategy_performance.csv
│   └── equity_curve.png
├── run_pipeline.py              # One-command end-to-end runner
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline (mock FinBERT — fast)
python run_pipeline.py

# Run with real FinBERT (GPU recommended)
python run_pipeline.py --real

# Or run step-by-step
python scraper/fetch_transcripts.py
python nlp/finbert_scorer.py
python signals/ic_analysis.py
python backtest/longshort_backtest.py
```

---

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.

