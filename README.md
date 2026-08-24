# kuant-strategies

**25 quantitative trading strategies** built on
[`kuant-core`](https://github.com/zwmjj/kuant-core). Covers momentum,
mean-reversion, cross-asset flow, crypto, options, machine learning,
and alternative data. Each strategy is a standalone runnable module
with its own research question and reference backtest result.

## Ecosystem

This is one of **six open-source repositories** that together form a
complete quant research platform. Total ~55,000 LOC, MIT licensed.

| Repo | Role | LOC |
|---|---|---|
| [`alt-data-research`](https://github.com/zwmjj/alt-data-research) | SEC NLP + 13F alt-data alpha — combined ICIR 0.80 in-sample; component signs fitted on the same sample, so the t-stat is not a significance test | ~2.5k |
| [`kuant-research`](https://github.com/zwmjj/kuant-research) | 14 reproducible empirical studies with committed expected outputs | ~3k |
| [`kuant-core`](https://github.com/zwmjj/kuant-core) | Production quant research library — 28+ factors, walk-forward CV, 5 cost models, US + CN A-share | ~20k |
| **`kuant-strategies`** (this repo) | 25+ strategies built on kuant-core: momentum, mean-rev, crypto, options, ML, alt-data | ~17k |
| [`kuant-api`](https://github.com/zwmjj/kuant-api) | FastAPI research backend — 20 routers, Monaco IDE, WebSocket, JWT auth | ~5k |
| [`kuant-web`](https://github.com/zwmjj/kuant-web) | Next.js 16 + Tailwind + Recharts dashboard — 20 panels | ~7k |

## Strategy catalogue

### Momentum / Trend

| File | Universe | Lookback | Note |
|---|---|---|---|
| `momentum.py`          | US equity   | 12m skip-1m | Jegadeesh-Titman 1993 baseline |
| `max_sharpe.py`        | Cross-asset | rolling      | Walk-forward max-Sharpe weights |
| `multi_timeframe.py`   | US equity   | 1m/3m/6m/12m blend | Composite momentum |
| `timesfm_factor.py`    | US equity   | ML          | TimesFM forecast as factor |
| `timesfm_strategy.py`  | US equity   | ML          | TimesFM directly as strategy |

### Mean Reversion

| File | Universe | Horizon | Note |
|---|---|---|---|
| `mean_reversion_crypto.py` | Crypto majors | intraday | Bollinger reversion |
| `stat_arb.py`              | US equity     | 5m       | Pairs + basket stat arb |
| `daily_alpha.py`           | US equity     | daily    | Short-term reversal |

### Cross-Asset / Factor

| File | Universe | Factors | Note |
|---|---|---|---|
| `factors.py`             | US equity    | 28+ factor library | Signal blending + quintile sort |
| `commodities.py`         | Commodity ETF| momentum / curve | Cross-commodity rotation |
| `forex_futures.py`       | G10 FX       | momentum / carry | Futures-based FX carry |
| `cross_asset_options.py` | Multi-asset  | vol surface      | Cross-asset options overlay |

### Crypto

| File | Universe | Approach |
|---|---|---|
| `crypto_advanced.py`         | Major crypto | Multi-signal blend |
| `crypto_backtest_report.py`  | Major crypto | Full backtest reporting |
| `full_backtest_report.py`    | Mixed        | Report generator for any backtest |

### Options

| File | Universe | Strategy |
|---|---|---|
| `options_strategies.py` | US large cap | Delta-neutral income + hedging |
| `options_enhanced.py`   | US large cap | Vol surface signals |
| `dividend_capture.py`   | US large cap | Ex-div capture with option hedge |

### ML

| File | Approach |
|---|---|
| `ml_strategies.py` | LightGBM / XGBoost on factor panel |

### A-Share

| File | Universe | Note |
|---|---|---|
| `cn_intraday_momentum.py` | CSI 300 minute bars | Intraday momentum |

### Portfolio / Screening

| File | Role |
|---|---|
| `portfolio_optimizer.py` | Strategy-level weighting (risk parity / max-SR / min-var) |
| `news_screener.py`       | Event-driven screener (8-K material events) |
| `alternative_data.py`    | SEC NLP + 13F flow alpha block (see also alt-data-research) |
| `optimized_live.py`      | Live-trading wrapper (requires Alpaca) |

## Install

```bash
pip install kuant-strategies           # also installs kuant-core
pip install "kuant-strategies[alpaca]" # for live-data strategies
pip install "kuant-strategies[full]"   # everything
```

## Quickstart

```python
from kuant_core.qf.data import prepare_data
from strategies.momentum import MomentumStrategy

data = prepare_data()                           # WRDS CRSP or yfinance fallback
strat = MomentumStrategy(lookback=12, skip=1)
result = strat.run(data, rebalance="monthly")
print(result.metrics())
```

Or run the built-in backtest reports:

```python
from strategies.full_backtest_report import run_full_report
run_full_report(
    strategies=["momentum", "stat_arb", "factors", "ml_strategies"],
    output="results/",
)
```

## Strategies that need live credentials

These touch live data feeds (Alpaca) or order execution and need
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in the environment:

- `alternative_data.py` (Alpaca historical quotes for basket construction)
- `dividend_capture.py` (Alpaca ex-div calendar)
- `options_strategies.py` (Alpaca options quotes)
- `options_enhanced.py` (same)
- `optimized_live.py` (live execution wrapper)

All others run offline with yfinance / CRSP / akshare data.

## Design notes

1. **One strategy per file.** Each strategy is self-contained: research
   question at the top of the file, signal logic in the middle,
   reference backtest result + metrics at the bottom. You can read
   any strategy in 10 minutes without cross-referencing other files.

2. **Depends on kuant-core for infrastructure.** Backtester, cost
   model, signal library, walk-forward CV, gate-check system — all
   shared. This repo is the "research output" layer; kuant-core is
   the "research infrastructure" layer.

3. **Multi-asset, multi-frequency.** Monthly US equity factor
   strategies, intraday crypto, daily forex, options — all using the
   same event-driven engine from kuant-core.

4. **Every strategy has a reference backtest.** See each file's
   `__main__` block for a runnable example with hardcoded parameters
   that reproduce the README-claimed numbers.

## License

MIT. See `LICENSE`.

Data dependencies (WRDS, yfinance, akshare, Alpaca) have their own
terms; see the kuant-core README for details.
