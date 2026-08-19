# Kalshi Market Microstructure & Market Making

[![Tests](https://github.com/ethan-yountz/Market-Making-Analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/ethan-yountz/Market-Making-Analysis/actions/workflows/tests.yml)

An event-driven research platform for studying liquidity provision in Kalshi
sports prediction markets. The project combines a live Level-2 order-book
pipeline, a calibrated counterfactual fill model, and a shared backtesting
engine for classical and reinforcement-learning market makers.

The checked-in strategy study uses NBA pregame moneyline markets. The live L2
recorder was deployed against MLB markets to collect the order-book data that
Kalshi does not make available historically.

## Key result

In the stored 2025–26 NBA evaluation (`2,895` market episodes), the calibrated
Avellaneda–Stoikov strategy produced:

- **55.4% lower average loss** than the fixed-spread benchmark: `-$10.44`
  versus `-$23.39` per market episode.
- **83.5% lower mean absolute inventory**: `75.4` versus `458.4` contracts.
- **50% fewer fees paid**: `$54.8k` versus `$108.7k` in aggregate.

Both strategies remained unprofitable after fees and adverse selection. The
result is therefore a reduction in loss and inventory risk—not evidence of a
profitable live-trading strategy. Full outputs are checked into
[`results/nba_2025-26`](results/nba_2025-26/summary.csv).

![Cumulative strategy PnL](results/nba_2025-26/cumulative_pnl.png)

## What this project demonstrates

- **Market-data engineering:** authenticated WebSocket ingestion, sequence-gap
  detection, snapshot/delta reconstruction, active-market discovery, and
  durable SQLite/Postgres storage.
- **Microstructure simulation:** an event-driven engine with passive quote
  constraints, inventory limits, exact fee accounting, and configurable
  terminal inventory treatment.
- **Fill-model calibration:** tape-anchored fills plus an optional bounded
  latent-demand model for price-improving quotes.
- **Strategy research:** fixed spread, inventory-skewed fixed spread,
  Avellaneda–Stoikov, tabular SARSA(λ), and dueling double DQN policies.
- **Reproducible evaluation:** a shared engine for every strategy, explicit
  train/evaluation seasons, persisted models, result artifacts, and automated
  tests for accounting and execution rules.

## Architecture

```mermaid
flowchart LR
    subgraph historical["Historical research"]
        REST[Kalshi REST API] --> DL[Candles + trade tape]
        DL --> CAL[Volatility and fill calibration]
        CAL --> BT[Event-driven backtester]
        BT --> EVAL[Metrics and figures]
    end

    subgraph live["Live L2 pipeline"]
        WS[Kalshi WebSocket] --> BOOK[Snapshot + delta reconstruction]
        BOOK --> DB[(SQLite / Postgres)]
        DB --> PQ[Incremental Parquet export]
    end

    PQ -. validates queue and depth assumptions .-> CAL
```

The live recorder subscribes to `orderbook_delta` and `trade`, maintains one
in-memory book per market, and reconnects for fresh snapshots whenever a
sequence gap is detected. Its default `topbook` mode stores a self-contained
near-touch snapshot only when the best bid or ask changes; every trade is
retained. A heavier replayable snapshot-and-delta mode is also available.

## Backtest model

Kalshi exposes historical one-minute bid/ask candles and the trade tape, but
not historical depth. The simulator therefore separates fills into two
explicit layers:

1. **Tape-anchored fills.** A trade through the strategy's quote fills the
   quote; a trade at a joined touch fills a configurable queue share `rho`;
   a trade at a price-improving quote fills it before the old touch.
2. **Latent demand (optional).** Price-improving quotes may attract incremental
   Poisson flow using a calibrated, capped intensity curve. This mode never
   replaces the tape and caps synthetic volume to limit extrapolation.

All strategies share the same fill engine, fee schedule, inventory cap, event
stream, and terminal accounting. PnL decomposes as:

```text
PnL = spread captured + inventory PnL - fees
```

The engine checks this identity in its test suite.

## Strategies

- **Fixed Spread** joins or improves the touch around the current midpoint,
  with an optional linear inventory skew.
- **Avellaneda–Stoikov** adapts its reservation price and spread to inventory,
  bounded local volatility, time to tip, and calibrated arrival intensity.
- **Spooner RL** uses SARSA(λ) with tile-coded inventory, spread, flow,
  volatility, time, and price-location features.
- **Deep RL** uses a temporal-convolution dueling double DQN with n-step
  returns and pluggable inventory-aware rewards.

## Repository layout

```text
kalshi_mm/
  api/          REST client, authentication, and historical downloads
  recorder/     live discovery, book reconstruction, and durable storage
  data/         normalization, event streams, and tip-time inference
  calib/        bounded volatility and fill-intensity estimation
  sim/          fees, fills, and the event-driven backtest engine
  strategies/   fixed spread, Avellaneda–Stoikov, SARSA, and DQN
  eval/         PnL decomposition, summary metrics, and plots
scripts/        numbered research workflow plus recorder/export utilities
tests/          execution, accounting, recorder, fee, reward, and strategy tests
models/         persisted RL policies and training histories
results/        checked-in evaluation tables and figures
```

## Setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

PyTorch is intentionally not a package dependency because the correct build
depends on the host's CUDA configuration. Install it separately only for deep
RL training.

## Record live L2 data

Create a Kalshi API key and store the credentials locally as:

```text
secrets/kalshi_key_id.txt
secrets/kalshi_private_key.pem
```

The entire `secrets/` directory is gitignored. Then start the MLB recorder:

```powershell
python scripts/record_lob.py --series KXMLBGAME
```

Without `DATABASE_URL`, data is written to `data/lob.sqlite`. With a Postgres
URL, the same process writes to Postgres. Export hosted data incrementally:

```powershell
$env:DATABASE_URL = "postgresql://user:password@host:port/database"
python scripts/export_pg.py
```

See [the recorder architecture](docs/RECORDER.md) and the
[Railway deployment guide](DEPLOY.md) for operational details.

## Reproduce the research workflow

```powershell
# 1. Download historical candles and trades
python scripts/01_download.py --sport nba --season 2024-25
python scripts/01_download.py --sport nba --season 2025-26

# 2. Fit microstructure parameters on the training season
python scripts/02_calibrate.py --sport nba --season 2024-25

# 3. Train optional RL policies
python scripts/04_train_spooner.py --season 2024-25
python scripts/05_train_drl.py --season 2024-25 --reward spooner

# 4. Evaluate every strategy with the same engine
python scripts/03_run_backtests.py --sport nba --season 2025-26 `
  --calib-season 2024-25 `
  --strategies fixed_spread fixed_spread_skew as spooner drl
```

Historical discovery enforces season boundaries both in the API request and
again client-side so an endpoint change cannot silently mix training and
evaluation periods.

## Limitations

- Historical queue position and depth are unobserved; queue participation is
  modeled and should be sensitivity-tested.
- Counterfactual quotes do not move the historical price path.
- Latent demand is a bounded scenario, not directly observed flow.
- Inferred tip times can differ from scheduled start times.
- Backtest performance is not live performance and is not financial advice.
