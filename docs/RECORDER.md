# Order-Book Logger

A tick-level Level-2 order-book data pipeline for Kalshi prediction markets.
It exists because Kalshi exposes **no historical order book** — only 1-minute
candlesticks and the trade tape. To calibrate a realistic fill model and run
high-fidelity backtests, we record the live book ourselves.

```
        ┌──────────────────── Railway ────────────────────┐
        │   ┌──────────────────┐      ┌────────────────┐   │
        │   │  Logger process  │────▶ │ Postgres addon │   │
        │   │   (WebSocket)    │      │                │   │
        │   └──────────────────┘      └────────────────┘   │
        └─────────────────────────────────│────────────────┘
                                          │  export_pg.py
                                          ▼  (public URL)
                            Local Parquet  (data/lob_export/)
```

## What it records

For each active game market it subscribes to `orderbook_delta` + `trade` and
maintains a local book by applying the snapshot + deltas. It runs in one of two
modes (default **`topbook`**):

- **`topbook`** (default) — writes a row only when the **top of book** (best
  yes bid/ask) moves, capturing the near-touch book within `--depth-cents`
  (default 5¢) of each side. That's all the fill model needs around the spread
  and is ~30× lighter than recording everything; every trade is still kept.
- **`full`** (`--mode full`) — snapshot rows carry the whole book; delta rows
  carry just the change, reconstructable offline by replaying in `seq` order.
  Heavy — storing the full feed filled the volume in minutes, which is why
  `topbook` is the default.

Tables:
- **`orderbook_events`** — `recv_ts`, `exch_ts`, `ticker`, `sid`, `seq`,
  `msg_type` (`book` in topbook mode; `snapshot`/`delta` in full mode),
  top-of-book `best_yes_bid` / `best_yes_ask`, and level arrays `yes_levels` /
  `no_levels` JSONB (`[[price_cents, qty], …]`). In topbook mode each row is a
  self-contained near-touch snapshot (no replay needed).
- **`trades`** — one row per print (always recorded, both modes): prices in
  cents, `count`, `taker_side`, raw payload.

Prices are normalised to integer cents in `[1, 99]`; a YES ask at price `p`
equals a NO bid at `100 − p`, so both raw sides are stored losslessly and asks
are derived in analysis.

### Correctness notes
- `seq` is monotonic **per `sid`** (per channel), across all markets on that
  channel. A gap means a possibly-missed delta, so the logger reconnects for
  fresh snapshots rather than guessing — `re-subscribe, don't patch`.
- The active market set is rediscovered every 15 min; a change forces a clean
  reconnect so new games get snapshots and finished ones drop off.

## Run it locally (zero config)

```bash
# writes to data/lob.sqlite, no database needed
python scripts/record_lob.py
# or narrow the slate / horizon
python scripts/record_lob.py --series KXMLBGAME --horizon-hours 24
```

Credentials come from `secrets/kalshi_key_id.txt` + `secrets/kalshi_private_key.pem`.
Verify the feed any time with `python scripts/probe_ws.py`.

## Deploy on Railway

1. **Create the project & Postgres**
   - `railway init` (or create a project in the dashboard), then add the
     **Postgres** plugin. Railway injects `DATABASE_URL` into the service
     automatically — `make_sink` picks it up and switches from SQLite to
     Postgres with no code change.
2. **Set credentials as service variables** (Settings → Variables):
   - `KALSHI_API_KEY_ID` — your key id
   - `KALSHI_PRIVATE_KEY` — paste the full multi-line PEM (Railway's editor
     accepts newlines; in a flat `.env`, encode them as `\n`)
3. **Deploy** — `railway up`. `railway.json` sets the start command
   (`python scripts/record_lob.py`) and an on-failure restart policy; the build
   uses `requirements.txt`.
4. **Watch the logs** — you should see `subscribed ['orderbook_delta','trade']
   x N markets` and a per-minute `alive: … rows written …` heartbeat.

### Get the data onto your machine
Run `python scripts/export_pg.py` from your laptop against Railway's **public**
Postgres URL (Postgres service → Connect tab). It pulls new rows into local
zstd Parquet under `data/lob_export/`, incrementally (a local state file tracks
the last exported id per table). Add `--prune` to delete already-exported rows
from Postgres and keep the volume bounded. Full steps in `DEPLOY.md` Part 2.

## Cost / data sizing
Default `topbook` mode: ~150 top-of-book rows + ~175 trades per **20 s** across
the full MLB slate (observed) — well under ~1 GB/day, comfortable on a small
volume for the whole run. `full` mode is ~30× heavier (multiple GB/day) and
needs frequent pruning. Either way `export_pg.py` pulls data down to local
Parquet (add `--prune` to bound the volume).
