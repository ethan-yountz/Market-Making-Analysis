# Order-Book Logger

A tick-level Level-2 order-book data pipeline for Kalshi prediction markets.
It exists because Kalshi exposes **no historical order book** — only 1-minute
candlesticks and the trade tape. To calibrate a realistic fill model and run
high-fidelity backtests, we record the live book ourselves.

```
        ┌──────────────────── Railway ────────────────────┐
        │                                                  │
        │   ┌──────────────────┐      ┌────────────────┐   │
        │   │  Logger process  │────▶ │ Postgres addon │   │
        │   │   (WebSocket)    │      │   ~5–15 GB     │   │
        │   └──────────────────┘      └────────────────┘   │
        │                                     │            │
        │   ┌──────────────────┐              │            │
        │   │  Nightly backup  │ ◀────────────┘            │
        │   │   (cron job)     │                           │
        │   └──────────────────┘                           │
        └───────────│──────────────────────────────────────┘
                    ▼
            Cloudflare R2  (Parquet, free egress)
```

## What it records

For each active game market it subscribes to `orderbook_delta` + `trade`,
maintains a local book by applying the snapshot + deltas, and writes:

- **`orderbook_events`** — one row per book change: `recv_ts`, `exch_ts`,
  `ticker`, `sid`, `seq`, `msg_type` (`snapshot`/`delta`), the **full
  reconstructed book** as `yes_levels` / `no_levels` JSONB (`[[price_cents,
  qty], …]`), top-of-book `best_yes_bid` / `best_yes_ask`, and the raw `delta`
  for audit.
- **`trades`** — one row per print: prices in cents, `count`, `taker_side`,
  raw payload.

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

## Deploy on Railway (host it tonight)

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

### Nightly backup to Cloudflare R2
Add a **cron service** (or a second service with a schedule, e.g. `0 9 * * *`)
running `python scripts/backup_to_r2.py`, with these variables set:
`R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
(see `.env.example`). It exports only new rows (tracked in a `backup_state`
table) to zstd Parquet partitioned by table and date — idempotent and
incremental.

## Cost / data sizing
~9k order-book events + ~350 trades per **30 s** across the full MLB slate
(observed). Plan for a few hundred MB/day in Postgres; the nightly Parquet
copy to R2 keeps long-term storage cheap and durable.

## Résumé framing
> Engineered a tick-level order-book data pipeline ingesting Kalshi prediction
> market L2 data via WebSocket across 300+ MLB game lifecycles, implementing
> delta-based book reconstruction and a calibrated fill-probability model to
> enable full-season backtesting in the absence of historical LOB data.
