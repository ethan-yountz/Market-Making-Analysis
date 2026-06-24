# Deploy the Order-Book Logger (go-live guide)

Step-by-step to get the Kalshi L2 logger collecting data 24/7 on Railway, with
nightly backups to Cloudflare R2. Architecture and internals are in
[docs/RECORDER.md](docs/RECORDER.md).

> **Heads up — credentials are not in this repo.** `secrets/` is gitignored, so
> on a fresh machine you won't have the key files. You only need the *values*:
> your **API key id** and your **RSA private key (PEM)**. Get them from the
> machine where `secrets/` lives, or generate a new key pair at
> kalshi.com → Settings → API. You configure them as Railway variables (below) —
> the clipboard commands in step 1 only work where the `secrets/` files exist.

---

## Part 1 — Get the logger live (~10 min)

### 1. Put your credentials on the clipboard
On the machine that has `secrets/` (Windows PowerShell):
```powershell
Get-Content secrets\kalshi_key_id.txt -Raw | Set-Clipboard        # the key id
Get-Content secrets\kalshi_private_key.pem -Raw | Set-Clipboard    # the full multi-line PEM
```
macOS/Linux:
```bash
cat secrets/kalshi_key_id.txt | pbcopy        # (xclip -sel clip on Linux)
cat secrets/kalshi_private_key.pem | pbcopy
```

### 2. Create the Railway project
- Go to [railway.app](https://railway.app) → sign in with GitHub.
- **New Project → Deploy from GitHub repo →** select `ethan-yountz/Market-Making-Analysis`.
- Railway reads `railway.json` + `requirements.txt`, builds with Nixpacks, and
  runs `python scripts/record_lob.py`.
- The first deploy **crash-loops until you add the variables below** — expected.

### 3. Add Postgres
- Project canvas: **New → Database → PostgreSQL.**
- Open your **logger service → Variables → New Variable → Add Reference →** pick
  the Postgres service's `DATABASE_URL`.
- This is what switches the storage sink from SQLite to Postgres — no code change.

### 4. Add your Kalshi credentials
Same **Variables** tab on the logger service:
| Variable | Value |
|---|---|
| `KALSHI_API_KEY_ID` | your key id |
| `KALSHI_PRIVATE_KEY` | the **entire PEM**, including the `-----BEGIN/END RSA PRIVATE KEY-----` lines (Railway's editor accepts multi-line) |

### 5. Deploy & verify
- Saving variables triggers a redeploy. Open **Deployments → View Logs.**
- ✅ Live when you see:
  ```
  storage sink: PostgresSink
  subscribed ['orderbook_delta', 'trade'] x N markets
  alive: N books, NNNN rows written, counts={...}      ← every 60s
  ```

It's now recording the full MLB slate until you stop it.

---

## Part 2 — Nightly backup to Cloudflare R2 (optional)

### 6. In Cloudflare
- Create an R2 bucket (e.g. `kalshi-lob`).
- Create an **R2 API Token**; note the **Access Key ID**, **Secret Access Key**,
  and your S3 endpoint `https://<account-id>.r2.cloudflarestorage.com`.

### 7. In Railway — add a cron service
- **New → GitHub Repo** (same repo) to add a second service.
- **Settings → Deploy → Custom Start Command:** `python scripts/backup_to_r2.py`
- **Settings → Cron Schedule:** `0 9 * * *` (daily 09:00 UTC)
- **Variables:** the `DATABASE_URL` reference (as in step 3) plus:
  | Variable | Value |
  |---|---|
  | `R2_ENDPOINT_URL` | `https://<account-id>.r2.cloudflarestorage.com` |
  | `R2_ACCESS_KEY_ID` | from the R2 token |
  | `R2_SECRET_ACCESS_KEY` | from the R2 token |
  | `R2_BUCKET` | `kalshi-lob` |

Ships only new rows each night (tracked in a `backup_state` table) as zstd
Parquet — idempotent, so re-runs never duplicate.

---

## Run it locally instead (no cloud)
```bash
python scripts/record_lob.py        # writes to data/lob.sqlite, zero config
python scripts/probe_ws.py          # sanity-check auth + live feed any time
```

## Worth knowing
- **Cost:** a small worker + Postgres on Railway is ~a few $/month (usage-based).
  R2 has no egress fees, so backups stay cheap.
- **Storage growth:** ~hundreds of MB/day in Postgres. Once R2 backups are
  confirmed, a retention prune on the Postgres tables can keep it bounded.
- **Don't sleep the machine** only matters for the *local* run — the hosted
  Railway worker runs continuously.
