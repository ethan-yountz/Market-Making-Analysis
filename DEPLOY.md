# Deploy the Order-Book Logger (go-live guide)

Step-by-step to get the Kalshi L2 logger collecting data 24/7 on Railway, with
nightly backups to Cloudflare R2. Architecture and internals are in
[docs/RECORDER.md](docs/RECORDER.md).

> **Heads up — credentials are not in this repo.** `secrets/` is gitignored, so
> on a fresh machine you won't have the key files. You only need the *values*:
> your **API key id** and your **RSA private key (PEM)**. Get them from the
> machine where `secrets/` lives, or generate a new key pair (next section).
> You configure them as Railway variables (below) — the clipboard commands in
> step 1 only work where the `secrets/` files exist.

---

## Part 0 — Get Kalshi API credentials (if you don't have them)

You need two things: an **API Key ID** (a UUID) and an **RSA private key** (PEM).
Generate them once at Kalshi:

1. Log in → **Profile Settings** ([kalshi.com/account/profile](https://kalshi.com/account/profile)) → **API Keys**.
2. Click **Create New API Key**.
3. Kalshi shows a **Key ID** and a **Private Key** (RSA, begins
   `-----BEGIN RSA PRIVATE KEY-----`) and downloads the private key as a `.txt`.
   ⚠️ The private key is shown **once and never again** — save it immediately.
   If you lose it, delete the key and create a new one.

The private key never leaves your machine/host; it's only used locally to sign
requests. **One key pair works on every machine and on Railway at the same
time** — you don't need a separate key per machine. Only generate a new one to
rotate or if you've lost the private key.

**Where it goes:**
- *Local runs* — create a `secrets/` folder in the repo and save:
  - the key id → `secrets/kalshi_key_id.txt`
  - the PEM (full contents of the downloaded `.txt`) → `secrets/kalshi_private_key.pem`

  `secrets/` is gitignored — **never commit it.** Verify the key loads:
  ```bash
  python scripts/probe_ws.py          # should print "auth: loaded" + live messages
  ```
- *Railway* — paste the same two values into the `KALSHI_API_KEY_ID` /
  `KALSHI_PRIVATE_KEY` variables in Part 1, step 4 (no `secrets/` folder needed
  on the host).

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

## Part 2 — Get the data somewhere durable (pick ONE)

Postgres on Railway is one fixed-size volume and a single point of failure, so
get the data off it. Two options — the first is far less setup.

### Option A (recommended for a short run) — pull to local Parquet
No extra services, no Cloudflare. Run from your laptop whenever, and once at the
end of collection.
1. Railway → **Postgres service → Connect tab** → copy the **Public Network**
   connection URL (host looks like `xxx.proxy.rlwy.net:PORT`).
2. Export to local Parquet — incremental, so re-running only fetches new rows:
   ```powershell
   $env:DATABASE_URL = "postgresql://postgres:PASS@xxx.proxy.rlwy.net:PORT/railway"
   python scripts/export_pg.py
   ```
   Files land in `data/lob_export/`.
3. If Postgres gets large, reclaim space after a clean export (deletes only rows
   already written to Parquet):
   ```powershell
   python scripts/export_pg.py --prune
   ```

### Option B (set-and-forget) — nightly backup to Cloudflare R2

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
