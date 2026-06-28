"""Pull logged data off (Railway) Postgres into local Parquet — the lightweight
alternative to the R2 cron. Run it from your laptop whenever you like and once
at the end of collection; the data ends up durably on your own disk.

Incremental: a local state file (``_export_state.json`` in the output dir) tracks
the last exported row id per table, so re-runs only fetch new rows.

    # Point at Railway's PUBLIC Postgres URL (Postgres service -> Connect tab,
    # "Public Network" — host looks like xxx.proxy.rlwy.net:PORT):
    #   PowerShell:  $env:DATABASE_URL = "postgresql://postgres:PASS@xxx.proxy.rlwy.net:PORT/railway"
    #   bash:        export DATABASE_URL="postgresql://postgres:PASS@xxx.proxy.rlwy.net:PORT/railway"
    python scripts/export_pg.py
    python scripts/export_pg.py --db-url postgresql://... --out data/lob_export

    # At the very end, free up the DB after a clean export (keeps Postgres from
    # growing without bound). Deletes only rows already written to Parquet:
    python scripts/export_pg.py --prune
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TABLES = ("orderbook_events", "trades")


def _coerce(x):
    if isinstance(x, (dict, list)):
        return json.dumps(x, separators=(",", ":"))
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, datetime):
        return x.isoformat()
    return x


def _load_state(path: Path) -> dict[str, int]:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_state(path: Path, state: dict[str, int]) -> None:
    path.write_text(json.dumps(state, indent=2))


def _export_table(conn, out_dir: Path, table: str, start: int, batch: int) -> int:
    last = start
    shipped = 0
    table_dir = out_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)
    while True:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {table} WHERE id > %s ORDER BY id LIMIT %s",
                (last, batch),
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        if not rows:
            break
        data = {c: [] for c in cols}
        for r in rows:
            for c, v in zip(cols, r):
                data[c].append(_coerce(v))
        pa_table = pa.table({c: pa.array(data[c]) for c in cols})
        first_id, last = rows[0][cols.index("id")], rows[-1][cols.index("id")]
        fname = table_dir / f"{table}_{first_id}_{last}.parquet"
        pq.write_table(pa_table, fname, compression="zstd")
        shipped += len(rows)
        print(f"  {table}: {len(rows)} rows -> {fname}")
        if len(rows) < batch:
            break
    return shipped, last


def _prune(conn, table: str, upto_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {table} WHERE id <= %s", (upto_id,))
        n = cur.rowcount
    conn.commit()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-url", default=os.environ.get("DATABASE_URL"),
                    help="Postgres URL; defaults to $DATABASE_URL")
    ap.add_argument("--out", default="data/lob_export", help="output directory")
    ap.add_argument("--batch", type=int, default=200_000, help="rows per Parquet file")
    ap.add_argument("--prune", action="store_true",
                    help="after exporting, DELETE exported rows from Postgres")
    args = ap.parse_args()

    if not args.db_url:
        print("set DATABASE_URL or pass --db-url (use Railway's PUBLIC Postgres URL)")
        return 2

    import psycopg2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_export_state.json"
    state = _load_state(state_path)

    conn = psycopg2.connect(args.db_url)
    try:
        total = 0
        for table in TABLES:
            start = int(state.get(table, 0))
            shipped, last = _export_table(conn, out_dir, table, start, args.batch)
            total += shipped
            if shipped:
                state[table] = int(last)
                _save_state(state_path, state)  # persist progress per table
            if args.prune and last > start:
                deleted = _prune(conn, table, last)
                print(f"  {table}: pruned {deleted} rows from Postgres")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"done @ {ts}: exported {total} new rows -> {out_dir}/")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
