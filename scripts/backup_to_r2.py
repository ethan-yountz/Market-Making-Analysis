"""Nightly backup: export new Postgres rows to Parquet and upload to Cloudflare
R2 (S3-compatible, free egress). Idempotent and incremental — it tracks the
last exported row id per table in a small ``backup_state`` table, so re-running
only ships new rows.

Run from cron (Railway cron service or any scheduler):
    python scripts/backup_to_r2.py

Required env:
    DATABASE_URL          postgres connection string
    R2_ENDPOINT_URL       https://<accountid>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET             target bucket name
Optional:
    R2_PREFIX             key prefix (default "kalshi-lob")
    BACKUP_BATCH          max rows per parquet file (default 500000)
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

TABLES = ("orderbook_events", "trades")


def _client():
    import boto3  # lazy: only needed at backup time

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _ensure_state(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS backup_state ("
            "table_name TEXT PRIMARY KEY, last_id BIGINT NOT NULL DEFAULT 0)"
        )
    conn.commit()


def _last_id(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT last_id FROM backup_state WHERE table_name=%s", (table,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def _set_last_id(conn, table: str, last_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO backup_state (table_name,last_id) VALUES (%s,%s) "
            "ON CONFLICT (table_name) DO UPDATE SET last_id=EXCLUDED.last_id",
            (table, last_id),
        )
    conn.commit()


def _export_table(conn, s3, bucket: str, prefix: str, table: str, batch: int) -> int:
    start = _last_id(conn, table)
    shipped = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {table} WHERE id > %s ORDER BY id LIMIT %s",
                (start, batch),
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        if not rows:
            break
        # JSONB columns arrive as Python objects; stringify for stable parquet.
        data = {c: [] for c in cols}
        for r in rows:
            for c, v in zip(cols, r):
                data[c].append(v if not isinstance(v, (dict, list)) else _to_json(v))
        table_pa = pa.table({c: pa.array([_coerce(x) for x in data[c]]) for c in cols})
        buf = io.BytesIO()
        pq.write_table(table_pa, buf, compression="zstd")
        buf.seek(0)
        last_id = rows[-1][cols.index("id")]
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{prefix}/{table}/dt={ts}/{table}_{start + 1}_{last_id}.parquet"
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        _set_last_id(conn, table, int(last_id))
        shipped += len(rows)
        start = int(last_id)
        print(f"  {table}: shipped {len(rows)} rows -> s3://{bucket}/{key}")
        if len(rows) < batch:
            break
    return shipped


def _to_json(v):
    import json

    return json.dumps(v, separators=(",", ":"))


def _coerce(x):
    # Parquet-friendly scalars; everything non-trivial already stringified.
    from decimal import Decimal

    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, datetime):
        return x.isoformat()
    return x


def main() -> int:
    import psycopg2

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2
    bucket = os.environ["R2_BUCKET"]
    prefix = os.environ.get("R2_PREFIX", "kalshi-lob")
    batch = int(os.environ.get("BACKUP_BATCH", "500000"))

    conn = psycopg2.connect(dsn)
    try:
        _ensure_state(conn)
        s3 = _client()
        total = 0
        for table in TABLES:
            total += _export_table(conn, s3, bucket, prefix, table, batch)
        print(f"backup complete: {total} rows shipped")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
