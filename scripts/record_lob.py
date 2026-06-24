"""Order-book logger entrypoint. Reconstructs the live book and writes one row
per event to Postgres (production) or SQLite (local default).

Local (zero config -> data/lob.sqlite):
    python scripts/record_lob.py

Postgres (Railway sets DATABASE_URL automatically):
    DATABASE_URL=postgresql://user:pass@host:5432/db python scripts/record_lob.py
    python scripts/record_lob.py --series KXMLBGAME KXNBAGAME --horizon-hours 24

Credentials: secrets/kalshi_key_id.txt + secrets/kalshi_private_key.pem locally,
or env KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY (inline PEM) when hosted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.api.client import load_default_auth
from kalshi_mm.recorder.lob_recorder import MLB, LobRecorder
from kalshi_mm.recorder.storage import make_sink


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", nargs="+", default=list(MLB))
    ap.add_argument("--db-url", default=os.environ.get("DATABASE_URL"),
                    help="postgres://... ; defaults to $DATABASE_URL or SQLite")
    ap.add_argument("--horizon-hours", type=float, default=36.0)
    ap.add_argument("--rediscover-mins", type=float, default=15.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    auth = load_default_auth()
    if auth is None:
        logging.warning("no API credentials found - websocket will likely be rejected")
    sink = make_sink(args.db_url)
    logging.info("storage sink: %s", type(sink).__name__)

    rec = LobRecorder(
        auth, sink,
        series=tuple(args.series),
        horizon_hours=args.horizon_hours,
        rediscover_minutes=args.rediscover_mins,
    )
    try:
        asyncio.run(rec.run_forever())
    except KeyboardInterrupt:
        logging.info("stopped by user")
    finally:
        sink.close()


if __name__ == "__main__":
    main()
