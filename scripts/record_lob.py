"""Order-book logger entrypoint. Reconstructs the live book and persists
near-touch updates or replayable events to Postgres (production) or SQLite
(local default).

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
from kalshi_mm.recorder.lob_recorder import DEFAULT_SERIES, LobRecorder
from kalshi_mm.recorder.storage import is_hosted_without_db, make_sink


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--series",
        "--sports",
        dest="series",
        nargs="+",
        default=list(DEFAULT_SERIES),
        help="Kalshi series tickers to record (default: KXMLBGAME)",
    )
    ap.add_argument("--db-url", default=os.environ.get("DATABASE_URL"),
                    help="postgres://... ; defaults to $DATABASE_URL or SQLite")
    ap.add_argument("--horizon-hours", type=float, default=36.0)
    ap.add_argument("--rediscover-mins", type=float, default=15.0)
    ap.add_argument("--mode", choices=("topbook", "full"), default="topbook",
                    help="topbook: near-touch book on change (light, default); "
                         "full: full snapshots + every delta (heavy)")
    ap.add_argument("--depth-cents", type=int, default=5,
                    help="topbook: cents around the touch to record per side")
    ap.add_argument("--allow-sqlite", action="store_true",
                    help="permit the local SQLite fallback even on a hosted "
                         "platform (otherwise the logger refuses to start "
                         "without DATABASE_URL, since container SQLite is wiped "
                         "on restart)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    auth = load_default_auth()
    if auth is None:
        logging.warning("no API credentials found - websocket will likely be rejected")

    if is_hosted_without_db(args.db_url) and not args.allow_sqlite:
        logging.error(
            "DATABASE_URL is not set but this looks like a hosted (Railway) deploy. "
            "Refusing to start: the SQLite fallback writes to container-local "
            "storage that is WIPED on every restart, so you'd collect nothing "
            "durable. Fix: add a DATABASE_URL reference to this service "
            "(Variables -> Add Reference -> Postgres -> DATABASE_URL). "
            "Pass --allow-sqlite only if you really want ephemeral local storage."
        )
        raise SystemExit(2)

    sink = make_sink(args.db_url)
    if not args.db_url:
        logging.warning("no DATABASE_URL - using local SQLite at data/lob.sqlite "
                        "(fine locally; ephemeral on hosted platforms)")
    logging.info("mode=%s depth=%dc, storage sink: %s",
                 args.mode, args.depth_cents, type(sink).__name__)

    rec = LobRecorder(
        auth, sink,
        series=tuple(args.series),
        horizon_hours=args.horizon_hours,
        rediscover_minutes=args.rediscover_mins,
        mode=args.mode,
        depth_cents=args.depth_cents,
    )
    try:
        asyncio.run(rec.run_forever())
    except KeyboardInterrupt:
        logging.info("stopped by user")
    finally:
        sink.close()


if __name__ == "__main__":
    main()
