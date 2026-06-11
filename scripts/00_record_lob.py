"""Run the websocket LOB recorder. Leave this running in a console.

    python scripts/00_record_lob.py
    python scripts/00_record_lob.py --sports KXMLBGAME KXNBAGAME --horizon-hours 24

Credentials (if required by the websocket endpoint): put your key id in
secrets/kalshi_key_id.txt and the RSA private key in
secrets/kalshi_private_key.pem, or set KALSHI_API_KEY_ID +
KALSHI_PRIVATE_KEY_PATH.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.api.client import load_default_auth
from kalshi_mm.recorder.discover import GAME_SERIES
from kalshi_mm.recorder.ws_recorder import WsRecorder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sports", nargs="+", default=list(GAME_SERIES))
    ap.add_argument("--base-dir", default="data/lob")
    ap.add_argument("--horizon-hours", type=float, default=36.0)
    ap.add_argument("--rediscover-mins", type=float, default=30.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("recorder.log", encoding="utf-8"),
        ],
    )

    auth = load_default_auth()
    if auth is None:
        logging.warning(
            "no API credentials found - attempting unauthenticated websocket. "
            "If the connection is rejected, generate an API key on kalshi.com "
            "(Settings -> API) and place it under secrets/."
        )
    rec = WsRecorder(
        auth,
        base_dir=args.base_dir,
        series=tuple(args.sports),
        horizon_hours=args.horizon_hours,
        rediscover_minutes=args.rediscover_mins,
    )
    try:
        asyncio.run(rec.run_forever())
    except KeyboardInterrupt:
        logging.info("stopped by user")
    finally:
        rec.writer.close()


if __name__ == "__main__":
    main()
