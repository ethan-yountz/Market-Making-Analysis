"""Download historical market data for a sport/season.

    python scripts/01_download.py --sport nba --season 2024-25
    python scripts/01_download.py --sport nba --season 2025-26 --min-volume 1000
    python scripts/01_download.py --sport nba --season 2025-26 --limit 25   # smoke run

Unauthenticated works for all endpoints used here. A full season is on the
order of a few hours (rate limited); the manifest makes it resumable, so it
is safe to Ctrl+C and rerun.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalshi_mm.api.client import KalshiClient, load_default_auth
from kalshi_mm.api.download import download_season

SPORTS = {
    "nba": "KXNBAGAME",
    "nfl": "KXNFLGAME",
    "mlb": "KXMLBGAME",
    "nhl": "KXNHLGAME",
}

# Season windows by market close date (UTC).
SEASONS = {
    ("nba", "2024-25"): ("2024-10-20", "2025-06-25"),
    ("nba", "2025-26"): ("2025-10-20", "2026-06-25"),
    ("nfl", "2024-25"): ("2024-09-01", "2025-02-15"),
    ("nfl", "2025-26"): ("2025-09-01", "2026-02-15"),
    ("nhl", "2024-25"): ("2024-10-01", "2025-06-30"),
    ("nhl", "2025-26"): ("2025-10-01", "2026-06-30"),
    ("mlb", "2025"): ("2025-03-15", "2025-11-10"),
    ("mlb", "2026"): ("2026-03-15", "2026-11-10"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sport", choices=SPORTS, default="nba")
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--base-dir", default="data/raw")
    ap.add_argument("--lookback-hours", type=float, default=48.0,
                    help="hours of 1-min candles to pull before market close")
    ap.add_argument("--min-volume", type=float, default=0.0,
                    help="skip markets with lifetime volume below this")
    ap.add_argument("--limit", type=int, default=None,
                    help="only download the first N markets (smoke test)")
    ap.add_argument("--rate", type=float, default=8.0, help="requests/sec")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    key = (args.sport, args.season)
    if key not in SEASONS:
        sys.exit(f"unknown season {key}; add it to SEASONS in this script")
    lo, hi = SEASONS[key]
    min_close = datetime.fromisoformat(lo).replace(tzinfo=timezone.utc)
    max_close = datetime.fromisoformat(hi).replace(tzinfo=timezone.utc)

    client = KalshiClient(auth=load_default_auth(), rate_per_s=args.rate)

    if args.limit is not None:
        # Smoke mode: monkeypatch discovery trim via min_volume sort isn't
        # needed; simplest is to narrow after discovery inside download_season
        # — handled here by a wrapper that truncates the markets parquet after
        # the first write would be more invasive, so we just warn.
        logging.info("limit=%d: downloading first markets only", args.limit)
        import kalshi_mm.api.download as dl

        orig = dl.discover_markets

        def limited(*a, **kw):
            df = orig(*a, **kw)
            return df.head(args.limit)

        dl.discover_markets = limited

    manifest = download_season(
        client,
        SPORTS[args.sport],
        args.sport,
        args.season,
        min_close,
        max_close,
        base_dir=args.base_dir,
        candle_lookback_hours=args.lookback_hours,
        min_volume=args.min_volume,
    )
    if manifest.empty:
        sys.exit("nothing downloaded")
    print(manifest["status"].value_counts())


if __name__ == "__main__":
    main()
