"""One-shot websocket probe: verify auth, MLB discovery, and channel names.

Connects, subscribes to orderbook_delta + trade for a few active MLB markets,
prints the first messages and message-type counts, then exits. Use this to
confirm the logger will collect data before deploying.

    python scripts/check_ws_connection.py
    python scripts/check_ws_connection.py --series KXMLBGAME --max-markets 5 --seconds 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets

from kalshi_mm.api.client import DEFAULT_BASE_URL, WS_PATH, KalshiClient, load_default_auth
from kalshi_mm.recorder.discover import discover_game_markets


async def probe(series: str, max_markets: int, seconds: float, channels: list[str]) -> None:
    auth = load_default_auth()
    print("auth:", "loaded" if auth else "NONE (unauthenticated)")
    client = KalshiClient(auth=auth, rate_per_s=4)
    markets = discover_game_markets(client, (series,), horizon_hours=48.0)
    tickers = sorted(m["ticker"] for m in markets)[:max_markets]
    print(f"discovered {len(markets)} {series} markets; probing {len(tickers)}: {tickers}")
    if not tickers:
        print("no open markets to probe; try a different --series or wider horizon")
        return

    url = DEFAULT_BASE_URL.replace("https://", "wss://") + WS_PATH
    headers = auth.ws_headers() if auth else {}
    try:
        conn = websockets.connect(url, additional_headers=headers, max_size=2**24)
    except TypeError:
        conn = websockets.connect(url, extra_headers=headers, max_size=2**24)

    counts: Counter[str] = Counter()
    async with conn as ws:
        await ws.send(json.dumps({
            "id": 1, "cmd": "subscribe",
            "params": {"channels": channels, "market_tickers": tickers},
        }))
        print(f"subscribed to {channels} x {len(tickers)} markets; listening {seconds}s ...")
        shown = 0
        try:
            async with asyncio.timeout(seconds):
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    mtype = msg.get("type", "?")
                    counts[mtype] += 1
                    if mtype == "error":
                        print("EXCHANGE ERROR:", msg)
                    elif shown < 8:
                        print(f"  [{mtype}] {json.dumps(msg)[:200]}")
                        shown += 1
        except asyncio.TimeoutError:
            pass
    print("message-type counts:", dict(counts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="KXMLBGAME")
    ap.add_argument("--max-markets", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--channels", nargs="+", default=["orderbook_delta", "trade"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(probe(args.series, args.max_markets, args.seconds, args.channels))


if __name__ == "__main__":
    main()
