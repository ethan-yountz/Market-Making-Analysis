"""One-off probe: verify series tickers, market/trade/candlestick field names,
and which endpoints need auth. Read-only. Run: python scripts/probe_api.py"""

import json
import sys
import time

from kalshi_mm.api.client import KalshiAPIError, KalshiClient, load_default_auth


def show(label, obj):
    print(f"\n=== {label} ===")
    print(json.dumps(obj, indent=2, default=str)[:2500])


def main():
    auth = load_default_auth()
    print(f"auth loaded: {auth is not None}")
    c = KalshiClient(auth=auth, rate_per_s=4)

    for st in ("KXNBAGAME", "KXNFLGAME", "KXMLBGAME", "KXNHLGAME"):
        try:
            s = c.get_series(st)
            ser = s.get("series", s)
            print(f"series {st}: OK  title={ser.get('title')!r} fee_type={ser.get('fee_type')!r} "
                  f"fee_multiplier={ser.get('fee_multiplier')!r} maker_fee={ser.get('maker_fee_rate')!r}")
        except KalshiAPIError as e:
            print(f"series {st}: FAIL {e.status}")

    # A recently settled NBA market (finals week — recent enough to be on the
    # live endpoints rather than behind the historical cutoff)
    mkts = list(c.paginate("/markets", "markets", {
        "series_ticker": "KXNBAGAME", "status": "settled",
        "min_close_ts": int(time.time()) - 7 * 86400,
        "limit": 5}, max_pages=1))
    print(f"\nsettled NBA markets found in window: {len(mkts)}")
    if not mkts:
        print("no markets found — adjust window/params"); sys.exit(1)
    m = mkts[0]
    show("market fields", m)
    ticker = m["ticker"]

    trades = list(c.paginate("/markets/trades", "trades", {"ticker": ticker, "limit": 5}, max_pages=1))
    show(f"trades for {ticker} (n={len(trades)})", trades[:2])

    open_ts = m.get("open_time") or m.get("open_ts")
    close_ts = m.get("close_time") or m.get("close_ts")
    print(f"\nopen={open_ts} close={close_ts}")
    from datetime import datetime, timezone
    close_ts_unix = (
        datetime.fromisoformat(str(close_ts).replace("Z", "+00:00")).timestamp()
        if isinstance(close_ts, str) else float(close_ts)
    )
    try:
        candles = c.get_candlesticks("KXNBAGAME", ticker,
                                     start_ts=int(close_ts_unix) - 2 * 86400,
                                     end_ts=int(close_ts_unix),
                                     period_interval=60)
        print(f"candles (60m): {len(candles)}")
        if candles:
            show("candlestick fields", candles[len(candles) // 2])
    except KalshiAPIError as e:
        print(f"candlesticks FAIL: {e}")

    try:
        cutoff = c.get_historical_cutoff()
        show("historical cutoff", cutoff)
    except KalshiAPIError as e:
        print(f"historical/cutoff FAIL ({e.status}) — likely needs auth")

    try:
        ev = c.get_event(m["event_ticker"], with_nested_markets=False)
        show("event fields", ev)
    except KalshiAPIError as e:
        print(f"event FAIL: {e.status}")


if __name__ == "__main__":
    main()
