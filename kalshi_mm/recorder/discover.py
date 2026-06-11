"""Discovery of active sports game markets to record."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from kalshi_mm.api.client import KalshiClient

log = logging.getLogger(__name__)

GAME_SERIES = ("KXNBAGAME", "KXNFLGAME", "KXMLBGAME", "KXNHLGAME")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def discover_game_markets(
    client: KalshiClient,
    series: tuple[str, ...] = GAME_SERIES,
    horizon_hours: float = 36.0,
) -> list[dict]:
    """All open game markets expected to expire within ``horizon_hours``.

    Imminent + in-progress games — the ones worth recording — without
    subscribing to far-future markets that sit idle for days.
    """
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=horizon_hours)
    out: list[dict] = []
    for st in series:
        try:
            for m in client.get_markets(series_ticker=st, status="open"):
                exp = _parse_iso(m.get("expected_expiration_time")) or _parse_iso(
                    m.get("close_time")
                )
                if exp is not None and exp > horizon:
                    continue
                out.append(
                    {
                        "ticker": m["ticker"],
                        "event_ticker": m.get("event_ticker"),
                        "series_ticker": st,
                        "expected_expiration_time": m.get("expected_expiration_time"),
                        "open_time": m.get("open_time"),
                    }
                )
        except Exception:
            log.exception("discovery failed for %s", st)
    log.info("discovered %d markets across %d series", len(out), len(series))
    return out
