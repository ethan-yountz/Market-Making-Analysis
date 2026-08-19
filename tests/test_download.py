from datetime import datetime, timezone

from kalshi_mm.api.download import discover_markets


class _MarketClient:
    def __init__(self, live: list[dict], historical: list[dict]):
        self.live = live
        self.historical = historical

    def get_markets(self, **_kwargs):
        return iter(self.live)

    def get_historical_markets(self, **_kwargs):
        return iter(self.historical)


def _market(ticker: str, close_time: str) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": ticker.rsplit("-", 1)[0],
        "close_time": close_time,
        "volume_fp": "100",
    }


def test_discovery_enforces_close_time_window_client_side():
    """An endpoint ignoring date params must not contaminate season splits."""
    client = _MarketClient(
        live=[
            _market("TOO-EARLY", "2024-09-30T23:59:59Z"),
            _market("IN-WINDOW", "2024-11-01T00:00:00Z"),
            _market("TOO-LATE", "2025-07-01T00:00:01Z"),
        ],
        historical=[_market("IN-WINDOW", "2024-11-01T00:00:00Z")],
    )

    result = discover_markets(
        client,
        "KXNBAGAME",
        datetime(2024, 10, 1, tzinfo=timezone.utc),
        datetime(2025, 7, 1, tzinfo=timezone.utc),
    )

    assert result["ticker"].tolist() == ["IN-WINDOW"]
