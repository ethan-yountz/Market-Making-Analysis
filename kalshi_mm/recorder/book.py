"""Local order-book reconstruction from Kalshi websocket messages.

Kalshi sends a full ``orderbook_snapshot`` per market on subscription, then
incremental ``orderbook_delta`` messages. Each delta adjusts the resting
quantity at one (side, price) level by a signed amount. We keep both raw
sides — ``yes`` and ``no`` resting bids — which is lossless: a YES ask at
price p is exactly a NO bid at price 100 - p, so asks are derived, not stored.

Prices on the wire are dollar strings ("0.0100" = 1 cent); we normalise to
integer cents in [1, 99]. Quantities are fixed-point strings and may be
fractional, so they are kept as floats.
"""

from __future__ import annotations

from decimal import Decimal


def price_to_cents(price_dollars: str | float) -> int:
    """'0.0100' -> 1, '0.9900' -> 99. Exact via Decimal (no float drift)."""
    return int((Decimal(str(price_dollars)) * 100).to_integral_value())


class OrderBook:
    """Resting depth for a single market, keyed by integer cents.

    ``yes`` and ``no`` map price_cents -> quantity for each outcome's bids.
    """

    __slots__ = ("ticker", "yes", "no", "last_seq")

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.yes: dict[int, float] = {}
        self.no: dict[int, float] = {}
        self.last_seq: int | None = None

    # ----------------------------------------------------------- mutation

    def apply_snapshot(self, msg: dict) -> None:
        """Reset the book from a full snapshot payload (the inner ``msg``)."""
        self.yes = self._levels(msg.get("yes_dollars_fp"))
        self.no = self._levels(msg.get("no_dollars_fp"))

    def apply_delta(self, msg: dict) -> None:
        """Apply one signed (side, price, delta) change."""
        side = msg["side"]
        book = self.yes if side == "yes" else self.no
        price = price_to_cents(msg["price_dollars"])
        qty = book.get(price, 0.0) + float(msg["delta_fp"])
        if qty > 1e-9:
            book[price] = qty
        else:
            book.pop(price, None)

    @staticmethod
    def _levels(raw: list | None) -> dict[int, float]:
        if not raw:
            return {}
        return {price_to_cents(p): float(q) for p, q in raw}

    # -------------------------------------------------------------- views

    def yes_levels(self) -> list[list]:
        """[[price_cents, qty], ...] sorted by price ascending (JSON-ready)."""
        return [[p, self.yes[p]] for p in sorted(self.yes)]

    def no_levels(self) -> list[list]:
        return [[p, self.no[p]] for p in sorted(self.no)]

    def side_window(self, side: str, depth_cents: int) -> list[list]:
        """Near-touch levels on one side: from the best price down to
        ``best - depth_cents``, as ``[[price_cents, qty], …]`` ascending. Empty
        if the side is empty. This is what the lighter ``topbook`` recording
        mode stores — all the fill model needs around the spread."""
        book = self.yes if side == "yes" else self.no
        if not book:
            return []
        cutoff = max(book) - depth_cents
        return [[p, book[p]] for p in sorted(book) if p >= cutoff]

    @property
    def best_yes_bid(self) -> int | None:
        return max(self.yes) if self.yes else None

    @property
    def best_yes_ask(self) -> int | None:
        """Best YES ask = 100 - (best NO bid)."""
        return 100 - max(self.no) if self.no else None

    @property
    def mid(self) -> float | None:
        b, a = self.best_yes_bid, self.best_yes_ask
        if b is None or a is None:
            return None
        return (b + a) / 2.0

    @property
    def spread(self) -> int | None:
        b, a = self.best_yes_bid, self.best_yes_ask
        if b is None or a is None:
            return None
        return a - b
