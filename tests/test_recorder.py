"""Order-book reconstruction + storage sink tests, using the real Kalshi wire
format captured from the live feed (dollar-string prices, fixed-point qtys)."""

import json

from kalshi_mm.recorder.book import OrderBook, price_to_cents
from kalshi_mm.recorder.storage import SqliteSink


def test_price_to_cents_exact():
    assert price_to_cents("0.0100") == 1
    assert price_to_cents("0.5000") == 50
    assert price_to_cents("0.9900") == 99


def test_snapshot_loads_both_sides():
    b = OrderBook("MKT")
    b.apply_snapshot({
        "yes_dollars_fp": [["0.4800", "100.00"], ["0.5000", "20.00"]],
        "no_dollars_fp": [["0.4900", "30.00"], ["0.0100", "9999.00"]],
    })
    assert b.yes == {48: 100.0, 50: 20.0}
    assert b.no == {49: 30.0, 1: 9999.0}
    # best yes bid = 50; best yes ask = 100 - best no bid (49) = 51
    assert b.best_yes_bid == 50
    assert b.best_yes_ask == 51
    assert b.mid == 50.5
    assert b.spread == 1


def test_delta_add_update_remove():
    b = OrderBook("MKT")
    b.apply_snapshot({"yes_dollars_fp": [["0.5000", "20.00"]], "no_dollars_fp": []})
    # add a new level
    b.apply_delta({"side": "yes", "price_dollars": "0.4900", "delta_fp": "10.00"})
    assert b.yes[49] == 10.0
    # increase existing
    b.apply_delta({"side": "yes", "price_dollars": "0.5000", "delta_fp": "5.00"})
    assert b.yes[50] == 25.0
    # decrease to zero -> level removed
    b.apply_delta({"side": "yes", "price_dollars": "0.4900", "delta_fp": "-10.00"})
    assert 49 not in b.yes


def test_levels_sorted_and_json_ready():
    b = OrderBook("MKT")
    b.apply_snapshot({
        "yes_dollars_fp": [["0.5000", "20.00"], ["0.4800", "100.00"]],
        "no_dollars_fp": [],
    })
    levels = b.yes_levels()
    assert levels == [[48, 100.0], [50, 20.0]]
    json.dumps(levels)  # must be serialisable


def test_empty_book_has_no_top():
    b = OrderBook("MKT")
    assert b.best_yes_bid is None and b.best_yes_ask is None
    assert b.mid is None and b.spread is None


def test_sqlite_sink_roundtrip(tmp_path):
    sink = SqliteSink(tmp_path / "lob.sqlite")
    sink.write_orderbook_event({
        "recv_ts": 1_700_000_000.0, "exch_ts": "2026-06-24T01:00:00Z",
        "ticker": "MKT", "sid": 1, "seq": 1, "msg_type": "snapshot",
        "yes_levels": [[50, 20.0]], "no_levels": [[49, 30.0]],
        "best_yes_bid": 50, "best_yes_ask": 51, "delta": None,
    })
    sink.write_trade({
        "recv_ts": 1_700_000_001.0, "exch_ts": "2026-06-24T01:00:01Z",
        "ticker": "MKT", "trade_id": "t1", "yes_price": 50, "no_price": 50,
        "count": 12.5, "taker_side": "yes", "raw": {"trade_id": "t1"},
    })
    # duplicate trade_id must be ignored (idempotent re-ingest)
    sink.write_trade({
        "recv_ts": 1_700_000_001.0, "exch_ts": None, "ticker": "MKT",
        "trade_id": "t1", "yes_price": 50, "no_price": 50, "count": 12.5,
        "taker_side": "yes", "raw": {},
    })
    sink.flush()
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "lob.sqlite"))
    assert conn.execute("SELECT COUNT(*) FROM orderbook_events").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
    yl = conn.execute("SELECT yes_levels FROM orderbook_events").fetchone()[0]
    assert json.loads(yl) == [[50, 20.0]]
    conn.close()
    sink.close()


def test_sqlite_sink_delta_stores_null_levels(tmp_path):
    # Delta rows omit the full book (null level arrays) to keep storage small;
    # only the compact delta payload is kept for offline replay.
    sink = SqliteSink(tmp_path / "lob.sqlite")
    sink.write_orderbook_event({
        "recv_ts": 1.0, "exch_ts": None, "ticker": "MKT", "sid": 1, "seq": 2,
        "msg_type": "delta", "yes_levels": None, "no_levels": None,
        "best_yes_bid": 50, "best_yes_ask": 51,
        "delta": {"side": "yes", "price_dollars": "0.5000", "delta_fp": "5.00"},
    })
    sink.close()
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "lob.sqlite"))
    yl, nl, delta = conn.execute(
        "SELECT yes_levels, no_levels, delta FROM orderbook_events").fetchone()
    assert yl is None and nl is None
    assert json.loads(delta)["side"] == "yes"
    conn.close()
