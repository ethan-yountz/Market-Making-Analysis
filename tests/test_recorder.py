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


def test_side_window_keeps_near_touch_only():
    b = OrderBook("MKT")
    b.apply_snapshot({
        "yes_dollars_fp": [["0.5000", "10.00"], ["0.4800", "5.00"], ["0.4400", "9.00"]],
        "no_dollars_fp": [],
    })
    # best yes bid = 50, depth 5 -> keep >= 45: levels 48 and 50, drop 44
    assert b.side_window("yes", 5) == [[48, 5.0], [50, 10.0]]
    assert b.side_window("no", 5) == []


def test_topbook_writes_only_on_touch_move():
    from kalshi_mm.recorder.lob_recorder import LobRecorder

    events: list[dict] = []

    class FakeSink:
        rows_written = 0

        def write_orderbook_event(self, ev):
            events.append(ev)

        def write_trade(self, tr):
            pass

        def flush(self):
            pass

    rec = LobRecorder(auth=None, sink=FakeSink(), mode="topbook", depth_cents=5)
    # snapshot: best yes bid 50, best no bid 48 -> best yes ask 52 -> first row
    rec._handle(json.dumps({
        "type": "orderbook_snapshot", "sid": 1, "seq": 1,
        "msg": {"market_ticker": "M",
                "yes_dollars_fp": [["0.5000", "10.00"], ["0.4900", "5.00"]],
                "no_dollars_fp": [["0.4800", "7.00"]]},
    }))
    assert len(events) == 1 and events[0]["msg_type"] == "book"
    assert events[0]["best_yes_bid"] == 50 and events[0]["best_yes_ask"] == 52

    # deep delta (1c) -> touch unchanged -> no row
    rec._handle(json.dumps({
        "type": "orderbook_delta", "sid": 1, "seq": 2,
        "msg": {"market_ticker": "M", "price_dollars": "0.0100",
                "delta_fp": "100.00", "side": "yes"},
    }))
    assert len(events) == 1

    # size change at an existing in-window level (49) that doesn't move the
    # touch -> still no row
    rec._handle(json.dumps({
        "type": "orderbook_delta", "sid": 1, "seq": 3,
        "msg": {"market_ticker": "M", "price_dollars": "0.4900",
                "delta_fp": "3.00", "side": "yes"},
    }))
    assert len(events) == 1

    # new best yes bid at 51 -> touch moves -> new row, window captured
    rec._handle(json.dumps({
        "type": "orderbook_delta", "sid": 1, "seq": 4,
        "msg": {"market_ticker": "M", "price_dollars": "0.5100",
                "delta_fp": "4.00", "side": "yes"},
    }))
    assert len(events) == 2 and events[1]["best_yes_bid"] == 51
    assert [51, 4.0] in events[1]["yes_levels"]


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
