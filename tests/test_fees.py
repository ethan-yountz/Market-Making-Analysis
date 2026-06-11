import math

from kalshi_mm.sim.fees import FeeSchedule


def test_taker_fee_matches_published_example():
    # Kalshi's canonical example: 100 contracts at 50c, 7% rate
    # 0.07 * 100 * 0.5 * 0.5 = $1.75 -> 175 cents
    fees = FeeSchedule()
    assert fees.taker_fee_c(50, 100) == 175


def test_maker_fee_quarter_of_taker():
    fees = FeeSchedule()
    # 0.0175 * 100 * 0.5 * 0.5 = $0.4375 -> ceil to 44 cents
    assert fees.maker_fee_c(50, 100) == 44


def test_fee_rounds_up():
    fees = FeeSchedule()
    # 0.07 * 1 * 0.5 * 0.5 = $0.0175 -> 2 cents (ceil of 1.75c)
    assert fees.taker_fee_c(50, 1) == 2


def test_fee_vanishes_toward_bounds():
    fees = FeeSchedule()
    assert fees.taker_fee_c(1, 100) < fees.taker_fee_c(50, 100)
    assert fees.maker_fee_c(99, 100) <= math.ceil(0.0175 * 100 * 0.99 * 0.01 * 100)


def test_zero_count_zero_fee():
    fees = FeeSchedule()
    assert fees.taker_fee_c(50, 0) == 0
    assert fees.maker_fee_c(50, 0) == 0
