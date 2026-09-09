"""
Anchor tests: the two published dashboards (AMZN 2026-07-30, MSFT 2026-07-29)
must be reproduced by the quant core from their *published inputs*.

If any of these fail, the model has drifted from the reference implementation.
"""
from datetime import date, datetime

import pytest

from fable_agent.model import core as C
from fable_agent.schema import (EarningsSchema, HistoryBlock, OptionsBlock, PastEarnings,
                                StreetBlock, TapeBlock)


def _mk(ticker, implied, hist_median, skew, pc, fresh, flow_z, rs_phase, core_adj_target):
    """Build a schema whose core score reproduces the published core(tau-adj)."""
    hist = [PastEarnings(report_date=date(2025, 1, 1), close_before=100, close_after=105,
                         move_pct=5, abs_move_pct=5, beat=True) for _ in range(8)]
    s = EarningsSchema(
        ticker=ticker, as_of=datetime(2026, 7, 29, 15, 0), report_date=date(2026, 7, 30),
        options=OptionsBlock(spot=100, atm_iv=1.0, implied_move_pct=implied, iv_skew_pp=skew,
                             pc_volume=pc, pc_oi=0.8, fresh_positioning_v_oi=fresh),
        history=HistoryBlock(last_n=hist, hist_median_abs_move=hist_median,
                             hist_mean_signed_move=0.0, beat_rate=1.0, up_rate=0.5),
        street=StreetBlock(eps_consensus=1.0),
        tape=TapeBlock(ret_5d=0, ret_20d=0, ret_60d=0, rsi_14=50, rs_vs_spy_20d=0, rs_phase=rs_phase,
                       flow_z_5d=flow_z, dist_from_52w_high_pct=-5, realized_vol_20d=0.3),
    )
    return s


AMZN = dict(implied=7.3, hist_median=5.9, skew=-1.8, pc=0.46, fresh=1.25, flow_z=-0.33, rs_phase=0.3,
            core=0.76, lean=0.39, em=6.7, p_up=0.55, edges=[-13, -7, -3, 3, 7, 13],
            bins=[5, 11, 12, 35, 15, 15, 7])
MSFT = dict(implied=6.9, hist_median=5.0, skew=9.7, pc=0.74, fresh=0.50, flow_z=0.44, rs_phase=0.3,
            core=0.44, lean=0.34, em=6.1, p_up=0.42, edges=[-12, -6, -3, 3, 6, 12],
            bins=[8, 17, 16, 34, 11, 10, 4])


@pytest.mark.parametrize("a", [AMZN, MSFT], ids=["AMZN", "MSFT"])
def test_expected_move(a):
    assert C.expected_move(a["implied"], a["hist_median"]) == pytest.approx(a["em"], abs=0.06)


@pytest.mark.parametrize("a", [AMZN, MSFT], ids=["AMZN", "MSFT"])
def test_system_lean(a):
    assert C.system_lean(a["core"], a["flow_z"], a["rs_phase"]) == pytest.approx(a["lean"], abs=0.01)


@pytest.mark.parametrize("a", [AMZN, MSFT], ids=["AMZN", "MSFT"])
def test_bin_edges(a):
    assert C.bin_edges(a["em"]) == a["edges"]


@pytest.mark.parametrize("a", [AMZN, MSFT], ids=["AMZN", "MSFT"])
def test_p_up_and_bins(a):
    em = C.expected_move(a["implied"], a["hist_median"])
    lean = C.system_lean(a["core"], a["flow_z"], a["rs_phase"])
    tu = C.tilt_units(a["skew"], a["pc"], a["fresh"], lean)
    dist = C.distribution(em, tu, a["fresh"])
    p_up = 1 - dist.cdf(0)
    assert p_up == pytest.approx(a["p_up"], abs=0.02)
    bins = C.bins_from_dist(dist, C.bin_edges(em))
    # bucket-level tolerance: 3.5 percentage points (rounding + df fit)
    for got, want in zip(bins, a["bins"]):
        assert abs(got - want) <= 3.5, (bins, a["bins"])
    assert sum(bins) == pytest.approx(100.0, abs=1e-6)


def test_decision_labels():
    # published calls: AMZN lean +.39 / P(up) .55 -> BULLISH LEAN
    assert C.decide(0.39, 0.55) == ("BULLISH", "LEAN")
    # MSFT lean +.34 / P(up) .42 -> still BULLISH LEAN (direction from lean)
    assert C.decide(0.34, 0.42) == ("BULLISH", "LEAN")
    assert C.decide(0.02, 0.50) == ("NEUTRAL", "LEAN")
    assert C.decide(0.60, 0.66) == ("BULLISH", "CONVICTION")
    assert C.decide(-0.60, 0.30) == ("BEARISH", "CONVICTION")
    assert C.decide(-0.30, 0.55) == ("BEARISH", "LEAN")


def test_tau_adjust_shrinks():
    assert C.tau_adjust(1.0, 0) == 1.0
    assert C.tau_adjust(1.0, 5) == pytest.approx(0.5)
    assert C.tau_adjust(1.0, 10) == pytest.approx(0.25)
