"""Overfit guard + leakage tests."""
from datetime import date, datetime

import pytest

from fable_agent.fable.memory import Rule, RuleBook, RuleRejected, validate_rule
from fable_agent.schema import (EarningsSchema, HistoryBlock, OptionsBlock, StreetBlock, TapeBlock)


def _schema(**opt):
    o = dict(spot=100, atm_iv=1.0, implied_move_pct=7, iv_skew_pp=9, pc_volume=0.5, pc_oi=0.8,
             fresh_positioning_v_oi=1.3)
    o.update(opt)
    return EarningsSchema(
        ticker="XYZ", as_of=datetime(2026, 1, 1), report_date=date(2026, 1, 2),
        options=OptionsBlock(**o),
        history=HistoryBlock(last_n=[], hist_median_abs_move=5, hist_mean_signed_move=1, beat_rate=1.0, up_rate=0.6),
        street=StreetBlock(), tape=TapeBlock(ret_5d=1, ret_20d=3, ret_60d=5, rsi_14=45, rs_vs_spy_20d=1, rs_phase=0.3,
                                             flow_z_5d=0.2, dist_from_52w_high_pct=-4, realized_vol_20d=0.3),
        realized_move_pct=12.3,
    )


def test_rule_fires_and_adjusts():
    rb = RuleBook()
    r = Rule(id="r_a", conditions=["options.pc_volume < 0.6", "options.fresh_positioning_v_oi > 1.0"],
             adjustment=0.05, rationale="call-heavy fresh flow", status="active")
    rb.add(r)
    p, fired = rb.apply(_schema(), 0.50)
    assert fired == ["r_a"] and p == pytest.approx(0.55)
    p2, fired2 = rb.apply(_schema(pc_volume=0.9), 0.50)
    assert fired2 == [] and p2 == 0.50


def test_total_adjustment_capped():
    rb = RuleBook()
    for i in range(5):
        rb.add(Rule(id=f"r{i}", conditions=["options.pc_volume < 0.6"], adjustment=0.08, rationale="x", status="active"))
    p, _ = rb.apply(_schema(), 0.50)
    assert p == pytest.approx(0.65)   # cap_total 0.15


@pytest.mark.parametrize("bad,msg", [
    (dict(conditions=["options.pc_volume < 0.4637"], rationale="ok"), "precise"),
    (dict(conditions=["options.pc_volume < 0.5"], rationale="AMZN always rips"), "tickers"),
    (dict(conditions=["options.pc_volume < 0.5"], rationale="in Q2 2026 it worked"), "date"),
    (dict(conditions=["options.pc_volume < 0.5"], adjustment=0.3, rationale="ok"), "adjustment"),
    (dict(conditions=["news.sentiment > 0"], rationale="ok"), "not allowed"),
    (dict(conditions=["a>1", "b>1", "c>1", "d>1"], rationale="ok"), "1-3"),
])
def test_overfit_guard_rejects(bad, msg):
    kw = dict(id="r_bad", adjustment=0.03)
    kw.update(bad)
    with pytest.raises(RuleRejected, match=msg):
        validate_rule(Rule(**kw))


def test_prompt_never_contains_realized_outcome():
    s = _schema()
    d = s.compact_dict()
    assert "realized_move_pct" not in d
    import json
    assert "12.3" not in json.dumps(d)


def test_rulebook_roundtrip(tmp_path, monkeypatch):
    from fable_agent.fable import memory
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    rb = RuleBook()
    rb.add(Rule(id="r_x", conditions=["tape.rsi_14 < 40"], adjustment=0.04, rationale="oversold", status="active"))
    rb.save("t")
    rb2 = RuleBook.load("t")
    assert rb2.active()[0].id == "r_x"
    assert "r_x" in rb2.as_prompt()
