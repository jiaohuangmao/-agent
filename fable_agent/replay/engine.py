"""
Replay & evolution engine.

    "以这个 schema 作为种子重建过去财报的相同输入 ... 把过去的历史进行一定的校准了以后，
     获得一些对于未来预测的策略规律 ... 观察他有没有强行记住不合理的记忆来约束过拟合"

Procedure
---------
1. For every ticker in the universe, for each of the last N prints, build the
   point-in-time schema (identical structure to live).          -> replay rows
2. Score the quant core on each row (direction hit, Brier, PIT calibration).
3. Split rows chronologically: OLD 70% = in-sample, NEW 30% = out-of-sample.
4. Evolution round:
     a. LLM proposes rules from the in-sample rows only.
     b. Each candidate is scored on in-sample and out-of-sample separately.
     c. Overfit monitor: keep only rules with support >= min_support on both
        splits and oos hit-rate >= in-sample hit-rate - max_gap and oos
        hit-rate > 0.5.
     d. Rules whose sign of usefulness flips vs. previous round -> quarantine.
5. Persist the rulebook.  Repeat rounds (the "training").

The realised outcome is only attached to a row *after* the model has produced
its call for it -- `EarningsSchema.compact_dict()` drops `realized_move_pct`
and the leakage test asserts the LLM prompt never contains that key.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np

from ..data import provider as P
from ..fable.llm import Fable
from ..fable.memory import Rule, RuleBook, RuleRejected
from ..model.core import Coefficients, run_core
from ..schema import EarningsSchema, ModelCall

log = logging.getLogger(__name__)


@dataclass
class ReplayRow:
    schema: EarningsSchema
    call: ModelCall
    realized: float

    @property
    def direction_hit(self) -> bool:
        if self.call.direction == "NEUTRAL":
            return False
        return (self.realized > 0) == (self.call.direction == "BULLISH")

    @property
    def p_up_hit(self) -> bool:
        return (self.realized > 0) == (self.call.p_up > 0.5)

    def brier(self, p_up: Optional[float] = None) -> float:
        p = self.call.p_up if p_up is None else p_up
        y = 1.0 if self.realized > 0 else 0.0
        return (p - y) ** 2

    def pit(self) -> float:
        """Probability-integral-transform of realised move under the model's
        Student-t; uniform on [0,1] iff the distribution is calibrated."""
        from scipy import stats
        c = self.call
        dist = stats.t(c.components["df"], loc=c.model_tilt_pct, scale=0.98 * c.expected_move_pct)
        return float(dist.cdf(self.realized))

    def features(self) -> Dict[str, float]:
        from ..fable.memory import _flatten
        f = _flatten(self.schema, 0.0)
        f = {k: round(v, 3) for k, v in f.items()}
        f.update({"core_p_up": round(self.call.p_up, 3), "system_lean": round(self.call.system_lean, 3),
                  "realized_move_pct": round(self.realized, 2),
                  "up": int(self.realized > 0)})
        return f


@dataclass
class ReplayReport:
    rows: List[ReplayRow]
    n: int = 0
    direction_hit_rate: float = 0.0
    p_up_hit_rate: float = 0.0
    brier: float = 0.0
    brier_baseline: float = 0.25
    abs_move_coverage_1em: float = 0.0     # share of realised |moves| within +-1 expected move
    implied_over_realized: float = 1.0      # variance risk premium check
    pit_ks_p: float = 1.0                   # KS test that PIT ~ U(0,1)
    by_ticker: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if k not in ("rows", "by_ticker")}


def build_rows(universe: List[str], n_prints: int = 8, coeffs: Optional[Coefficients] = None) -> List[ReplayRow]:
    coeffs = coeffs or Coefficients()
    rows: List[ReplayRow] = []
    for t in universe:
        try:
            dates = P.past_report_dates(t, n=n_prints)
        except Exception as e:  # pragma: no cover
            log.warning("skip %s: %s", t, e)
            continue
        for d in dates:
            try:
                s = P.build_point_in_time(t, d)
            except Exception as e:  # pragma: no cover
                log.warning("skip %s %s: %s", t, d, e)
                continue
            if s.realized_move_pct is None or len(s.history.last_n) < 3:
                continue
            call = run_core(s, coeffs)
            rows.append(ReplayRow(schema=s, call=call, realized=s.realized_move_pct))
    rows.sort(key=lambda r: r.schema.report_date)
    return rows


def score(rows: List[ReplayRow], rb: Optional[RuleBook] = None) -> ReplayReport:
    from scipy import stats
    rep = ReplayReport(rows=rows, n=len(rows))
    if not rows:
        return rep
    p_adj = []
    for r in rows:
        p = r.call.p_up
        if rb is not None:
            p, _ = rb.apply(r.schema, p)
        p_adj.append(p)
    ups = np.array([r.realized > 0 for r in rows], dtype=float)
    p_adj = np.array(p_adj)
    rep.direction_hit_rate = float(np.mean([r.direction_hit for r in rows]))
    rep.p_up_hit_rate = float(np.mean((p_adj > 0.5) == (ups > 0.5)))
    rep.brier = float(np.mean((p_adj - ups) ** 2))
    base = ups.mean()
    rep.brier_baseline = float(np.mean((base - ups) ** 2))
    rep.abs_move_coverage_1em = float(np.mean([abs(r.realized) <= r.call.expected_move_pct for r in rows]))
    rep.implied_over_realized = float(np.median([r.call.implied_move_pct for r in rows]) /
                                      max(np.median([abs(r.realized) for r in rows]), 1e-6))
    pits = np.array([r.pit() for r in rows])
    rep.pit_ks_p = float(stats.kstest(pits, "uniform").pvalue)
    for t in sorted({r.schema.ticker for r in rows}):
        rs = [r for r in rows if r.schema.ticker == t]
        rep.by_ticker[t] = {
            "n": len(rs),
            "dir_hit": float(np.mean([r.direction_hit for r in rs])),
            "median_abs_move": float(np.median([abs(r.realized) for r in rs])),
            "median_em": float(np.median([r.call.expected_move_pct for r in rs])),
        }
    return rep


# ----------------------------------------------------------------------------
# evolution
# ----------------------------------------------------------------------------
def _rule_stats(rule: Rule, rows: List[ReplayRow]):
    sup, hit = 0, 0
    for r in rows:
        if rule.fires(r.schema):
            sup += 1
            # the rule "hits" if its adjustment moves p_up toward the truth
            if (rule.adjustment > 0) == (r.realized > 0):
                hit += 1
    return sup, hit


def evolve(rows: List[ReplayRow], rb: RuleBook, fable: Fable, rounds: int = 2,
           min_support: int = 5, max_gap: float = 0.15, k_new: int = 3) -> RuleBook:
    """Walk-forward rule evolution with an overfit monitor."""
    if not rows:
        return rb
    split = int(len(rows) * 0.7)
    ins, oos = rows[:split], rows[split:]
    table = [r.features() for r in ins]

    for rd in range(rounds):
        before = score(ins, rb).brier
        proposal = fable.propose_rules(table, rb, k=k_new) if fable.online else {"new_rules": [], "retire": []}
        if not fable.online and rd == 0:
            proposal["new_rules"] = seed_rules()          # deterministic offline seeds
        added, rejected = [], []
        for rule in proposal["new_rules"]:
            try:
                rb.add(rule)
            except RuleRejected as e:
                rejected.append((rule.id, str(e)))
                continue
            rule.support_in, rule.hits_in = _rule_stats(rule, ins)
            rule.support_out, rule.hits_out = _rule_stats(rule, oos)
            ok = _passes_monitor(rule, min_support, max_gap)
            rule.status = "active" if ok else "quarantined"
            rule.history.append({"round": rd, "ts": datetime.utcnow().isoformat(),
                                 "in": rule.hit_rate_in(), "oos": rule.hit_rate_out(), "status": rule.status})
            added.append(rule.id)
        for rid in proposal.get("retire", []):
            for r in rb.rules:
                if r.id == rid and r.status == "active":
                    r.status = "retired"
        # re-check existing active rules each round (sign flip -> quarantine)
        for r in rb.active():
            si, hi = _rule_stats(r, ins)
            so, ho = _rule_stats(r, oos)
            prev_useful = (r.hit_rate_out() or 0.5) > 0.5
            r.support_in, r.hits_in, r.support_out, r.hits_out = si, hi, so, ho
            now_useful = (r.hit_rate_out() or 0.5) > 0.5
            if prev_useful != now_useful or not _passes_monitor(r, min_support, max_gap):
                r.status = "quarantined"
        after = score(ins, rb).brier
        oos_rep = score(oos, rb)
        rb.log.append({"ts": datetime.utcnow().isoformat(), "round": rd, "added": added, "rejected": rejected,
                       "brier_in_before": before, "brier_in_after": after, "brier_oos": oos_rep.brier,
                       "oos_hit": oos_rep.p_up_hit_rate, "n_active": len(rb.active()),
                       "notes": proposal.get("notes", "")})
        table = [r.features() for r in ins]
    rb.version += 1
    return rb


def _passes_monitor(r: Rule, min_support: int, max_gap: float) -> bool:
    if r.support_in < min_support or r.support_out < max(2, min_support // 2):
        return False
    hi, ho = r.hit_rate_in(), r.hit_rate_out()
    if hi is None or ho is None:
        return False
    if ho < 0.5:
        return False
    return ho >= hi - max_gap


def seed_rules() -> List[Rule]:
    """Deterministic, textbook market-structure rules used when the LLM is
    offline.  They still have to pass the overfit monitor on your data."""
    return [
        Rule(id="r_call_heavy_fresh", conditions=["options.pc_volume < 0.6", "options.fresh_positioning_v_oi > 1.0"],
             adjustment=0.05, rationale="call-heavy flow that exceeds standing OI = new money leaning long"),
        Rule(id="r_put_skew_fear", conditions=["options.iv_skew_pp > 8"],
             adjustment=-0.03, rationale="steep put skew means hedgers pay up for downside; realised skews follow"),
        Rule(id="r_oversold_beater", conditions=["tape.rsi_14 < 40", "history.beat_rate >= 0.9"],
             adjustment=0.04, rationale="oversold habitual beater: low bar into the print"),
        Rule(id="r_extended_into_print", conditions=["tape.ret_20d > 12", "tape.rsi_14 > 70"],
             adjustment=-0.04, rationale="over-extended tape into the event: good news is priced, asymmetry is down"),
        Rule(id="r_cheap_implied", conditions=["derived.implied_over_hist < 0.9"],
             adjustment=0.0, rationale="implied below historical median: vol is cheap, no directional edge (placeholder for sizing)"),
    ]
