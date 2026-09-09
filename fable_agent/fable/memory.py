"""
Self-evolving rule memory  ("配置一些自我进化的机制，进行提示词的元学习").

The LLM is never allowed to free-associate.  Everything it "learns" from
replay must be written down as a *Rule*: an explicit, machine-checkable
condition over schema fields and a bounded adjustment to P(up).

    when  options.iv_skew_pp > 8 and history.beat_rate >= 0.9
    then  p_up += -0.04
    because "heavy put skew before a habitual beater = hedged longs, fade the fear less"

Rules are the *prompt* in "prompt meta-learning": the system prompt shown
to the LLM at inference time is literally the current rule set.  Evolution =
propose / mutate / retire rules based on replay scores.

Overfit guard ("监控提示词，观察他有没有强行记住不合理的记忆")
-----------------------------------------------------------------
* a rule must have fired on >= `min_support` replay events
* a rule may not reference a ticker, a date, or an exact number with > 2
  significant digits (that is memorising, not generalising)
* a rule's out-of-sample hit-rate must be >= in-sample hit-rate - `max_gap`
* the total |adjustment| a rule can apply is capped (`max_abs_adj`)
* rules that flip sign between two evolution rounds are quarantined
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..schema import EarningsSchema

MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory"

ALLOWED_FIELDS = {
    "options.implied_move_pct", "options.iv_skew_pp", "options.pc_volume", "options.pc_oi",
    "options.fresh_positioning_v_oi", "options.atm_iv",
    "history.hist_median_abs_move", "history.hist_mean_signed_move", "history.beat_rate",
    "history.up_rate",
    "tape.ret_5d", "tape.ret_20d", "tape.ret_60d", "tape.rsi_14", "tape.rs_vs_spy_20d",
    "tape.rs_phase", "tape.flow_z_5d", "tape.dist_from_52w_high_pct", "tape.realized_vol_20d",
    "tape.vix",
    "derived.implied_over_hist",   # implied / hist_median
    "derived.days_to_event",
}

_COND_RE = re.compile(r"^\s*([a-z_.0-9]+)\s*(>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)\s*$")


class Rule(BaseModel):
    id: str
    conditions: List[str] = Field(..., description="each 'field op number'; ANDed")
    adjustment: float = Field(..., description="added to p_up when all conditions hold")
    rationale: str
    created: datetime = Field(default_factory=datetime.utcnow)
    support_in: int = 0
    hits_in: int = 0
    support_out: int = 0
    hits_out: int = 0
    status: str = "candidate"   # candidate | active | quarantined | retired
    history: List[Dict[str, Any]] = Field(default_factory=list)

    # -- evaluation -----------------------------------------------------------
    def fires(self, s: EarningsSchema, days_to_event: float = 0.0) -> bool:
        ctx = _flatten(s, days_to_event)
        for cond in self.conditions:
            m = _COND_RE.match(cond)
            if not m:
                return False
            field, op, num = m.group(1), m.group(2), float(m.group(3))
            v = ctx.get(field)
            if v is None:
                return False
            if not _cmp(v, op, num):
                return False
        return True

    def hit_rate_in(self) -> Optional[float]:
        return self.hits_in / self.support_in if self.support_in else None

    def hit_rate_out(self) -> Optional[float]:
        return self.hits_out / self.support_out if self.support_out else None


def _cmp(v: float, op: str, n: float) -> bool:
    return {"<": v < n, "<=": v <= n, ">": v > n, ">=": v >= n, "==": abs(v - n) < 1e-9}[op]


def _flatten(s: EarningsSchema, days_to_event: float) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for blk in ("options", "history", "tape"):
        for k, v in getattr(s, blk).model_dump().items():
            if isinstance(v, (int, float)) and v is not None:
                d[f"{blk}.{k}"] = float(v)
    d["derived.implied_over_hist"] = (
        s.options.implied_move_pct / s.history.hist_median_abs_move if s.history.hist_median_abs_move else 1.0
    )
    d["derived.days_to_event"] = days_to_event
    return d


# ----------------------------------------------------------------------------
# validation = the overfit monitor's static half
# ----------------------------------------------------------------------------
class RuleRejected(ValueError):
    pass


_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_DATE_RE = re.compile(r"20\d\d[-/]\d\d|Q[1-4]\s*20\d\d|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+20\d\d", re.I)


def validate_rule(r: Rule, max_abs_adj: float = 0.08) -> None:
    if not r.conditions or len(r.conditions) > 3:
        raise RuleRejected("rules need 1-3 conditions (more = memorising)")
    for c in r.conditions:
        m = _COND_RE.match(c)
        if not m:
            raise RuleRejected(f"bad condition syntax: {c!r}")
        if m.group(1) not in ALLOWED_FIELDS:
            raise RuleRejected(f"field not allowed: {m.group(1)}")
        num = m.group(3).lstrip("-").replace(".", "")
        if len(num.strip("0")) > 2:
            raise RuleRejected(f"threshold too precise (memorising?): {c!r}")
    if abs(r.adjustment) > max_abs_adj:
        raise RuleRejected(f"|adjustment| > {max_abs_adj}")
    if _DATE_RE.search(r.rationale):
        raise RuleRejected("rationale references a specific date/quarter")
    tickers = [t for t in _TICKER_RE.findall(r.rationale) if t not in {"IV", "OI", "ATM", "EPS", "RSI", "RS", "SPY", "VIX", "AND", "OR", "NOT", "PC"}]
    if tickers:
        raise RuleRejected(f"rationale references tickers {tickers} -> not generalisable")


# ----------------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------------
class RuleBook(BaseModel):
    version: int = 1
    rules: List[Rule] = Field(default_factory=list)
    coefficient_overrides: Dict[str, float] = Field(default_factory=dict)
    log: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def load(cls, name: str = "rulebook") -> "RuleBook":
        p = MEMORY_DIR / f"{name}.json"
        if p.exists():
            return cls.model_validate_json(p.read_text())
        return cls()

    def save(self, name: str = "rulebook") -> Path:
        MEMORY_DIR.mkdir(exist_ok=True, parents=True)
        p = MEMORY_DIR / f"{name}.json"
        p.write_text(self.model_dump_json(indent=2))
        return p

    def active(self) -> List[Rule]:
        return [r for r in self.rules if r.status == "active"]

    def add(self, r: Rule) -> None:
        validate_rule(r)
        if any(x.id == r.id for x in self.rules):
            raise RuleRejected(f"duplicate id {r.id}")
        self.rules.append(r)

    def apply(self, s: EarningsSchema, p_up: float, days_to_event: float = 0.0,
              cap_total: float = 0.15):
        """Return (adjusted_p_up, fired_rule_ids)."""
        adj, fired = 0.0, []
        for r in self.active():
            if r.fires(s, days_to_event):
                adj += r.adjustment
                fired.append(r.id)
        adj = max(-cap_total, min(cap_total, adj))
        return max(0.02, min(0.98, p_up + adj)), fired

    def as_prompt(self) -> str:
        if not self.active():
            return "(no learned rules yet)"
        lines = []
        for r in self.active():
            hr_in, hr_out = r.hit_rate_in(), r.hit_rate_out()
            lines.append(
                f"- [{r.id}] IF {' AND '.join(r.conditions)} THEN p_up {r.adjustment:+.3f}  "
                f"(in-sample {hr_in if hr_in is None else round(hr_in,2)} n={r.support_in}; "
                f"oos {hr_out if hr_out is None else round(hr_out,2)} n={r.support_out})  -- {r.rationale}"
            )
        return "\n".join(lines)
