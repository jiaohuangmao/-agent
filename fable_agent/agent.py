"""
EarningsRadarAgent -- glue.

    schema (seed)  ->  quant core  ->  learned rules  ->  Fable judgement  ->  OUR CALL

The order matters: the LLM sees the core's numbers and the rulebook, it never
sees a blank page.  Its adjustment is bounded, logged, and attributed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from .data import provider as P
from .fable.llm import Fable
from .fable.memory import RuleBook
from .model.core import Coefficients, decide, run_core
from .schema import EarningsSchema, ModelCall

log = logging.getLogger(__name__)
OUT_DIR = Path(__file__).resolve().parents[1] / "out"


@dataclass
class AgentConfig:
    use_llm: bool = True
    llm_model: str = "gpt-5-mini"
    rulebook_name: str = "rulebook"
    coeffs: Coefficients = field(default_factory=Coefficients)


class EarningsRadarAgent:
    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        cfg = self.cfg
        self.rb = RuleBook.load(cfg.rulebook_name)
        self.fable = Fable(model=cfg.llm_model) if cfg.use_llm else Fable(client=False)
        if cfg.use_llm and not self.fable.online:
            log.warning("LLM offline (no OPENAI_API_KEY); running quant core + rules only")
        # apply learned coefficient overrides
        for k, v in self.rb.coefficient_overrides.items():
            if hasattr(self.cfg.coeffs, k):
                setattr(self.cfg.coeffs, k, v)

    # ------------------------------------------------------------------
    def call(self, s: EarningsSchema, days_to_event: Optional[float] = None) -> ModelCall:
        if days_to_event is None:
            days_to_event = max((s.report_date - s.as_of.date()).days, 0)
        c = run_core(s, self.cfg.coeffs, days_to_event=days_to_event)

        # 1) learned rules
        p_rules, fired = self.rb.apply(s, c.p_up, days_to_event)
        c.rules_fired = fired
        c.components["p_up_core"] = c.p_up
        c.components["p_up_after_rules"] = p_rules

        # 2) Fable judgement (bounded)
        adj_pp, rationale = 0.0, None
        if self.fable.online:
            adj_pp, rationale, _ = self.fable.judge(s, c, self.rb, days_to_event)
        p_final = max(0.02, min(0.98, p_rules + adj_pp / 100.0))
        c.fable_adjustment = adj_pp
        c.fable_rationale = rationale
        c.p_up = p_final
        c.direction, c.strength = decide(c.system_lean, p_final, self.cfg.coeffs)
        return c

    def call_live(self, ticker: str) -> tuple[EarningsSchema, ModelCall]:
        s = P.build_live(ticker)
        return s, self.call(s)

    def call_past(self, ticker: str, report_date: date) -> tuple[EarningsSchema, ModelCall]:
        s = P.build_point_in_time(ticker, report_date)
        return s, self.call(s, days_to_event=0)

    # ------------------------------------------------------------------
    def radar_next_days(self, universe: List[str], days: int = 3) -> List[tuple[str, date]]:
        """'ON THE RADAR NEXT 3 DAYS' footer."""
        out = []
        today = date.today()
        for t in universe:
            try:
                d, _ = P.next_report_date(t)
                if 0 <= (d - today).days <= days:
                    out.append((t, d))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[1])

    def dump(self, s: EarningsSchema, c: ModelCall, tag: str = "") -> Path:
        OUT_DIR.mkdir(exist_ok=True)
        p = OUT_DIR / f"{s.ticker}_{s.report_date}{('_' + tag) if tag else ''}.json"
        p.write_text(json.dumps({"schema": s.model_dump(mode="json"), "call": c.model_dump(mode="json")},
                                indent=2, default=str))
        return p
