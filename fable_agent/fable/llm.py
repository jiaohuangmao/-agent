"""
The "Fable" layer: an LLM that is only ever asked *structured* questions.

Two calls exist:

1. `judge(schema, core_call, rulebook)`  -- inference time.
   Input : compact schema JSON + quant core output + current rulebook.
   Output: JSON {adjustment_pp, rationale, rules_used}.  Bounded to +-8pp.

2. `propose_rules(replay_table, rulebook)`  -- evolution time.
   Input : a table of (schema features, core p_up, realised move, error) over
           many past prints, plus the existing rules and their scores.
   Output: JSON list of *new* Rule candidates (validated by memory.validate_rule)
           and a list of rule ids to retire.

Both prompts explicitly forbid using ticker names/dates in rules, which is the
prompt-side half of the overfit monitor.

If no API key is configured the layer degrades to a no-op (`FableOffline`)
so the quant core still runs end-to-end.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..schema import EarningsSchema, ModelCall
from .memory import Rule, RuleBook, RuleRejected, validate_rule

log = logging.getLogger(__name__)

MAX_ADJ_PP = 8.0


def _client():
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        return None
    key, base = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")
    cfg = Path.home() / ".genspark_llm.yaml"
    if (not key) and cfg.exists():
        y = yaml.safe_load(cfg.read_text()) or {}
        key = y.get("openai", {}).get("api_key")
        base = base or y.get("openai", {}).get("base_url")
    if not key:
        return None
    return OpenAI(api_key=key, base_url=base)


class Fable:
    def __init__(self, model: str = "gpt-5-mini", client=None):
        """client=None -> auto-detect from env; client=False -> force offline."""
        self.model = model
        if client is False:
            self.client = None
        else:
            self.client = client if client is not None else _client()

    @property
    def online(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------
    def _chat(self, system: str, user: str, max_tokens: int = 1200) -> Optional[Dict[str, Any]]:
        if not self.online:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # pragma: no cover
            log.warning("LLM call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    JUDGE_SYSTEM = """You are the judgement layer of an earnings-night positioning model.
You receive a STRUCTURED schema of one upcoming earnings event (options micro-structure,
historical reactions, street consensus, tape) and the output of a quantitative core.
Your job is NOT to predict the company's results. Your job is to read what the *options
market and flow* are already saying and decide whether the core's P(up) should be nudged.

Learned rules (from replaying past prints). Apply them literally when their conditions hold;
they encode what the market has historically rewarded. Do not invent new rules here.
{rules}

Constraints:
- Output JSON only: {{"adjustment_pp": float, "rationale": str, "rules_used": [ids]}}
- adjustment_pp is in percentage points of P(up), in [-{cap}, +{cap}].
- Rationale must cite schema fields (e.g. 'pc_volume 0.46 < 0.7', 'iv_skew +9.7pp'), never news,
  never the ticker's story, never anything not present in the schema.
- If the schema options fields are RECONSTRUCTED proxies (expiry == null), halve any adjustment.
"""

    def judge(self, s: EarningsSchema, core: ModelCall, rb: RuleBook, days_to_event: float = 0.0):
        """Returns (adjustment_pp, rationale, rules_used) -- bounded."""
        sys_p = self.JUDGE_SYSTEM.format(rules=rb.as_prompt(), cap=MAX_ADJ_PP)
        user = json.dumps({
            "schema": s.compact_dict(),
            "core": {k: core.model_dump(mode="json")[k] for k in
                     ("p_up", "system_lean", "expected_move_pct", "implied_move_pct",
                      "hist_median_abs_move", "model_tilt_pct", "components")},
            "days_to_event": days_to_event,
        }, default=str)
        out = self._chat(sys_p, user)
        if not out:
            return 0.0, None, []
        adj = float(out.get("adjustment_pp", 0.0))
        adj = max(-MAX_ADJ_PP, min(MAX_ADJ_PP, adj))
        if s.options.expiry is None:
            adj *= 0.5
        return adj, out.get("rationale"), list(out.get("rules_used", []))

    # ------------------------------------------------------------------
    PROPOSE_SYSTEM = """You are the meta-learning layer of an earnings positioning model.
You are shown a replay table: for many past earnings prints, the pre-print schema features,
the quant core's P(up), and the realised move. You are also shown the current rulebook with
in-sample / out-of-sample hit rates.

Propose at most {k} NEW rules and list rule ids to RETIRE. A rule is:
{{"id": "r_<short>", "conditions": ["<field> <op> <number>", ...], "adjustment": float,
  "rationale": str}}

Hard constraints (violations are discarded automatically):
- 1 to 3 conditions, fields only from: {fields}
- ops: > >= < <= ; thresholds with at most 2 significant digits (e.g. 0.7, 8, 1.2, 12)
- |adjustment| <= 0.08 (fraction of P(up))
- rationale must be a *market-structure* generalisation. It must NOT mention any ticker,
  company, date, quarter or year. Rules that describe one event are memorisation.
- Prefer rules supported by >= 6 replay rows. Do not propose a rule that just restates a
  rule already in the book.

Output JSON: {{"new_rules": [...], "retire": ["id", ...], "notes": str}}
"""

    def propose_rules(self, table: List[Dict[str, Any]], rb: RuleBook, k: int = 3) -> Dict[str, Any]:
        from .memory import ALLOWED_FIELDS
        sys_p = self.PROPOSE_SYSTEM.format(k=k, fields=", ".join(sorted(ALLOWED_FIELDS)))
        user = json.dumps({"replay_table": table, "rulebook": [r.model_dump(mode="json") for r in rb.rules]},
                          default=str)
        out = self._chat(sys_p, user, max_tokens=2000) or {}
        accepted: List[Rule] = []
        for raw in out.get("new_rules", []) or []:
            try:
                r = Rule(**{k2: raw[k2] for k2 in ("id", "conditions", "adjustment", "rationale")})
                validate_rule(r)
                accepted.append(r)
            except (RuleRejected, KeyError, TypeError, ValueError) as e:
                log.info("rule rejected by overfit guard: %s (%s)", raw, e)
        return {"new_rules": accepted, "retire": list(out.get("retire", []) or []), "notes": out.get("notes", "")}
