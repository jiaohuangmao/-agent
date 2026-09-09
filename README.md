# Fable Earnings Radar — Balder 交易 Agent 复现

Reproduction of the "Earnings Radar" / "Fable Picks" trading agent described by
[@Balder13946731](https://x.com/Balder13946731): a **schema-seeded, options-flow-driven
earnings positioning model** with an LLM judgement layer whose prompt is *meta-learned*
through replay, guarded against over-fitting.

Deep-dive on the trading logic: **[ANALYSIS.md](ANALYSIS.md)**.

```
schema (seed) → quant core → learned rules → Fable (LLM) judge → OUR CALL → dashboard PNG
      ▲                                              │
      └────────── replay past prints ◄──── evolve rules (overfit monitor) ◄──┘
```

## Quick start

```bash
pip install -r requirements.txt

# Reproduce a published call from point-in-time data (no LLM needed)
python -m fable_agent.cli call AMZN --date 2026-07-30 --no-llm
python -m fable_agent.cli call MSFT --date 2026-07-29 --no-llm

# Live call for the next print (real option chain via yfinance; LLM judge if OPENAI_API_KEY set)
python -m fable_agent.cli call ORCL

# Replay the quant core over history and score it
python -m fable_agent.cli replay AMZN MSFT GOOGL META NVDA --n 10 -v

# "疯狂的 replay 和 training": let the LLM propose rules, keep only those that survive OOS
python -m fable_agent.cli train AMZN MSFT AAPL GOOGL META NVDA NFLX AMD ORCL WMT --n 10 --rounds 2
python -m fable_agent.cli rules --log

# 5-minute clock: short-term + long-term ("只会抄底") bots
python -m fable_agent.cli intraday NVDA AAPL WMT --interval 300 --once

pytest -q     # 20 tests: screenshot anchors, overfit guard, leakage
```

Outputs go to `out/` (`*_radar.png` dashboards, `*.json` schema+call dumps, `intraday_journal.jsonl`),
the evolving rulebook lives in `memory/rulebook.json`.

## What is reproduced

| Balder's description | Module |
|---|---|
| schema 作为种子（期权 / 订单 / 历史） | `fable_agent/schema.py` |
| 量化骨架：P(up)、expected move、system lean、7 桶分布、model tilt | `fable_agent/model/core.py` (formulas reverse-engineered from the AMZN/MSFT cards; `tests/test_anchors.py`) |
| 重建过去财报的相同输入 | `fable_agent/data/provider.build_point_in_time` |
| 提示词元学习 / 自我进化 | `fable_agent/fable/memory.py` (rules **are** the prompt), `fable_agent/fable/llm.py` |
| 监控提示词、约束过拟合 | `validate_rule` (no tickers / dates / over-precise thresholds) + walk-forward `_passes_monitor` |
| 每 5 分钟询问一次的短线 / 长线机器人 | `fable_agent/intraday/loop.py` |
| Earnings Radar 卡片 | `fable_agent/render/dashboard.py` |

## Anchors (from the two published cards)

| | AMZN | MSFT | formula |
|---|---|---|---|
| expected move | 6.7 | 6.1 | `0.6·implied + 0.4·hist_median` |
| system lean | +0.39 | +0.34 | `0.5·core + 0.15·flow_z + 0.2·rs_phase` |
| model tilt / EM | +0.128 | −0.243 | `0.23·(−skew/10) + 0.1·(0.7−P/C) + 0.1·(fresh−1) + 0.1·lean` |
| P(up) | 55% | 42% | `1 − T_cdf(0)` Student-t, df ≈ 4 − 1.4·(fresh − 0.8) |
| OUR CALL | BULLISH LEAN | BULLISH LEAN | direction from **lean**, P(up) only upgrades to CONVICTION |

## Honest limitations

* Free data (yfinance) has **no historical option chains**; replay uses reconstructed proxies (flagged `expiry=None`).
  With those proxies the quant core alone scores ~random (Brier ≈ baseline) — the edge in Balder's system
  lives in *real-time* options positioning, which the live path does use.
* LLM layer degrades to no-op without `OPENAI_API_KEY`.
* This is research code, not trading advice.
