"""
CLI

  python -m fable_agent.cli call AMZN                     # live call for next print
  python -m fable_agent.cli call AMZN --date 2026-07-30   # what would we have said
  python -m fable_agent.cli replay AMZN MSFT GOOGL --n 8  # score core on past prints
  python -m fable_agent.cli train AMZN MSFT ... --rounds 2
  python -m fable_agent.cli rules                         # show rulebook
  python -m fable_agent.cli intraday NVDA AAPL --interval 300 --once
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from .agent import AgentConfig, EarningsRadarAgent
from .fable.llm import Fable
from .fable.memory import RuleBook
from .render.dashboard import render
from .replay.engine import build_rows, evolve, score

DEFAULT_UNIVERSE = ["AMZN", "MSFT", "AAPL", "GOOGL", "META", "NVDA", "TSLA", "NFLX", "AMD", "AVGO",
                    "ORCL", "CRM", "COIN", "MA", "XOM", "CVX", "WMT", "UNH", "CSCO", "APP"]


def _print_call(s, c):
    print(f"\n{'='*64}\n {s.ticker}  earnings {s.report_date} ({s.report_time})   as-of {s.as_of:%Y-%m-%d %H:%M}")
    print(f"{'='*64}")
    print(f" OUR CALL : {c.direction} {c.strength}     P(up)={c.p_up*100:.0f}%")
    print(f" expected ±{c.expected_move_pct:.1f}%  implied ±{c.implied_move_pct:.1f}%  hist median ±{c.hist_median_abs_move:.1f}%")
    print(f" bins     : " + "  ".join(f"{b:.0f}%" for b in c.bins_pct) + f"   edges {c.bin_edges}")
    print(f" lean {c.system_lean:+.2f} = .5*core({c.core_tau_adj:+.2f}) + .15*flow({c.flow_z_5d:+.2f}) + .2*rs({c.rs_phase:+.1f})")
    o = s.options
    print(f" options  : skew {o.iv_skew_pp:+.1f}pp  P/C vol {o.pc_volume:.2f}  fresh v/OI {o.fresh_positioning_v_oi:.2f}  ATM IV {o.atm_iv*100:.0f}%"
          + ("" if o.expiry else "   [RECONSTRUCTED]"))
    print(f" tilt     : {c.model_tilt_pct:+.2f}%  (skew {c.components['tilt_skew']:+.3f} pc {c.components['tilt_pc']:+.3f} "
          f"fresh {c.components['tilt_fresh']:+.3f} lean {c.components['tilt_lean']:+.3f})")
    if c.rules_fired:
        print(f" rules    : {', '.join(c.rules_fired)}  -> p_up {c.components['p_up_core']:.3f} -> {c.components['p_up_after_rules']:.3f}")
    if c.fable_adjustment is not None and c.fable_rationale:
        print(f" fable    : {c.fable_adjustment:+.1f}pp  {c.fable_rationale}")
    if s.realized_move_pct is not None:
        hit = (s.realized_move_pct > 0) == (c.direction == "BULLISH") if c.direction != "NEUTRAL" else None
        print(f" REALIZED : {s.realized_move_pct:+.2f}%   {'HIT' if hit else ('MISS' if hit is False else 'n/a')}")


def cmd_call(a):
    agent = EarningsRadarAgent(AgentConfig(use_llm=not a.no_llm))
    for t in a.tickers:
        if a.date:
            s, c = agent.call_past(t, date.fromisoformat(a.date))
        else:
            s, c = agent.call_live(t)
        _print_call(s, c)
        radar = agent.radar_next_days(a.universe or DEFAULT_UNIVERSE) if not a.date else None
        png = render(s, c, radar=radar)
        js = agent.dump(s, c)
        print(f" -> {png}\n -> {js}")


def cmd_replay(a):
    rows = build_rows(a.tickers, n_prints=a.n)
    rb = RuleBook.load() if a.with_rules else None
    rep = score(rows, rb)
    print(json.dumps(rep.summary(), indent=2))
    print(json.dumps(rep.by_ticker, indent=2))
    if a.verbose:
        for r in rows:
            print(f"{r.schema.ticker:5} {r.schema.report_date} core p_up={r.call.p_up:.2f} lean={r.call.system_lean:+.2f} "
                  f"{r.call.direction:8} realized={r.realized:+6.2f}% em=±{r.call.expected_move_pct:.1f} {'HIT' if r.direction_hit else 'miss'}")


def cmd_train(a):
    rows = build_rows(a.tickers, n_prints=a.n)
    rb = RuleBook.load()
    fable = Fable(client=False) if a.no_llm else Fable(model=a.model)
    print(f"rows={len(rows)}  llm={'on' if fable.online else 'off'}  rules before={len(rb.active())}")
    before = score(rows, rb)
    rb = evolve(rows, rb, fable, rounds=a.rounds)
    after = score(rows, rb)
    p = rb.save()
    print(f"saved {p}")
    print("before:", json.dumps({k: round(v, 4) for k, v in before.summary().items() if isinstance(v, float)}))
    print("after :", json.dumps({k: round(v, 4) for k, v in after.summary().items() if isinstance(v, float)}))
    for r in rb.rules:
        print(f" [{r.status:11}] {r.id:24} in={r.hit_rate_in()} n={r.support_in}  oos={r.hit_rate_out()} n={r.support_out}  adj={r.adjustment:+.2f}")


def cmd_rules(a):
    rb = RuleBook.load()
    print(rb.as_prompt())
    if a.log:
        print(json.dumps(rb.log[-5:], indent=2, default=str))


def cmd_intraday(a):
    from .intraday.loop import IntradayLoop
    loop = IntradayLoop(a.tickers, interval=a.interval, use_llm=not a.no_llm)
    loop.run(once=a.once, max_iter=a.max_iter)


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="fable_agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("call"); c.add_argument("tickers", nargs="+"); c.add_argument("--date")
    c.add_argument("--no-llm", action="store_true"); c.add_argument("--universe", nargs="*"); c.set_defaults(f=cmd_call)

    r = sub.add_parser("replay"); r.add_argument("tickers", nargs="+"); r.add_argument("--n", type=int, default=8)
    r.add_argument("--with-rules", action="store_true"); r.add_argument("-v", "--verbose", action="store_true"); r.set_defaults(f=cmd_replay)

    t = sub.add_parser("train"); t.add_argument("tickers", nargs="+"); t.add_argument("--n", type=int, default=8)
    t.add_argument("--rounds", type=int, default=2); t.add_argument("--no-llm", action="store_true")
    t.add_argument("--model", default="gpt-5-mini"); t.set_defaults(f=cmd_train)

    ru = sub.add_parser("rules"); ru.add_argument("--log", action="store_true"); ru.set_defaults(f=cmd_rules)

    i = sub.add_parser("intraday"); i.add_argument("tickers", nargs="+"); i.add_argument("--interval", type=int, default=300)
    i.add_argument("--once", action="store_true"); i.add_argument("--max-iter", type=int, default=0)
    i.add_argument("--no-llm", action="store_true"); i.set_defaults(f=cmd_intraday)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
