"""
Quant core of the Earnings Radar.

Every formula here was reverse-engineered from the two published dashboards
(AMZN 2026-07-30, MSFT 2026-07-29).  The published numbers act as *anchor
tests* (see tests/test_anchors.py) so that if you change a coefficient you
immediately know you have diverged from the reference implementation.

Anchors (from the screenshots)
------------------------------
                       AMZN      MSFT
expected move          ±6.7%     ±6.1%
implied move           ±7.3%     ±6.9%
hist median            5.9%      5.0%
P(up)                  55%       42%
system lean (tau-adj)  +0.39     +0.34
IV skew (put-call)     -1.8pp    +9.7pp
options P/C volume     0.46      0.74
fresh positioning v/OI 1.25      0.50
core(tau-adj)          +0.76     +0.44
flow z(5d)             -0.33     +0.44
rs_phase               +0.3      +0.3
bins                   5/11/12/35/15/15/7     8/17/16/34/11/10/4
edges                  ±3, ±7, ±13            ±3, ±6, ±12

Findings
--------
* expected_move  = 0.6*implied + 0.4*hist_median          (6.74 / 6.14 ✓)
* system_lean    = 0.5*core + 0.15*flow_z + 0.2*rs_phase  (0.391 / 0.346 ✓)
* bin edges      = ±3, ±round(EM), ±round(2*EM)           (7/13 , 6/12 ✓)
* distribution   = Student-t, df≈3.4-4.5 (fat tails), scale≈EM
* model tilt (mu/EM) = 0.23*(-skew/10) + 0.10*(0.70-pc_vol)
                       + 0.10*(fresh-1) + 0.10*lean       (0.129 / -0.243 ✓)
  -> P(up) = 1 - T_cdf(0; mu, scale, df)                  (0.55 / 0.41 ✓)

'tau-adj' = time-to-event adjustment: signals measured far from the event are
shrunk toward 0 because positioning can still change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy import stats

from ..schema import EarningsSchema, ModelCall


# ----------------------------------------------------------------------------
# coefficients (kept in one place so the evolution layer can propose changes)
# ----------------------------------------------------------------------------
@dataclass
class Coefficients:
    # expected move blend
    w_implied: float = 0.60
    w_hist: float = 0.40
    # system lean
    w_core: float = 0.50
    w_flow: float = 0.15
    w_rs: float = 0.20
    # tilt (in units of expected-move)
    t_skew: float = 0.23      # per 10pp of (call - put) skew
    t_pc: float = 0.10        # per 1.0 of (0.70 - P/C volume)
    t_fresh: float = 0.10     # per 1.0 of (fresh v/OI - 1)
    t_lean: float = 0.10      # per 1.0 of system lean
    pc_neutral: float = 0.70
    # distribution
    df_base: float = 4.0
    scale_mult: float = 0.98
    # decision thresholds  (direction from system lean, see decide())
    lean_dead_zone: float = 0.10
    conviction_lean: float = 0.50
    conviction_pup_edge: float = 0.10
    # tau adjustment
    tau_halflife_days: float = 5.0


DEFAULT = Coefficients()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def tau_adjust(x: float, days_to_event: float, c: Coefficients = DEFAULT) -> float:
    """Shrink a signal toward zero the further we are from the event.

    At 0 days -> x, at halflife -> 0.5x.  Reflects that positioning read 2
    weeks out is only weakly informative about positioning at the print.
    """
    if days_to_event <= 0:
        return x
    return x * 0.5 ** (days_to_event / c.tau_halflife_days)


def core_score(s: EarningsSchema) -> float:
    """'core' component: fundamentals inertia + street setup, in [-1, 1].

    Ingredients (all observable *before* the print):
      * beat rate of last N prints        (fundamental inertia)
      * up rate of last N prints          (does the stock reward beats?)
      * mean signed historical move       (drift)
      * consensus dispersion              (wide range -> uncertainty -> shrink)
    """
    h = s.history
    beat = 2 * (h.beat_rate if h.beat_rate is not None else 0.5) - 1   # [-1,1]
    up = 2 * h.up_rate - 1                                             # [-1,1]
    drift = np.tanh(h.hist_mean_signed_move / 5.0)                     # [-1,1]
    # beat-rate dominates: 8/8 beats alone gives +0.8 (AMZN published core +0.76
    # with 100% beat rate but only 3/8 up-reactions)
    raw = 0.80 * beat + 0.25 * up + 0.15 * drift
    # dispersion penalty
    st = s.street
    if st.eps_consensus and st.eps_low is not None and st.eps_high is not None and st.eps_consensus != 0:
        disp = (st.eps_high - st.eps_low) / abs(st.eps_consensus)
        raw *= float(np.clip(1.15 - disp, 0.6, 1.0))
    return float(np.clip(raw, -1, 1))


def system_lean(core_adj: float, flow_z: float, rs_phase: float, c: Coefficients = DEFAULT) -> float:
    return c.w_core * core_adj + c.w_flow * flow_z + c.w_rs * rs_phase


def expected_move(implied: float, hist_median: float, c: Coefficients = DEFAULT) -> float:
    return c.w_implied * implied + c.w_hist * hist_median


def bin_edges(em: float) -> List[float]:
    a = 3.0
    b = float(max(round(em), a + 1))
    d = float(max(round(2 * em), b + 1))
    return [-d, -b, -a, a, b, d]


def tilt_units(skew_pp: float, pc_vol: float, fresh: float, lean: float, c: Coefficients = DEFAULT) -> float:
    """Location shift of the distribution, in units of expected move."""
    return (
        c.t_skew * (-skew_pp / 10.0)
        + c.t_pc * (c.pc_neutral - pc_vol)
        + c.t_fresh * (fresh - 1.0)
        + c.t_lean * lean
    )


def distribution(em: float, tilt_u: float, fresh: float, c: Coefficients = DEFAULT):
    """Student-t with location = tilt*EM, scale ~ EM, df lower when flow is frantic."""
    mu = tilt_u * em
    scale = c.scale_mult * em
    # more fresh money -> fatter tails (AMZN fresh 1.25 -> df 3.4 ; MSFT 0.5 -> df 4.5)
    df = float(np.clip(c.df_base - 1.4 * (fresh - 0.8), 2.5, 8.0))
    return stats.t(df, loc=mu, scale=scale)


def bins_from_dist(dist, edges: Sequence[float]) -> List[float]:
    cdf = dist.cdf(np.asarray(edges))
    cdf = np.concatenate([[0.0], cdf, [1.0]])
    return [float(x) for x in np.diff(cdf) * 100.0]


def decide(lean: float, p_up: float, c: Coefficients = DEFAULT):
    """'OUR CALL'.

    Anchor: MSFT published P(up)=42% yet the call was **BULLISH LEAN** with
    system lean +0.34.  So direction is driven by the *system lean*
    (fundamental inertia + flow + relative strength), not by the
    options-implied P(up).  P(up) only upgrades LEAN -> CONVICTION when
    the options distribution agrees with the lean.
    """
    if abs(lean) < c.lean_dead_zone:
        return "NEUTRAL", "LEAN"
    direction = "BULLISH" if lean > 0 else "BEARISH"
    agrees = (p_up > 0.5) if lean > 0 else (p_up < 0.5)
    if abs(lean) >= c.conviction_lean and agrees and abs(p_up - 0.5) >= c.conviction_pup_edge:
        return direction, "CONVICTION"
    return direction, "LEAN"


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------
def run_core(s: EarningsSchema, c: Coefficients = DEFAULT, days_to_event: float = 0.0) -> ModelCall:
    o, t, h = s.options, s.tape, s.history

    core_raw = core_score(s)
    core_adj = tau_adjust(core_raw, days_to_event, c)
    lean = system_lean(core_adj, t.flow_z_5d, t.rs_phase, c)

    em = expected_move(o.implied_move_pct, h.hist_median_abs_move, c)
    edges = bin_edges(em)
    tu = tilt_units(o.iv_skew_pp, o.pc_volume, o.fresh_positioning_v_oi, lean, c)
    dist = distribution(em, tu, o.fresh_positioning_v_oi, c)
    df = float(dist.args[0])
    p_up = float(1.0 - dist.cdf(0.0))
    bins = bins_from_dist(dist, edges)
    direction, strength = decide(lean, p_up, c)

    return ModelCall(
        ticker=s.ticker,
        report_date=s.report_date,
        direction=direction,
        strength=strength,
        p_up=p_up,
        expected_move_pct=em,
        implied_move_pct=o.implied_move_pct,
        hist_median_abs_move=h.hist_median_abs_move,
        model_tilt_pct=tu * em,
        bins_pct=bins,
        bin_edges=edges,
        system_lean=lean,
        core_tau_adj=core_adj,
        flow_z_5d=t.flow_z_5d,
        rs_phase=t.rs_phase,
        components={
            "core_raw": core_raw,
            "tilt_units": tu,
            "tilt_skew": c.t_skew * (-o.iv_skew_pp / 10.0),
            "tilt_pc": c.t_pc * (c.pc_neutral - o.pc_volume),
            "tilt_fresh": c.t_fresh * (o.fresh_positioning_v_oi - 1.0),
            "tilt_lean": c.t_lean * lean,
            "df": df,
        },
    )
