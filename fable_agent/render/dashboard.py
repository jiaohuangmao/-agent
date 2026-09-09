"""
Render the 'EARNINGS RADAR' card (1024x640) in the style of the published
AMZN / MSFT dashboards.

Layout (top -> bottom)
    header bar        : EARNINGS RADAR / TICKER / 'earnings Thu Jul 30' + OUR CALL pill
    stat line         : P(up) · expected move · implied · hist median
    7-bucket bar chart: red -> grey -> green with 'model tilt' dashed marker
    left panel        : implied vs last 8 earnings (dots + dashed hist median + implied line)
    right panel       : positioning bars (bullish -> right)
    STREET line       : EPS cons / revenue / beat rate / ATM IV
    system components : core(tau-adj) / flow z / rs_phase
    footer            : ON THE RADAR NEXT 3 DAYS
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from ..schema import EarningsSchema, ModelCall

NAVY = "#14213d"
GREEN_PILL = "#1f8a3a"
RED_PILL = "#b3261e"
GREY_PILL = "#6b7280"
BIN_COLORS = ["#9b1c1c", "#dc4b4b", "#e59a9a", "#8f8f8a", "#7fd4a8", "#2eb872", "#0f7a45"]
GREEN_BAR = "#2eb872"
RED_BAR = "#dc4b4b"


def _pill_color(direction: str) -> str:
    return {"BULLISH": GREEN_PILL, "BEARISH": RED_PILL}.get(direction, GREY_PILL)


def _fmt_edges(edges: List[float]) -> List[str]:
    e = [int(round(x)) for x in edges]
    return [f"< {e[0]}%", f"{e[0]}…{e[1]}%", f"{e[1]}…{e[2]}%", f"±{abs(e[2])}%",
            f"+{e[3]}…+{e[4]}%", f"+{e[4]}…+{e[5]}%", f"> +{e[5]}%"]


def render(s: EarningsSchema, c: ModelCall, radar: Optional[List[Tuple[str, date]]] = None,
           out_path: Optional[Path] = None, brand: str = "Balder's Fable Picks 龙气榜 · Earnings Radar") -> Path:
    fig = plt.figure(figsize=(10.24, 6.4), dpi=100)
    fig.patch.set_facecolor("white")

    # ---------------- header ----------------
    hdr = fig.add_axes([0, 0.865, 1, 0.135])
    hdr.set_facecolor(NAVY)
    hdr.set_xticks([]); hdr.set_yticks([])
    for sp in hdr.spines.values():
        sp.set_visible(False)
    hdr.text(0.11, 0.78, "E A R N I N G S   R A D A R", color="#c7d2fe", fontsize=9, va="center", transform=hdr.transAxes)
    hdr.text(0.11, 0.42, s.ticker, color="white", fontsize=26, fontweight="bold", va="center", transform=hdr.transAxes)
    hdr.text(0.11, 0.13, f"earnings {s.report_date.strftime('%a %b %d')}", color="#e5e7eb", fontsize=10, va="center", transform=hdr.transAxes)
    # cat avatar placeholder (circle)
    from matplotlib.patches import Ellipse
    hdr.add_patch(Ellipse((0.055, 0.5), 0.062, 0.62, transform=hdr.transAxes, color="white"))
    hdr.add_patch(Ellipse((0.055, 0.5), 0.052, 0.52, transform=hdr.transAxes, color="#111827"))
    hdr.text(0.055, 0.5, "B", ha="center", va="center", fontsize=13, color="white", fontweight="bold", transform=hdr.transAxes)
    # OUR CALL pill
    pill = FancyBboxPatch((0.62, 0.12), 0.345, 0.76, boxstyle="round,pad=0.01,rounding_size=0.04",
                          transform=hdr.transAxes, fc=_pill_color(c.direction), ec="white", lw=1.5)
    hdr.add_patch(pill)
    hdr.text(0.645, 0.5, "OUR\nCALL", color="white", fontsize=8, va="center", transform=hdr.transAxes)
    arrow = "▲" if c.direction == "BULLISH" else ("▼" if c.direction == "BEARISH" else "■")
    hdr.text(0.94, 0.58, f"{arrow} {c.direction}", color="white", fontsize=17, ha="right", va="center", transform=hdr.transAxes)
    hdr.text(0.94, 0.22, c.strength, color="#d1fae5", fontsize=8, ha="right", va="center", transform=hdr.transAxes)

    # ---------------- stat line ----------------
    fig.text(0.07, 0.825, f"P(up) ≈ {c.p_up*100:.0f}%    ·    expected move ±{c.expected_move_pct:.1f}%    ·    "
             f"implied ±{c.implied_move_pct:.1f}%" + (f" (exp {s.options.expiry})" if s.options.expiry else " (reconstructed)") +
             f"    ·    hist median ±{c.hist_median_abs_move:.1f}%", fontsize=9, color="#374151")
    fig.text(0.07, 0.80, "probability of earnings-night move", fontsize=9, color="#374151")

    # ---------------- bins ----------------
    axb = fig.add_axes([0.10, 0.60, 0.83, 0.18])
    bins = c.bins_pct
    x = np.arange(7)
    bars = axb.bar(x, bins, color=BIN_COLORS, width=0.72)
    for xi, b in zip(x, bins):
        axb.text(xi, b + 1.0, f"{b:.0f}%", ha="center", va="bottom", fontsize=9)
    labels = _fmt_edges(c.bin_edges)
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=7.5, color="#6b7280")
    axb.set_yticks([]); axb.set_ylim(0, max(bins) * 1.35)
    for sp in axb.spines.values():
        sp.set_visible(False)
    # model tilt marker: position inside the centre bucket proportional to tilt/edge
    a = c.bin_edges[3]
    tilt_x = 3 + 0.36 * float(np.clip(c.model_tilt_pct / a, -1, 1))
    axb.axvline(tilt_x, ymin=0, ymax=0.78, color="#4f46e5", ls="--", lw=1.3)
    axb.text(tilt_x, max(bins) * 1.22, "model tilt", ha="center", fontsize=7.5, color="#4f46e5")

    # ---------------- implied vs last N ----------------
    axl = fig.add_axes([0.07, 0.35, 0.40, 0.16])
    moves = [abs(p.move_pct) for p in reversed(s.history.last_n)]  # oldest -> newest
    axl.scatter(range(len(moves)), moves, color="#2f6fdb", s=45, zorder=3, label="past |move|")
    axl.axhline(c.hist_median_abs_move, color="#2f6fdb", ls="--", lw=1.2)
    axl.axhline(c.implied_move_pct, color="#e6a100", lw=2, label=f"implied ±{c.implied_move_pct:.1f}%")
    axl.set_title("implied vs last 8 earnings", loc="left", fontsize=9.5, color="#374151")
    axl.tick_params(labelsize=7.5, colors="#6b7280"); axl.grid(axis="y", color="#e5e7eb")
    for sp in ("top", "right"):
        axl.spines[sp].set_visible(False)
    axl.legend(fontsize=7, frameon=False, loc="lower right")

    # ---------------- positioning ----------------
    axp = fig.add_axes([0.57, 0.35, 0.36, 0.16])
    axp.set_title("positioning (bullish → right)", loc="left", fontsize=9.5, color="#374151")
    o = s.options
    rows = [
        ("system lean (τ-adj)", c.system_lean, c.system_lean, f"{c.system_lean:+.2f}"),
        ("IV skew (put−call, pp)", -o.iv_skew_pp / 10, o.iv_skew_pp, f"{o.iv_skew_pp:+.1f}pp"),
        ("options P/C volume", (0.7 - o.pc_volume), o.pc_volume, f"{o.pc_volume:.2f}"),
        ("fresh positioning v/OI", (o.fresh_positioning_v_oi - 0.8) * 0.8, o.fresh_positioning_v_oi,
         f"{o.fresh_positioning_v_oi:.2f}"),
    ]
    axp.set_xlim(-1.1, 1.1); axp.set_ylim(-0.6, len(rows) - 0.4)
    axp.axis("off")
    for i, (lab, bull, _, txt) in enumerate(rows):
        y = len(rows) - 1 - i
        v = float(np.clip(bull, -1, 1))
        axp.text(-1.08, y, lab, fontsize=8, va="center", color="#374151")
        axp.add_patch(Rectangle((0.45, y - 0.15), 0.35 * v, 0.3, color=GREEN_BAR if v >= 0 else RED_BAR))
        axp.text(1.08, y, txt, fontsize=8.5, va="center", ha="right", fontweight="bold")

    # ---------------- street ----------------
    st = s.street
    eps = f"EPS cons ${st.eps_consensus:.2f}" if st.eps_consensus else "EPS cons n/a"
    if st.eps_low and st.eps_high:
        eps += f" (${st.eps_low:.2f}–${st.eps_high:.2f}" + (f", n={st.eps_n_analysts})" if st.eps_n_analysts else ")")
    rev = f"revenue ${st.revenue_consensus/1e9:.2f}B" if st.revenue_consensus else "revenue n/a"
    beat = f"beat rate last {len(s.history.last_n)}: {s.history.beat_rate*100:.0f}%" if s.history.beat_rate is not None else "beat rate n/a"
    fig.text(0.07, 0.215, f"STREET:  {eps}    ·    {rev}    ·    {beat}    ·    ATM IV {o.atm_iv*100:.0f}%",
             fontsize=9.5, color="#111827")
    fig.text(0.07, 0.14, f"system components:   core(τ-adj): {c.core_tau_adj:+.2f}    flow z(5d): {c.flow_z_5d:+.2f}    "
             f"rs_phase: {c.rs_phase:+.1f}" + (f"    fable: {c.fable_adjustment:+.1f}pp" if c.fable_adjustment else "") +
             (f"    rules: {', '.join(c.rules_fired)}" if c.rules_fired else ""), fontsize=8.5, color="#374151")

    # ---------------- footer ----------------
    fig.add_artist(plt.Line2D([0, 1], [0.105, 0.105], color="#e5e7eb", lw=1))
    fig.add_artist(Rectangle((0, 0.06), 1, 0.045, color="#f3f4f6", transform=fig.transFigure))
    fig.text(0.07, 0.078, "> ON THE RADAR NEXT 3 DAYS", fontsize=8.5, color="#2f6fdb")
    if radar:
        fig.text(0.34, 0.078, "     ".join(f"{t} · {d.strftime('%a')}" for t, d in radar[:8]), fontsize=8.5, color="#111827")
    fig.text(0.5, 0.025, brand, ha="center", fontsize=8, color="#6b7280")

    out_path = out_path or (Path(__file__).resolve().parents[2] / "out" / f"{s.ticker}_{s.report_date}_radar.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
