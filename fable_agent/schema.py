"""
The *schema seed* -- Balder's core idea:

    "没有结构性的输入就不会有结构性的输出。所以我会有一个 schema 作为种子，
     这个 schema 就包含了很多关于这个标的财报的信息，例如期权、订单、历史等等"

Everything the agent knows about an earnings event is forced through this
one Pydantic schema.  The same schema is filled for *today* (live) and for
*each past earnings date* (replay), so the LLM layer and the quant layer
always see identical structure -- which is what makes replay / calibration
meaningful instead of anecdotal.

Field names deliberately mirror the labels on the Earnings Radar dashboard
(system lean, IV skew, options P/C volume, fresh positioning v/OI, ...).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------------
# 1. Options market micro-structure  ("期权")
# ----------------------------------------------------------------------------
class OptionsBlock(BaseModel):
    """Front-expiry option chain summary as of the snapshot timestamp."""

    expiry: Optional[date] = Field(None, description="Expiry used for straddle (first after earnings)")
    spot: float
    atm_iv: float = Field(..., description="ATM implied vol, annualised, e.g. 1.63 for 163%")
    implied_move_pct: float = Field(..., description="Straddle-implied earnings move, % of spot")
    iv_skew_pp: float = Field(
        ...,
        description="25-delta put IV minus call IV, in vol points (pp). "
        "Positive = puts richer = crash hedging demand (bearish).",
    )
    pc_volume: float = Field(..., description="Put volume / call volume, all strikes at the chosen expiry")
    pc_oi: float = Field(..., description="Put OI / call OI")
    fresh_positioning_v_oi: float = Field(
        ...,
        description="Total option volume / total OI. >1 means today's flow exceeds the "
        "entire standing book -> lots of *new* money.",
    )
    call_volume: int = 0
    put_volume: int = 0
    call_oi: int = 0
    put_oi: int = 0


# ----------------------------------------------------------------------------
# 2. Historical earnings reaction  ("历史")
# ----------------------------------------------------------------------------
class PastEarnings(BaseModel):
    report_date: date
    close_before: float
    close_after: float
    move_pct: float = Field(..., description="Signed close-to-close move, %")
    abs_move_pct: float
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    beat: Optional[bool] = None
    implied_move_pct_then: Optional[float] = Field(None, description="What options implied at the time (if known)")


class HistoryBlock(BaseModel):
    last_n: List[PastEarnings]
    hist_median_abs_move: float
    hist_mean_signed_move: float
    beat_rate: Optional[float] = Field(None, description="fraction of last N that beat EPS consensus")
    up_rate: float = Field(..., description="fraction of last N with positive move")


# ----------------------------------------------------------------------------
# 3. Street consensus ("订单" in the broad sense: what the market has ordered)
# ----------------------------------------------------------------------------
class StreetBlock(BaseModel):
    eps_consensus: Optional[float] = None
    eps_low: Optional[float] = None
    eps_high: Optional[float] = None
    eps_n_analysts: Optional[int] = None
    revenue_consensus: Optional[float] = None
    revisions_30d: Optional[float] = Field(None, description="net EPS revision % over 30d, if available")


# ----------------------------------------------------------------------------
# 4. Price / flow context
# ----------------------------------------------------------------------------
class TapeBlock(BaseModel):
    ret_5d: float
    ret_20d: float
    ret_60d: float
    rsi_14: float
    rs_vs_spy_20d: float = Field(..., description="relative strength vs SPY over 20d, %")
    rs_phase: float = Field(..., description="discretised regime in [-1, 1] (see features.rs_phase)")
    flow_z_5d: float = Field(..., description="z-score of 5d dollar-volume vs 60d")
    dist_from_52w_high_pct: float
    realized_vol_20d: float = Field(..., description="annualised")
    vix: Optional[float] = None


# ----------------------------------------------------------------------------
# 5. The seed itself
# ----------------------------------------------------------------------------
class EarningsSchema(BaseModel):
    """One fully-populated observation = one 'row' for the agent."""

    ticker: str
    as_of: datetime = Field(..., description="Snapshot time (must be strictly BEFORE the report)")
    report_date: date
    report_time: Literal["bmo", "amc", "unknown"] = "amc"

    options: OptionsBlock
    history: HistoryBlock
    street: StreetBlock
    tape: TapeBlock

    # Populated only during replay -- what actually happened. Never shown to the LLM
    # before it makes its call (leakage guard enforced in replay.engine).
    realized_move_pct: Optional[float] = None

    def compact_dict(self) -> dict:
        """Compact JSON for the prompt (drops realized outcome)."""
        d = self.model_dump(mode="json")
        d.pop("realized_move_pct", None)
        return d


# ----------------------------------------------------------------------------
# 6. Output contract -- what "OUR CALL" must look like
# ----------------------------------------------------------------------------
Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]
Strength = Literal["LEAN", "CONVICTION"]


class ModelCall(BaseModel):
    ticker: str
    report_date: date
    direction: Direction
    strength: Strength
    p_up: float = Field(..., ge=0, le=1)
    expected_move_pct: float
    implied_move_pct: float
    hist_median_abs_move: float
    model_tilt_pct: float = Field(..., description="location shift of the distribution, % of spot")
    bins_pct: List[float] = Field(..., description="7 bucket probabilities, %")
    bin_edges: List[float] = Field(..., description="6 inner edges (%), symmetric about 0")
    system_lean: float
    core_tau_adj: float
    flow_z_5d: float
    rs_phase: float
    components: dict = Field(default_factory=dict)
    fable_adjustment: Optional[float] = Field(None, description="LLM layer's tilt adjustment to p_up (pp)")
    fable_rationale: Optional[str] = None
    rules_fired: List[str] = Field(default_factory=list)
