"""
Data provider: turns raw market data into an `EarningsSchema`.

Two modes
---------
build_live(ticker)                  -> schema for the *next* earnings event,
                                       using today's real option chain.
build_point_in_time(ticker, date)   -> schema *as it would have looked* the
                                       day before a past earnings date.

The second is the heart of Balder's "以这个 schema 作为种子重建过去财报的相同
输入" (rebuild identical inputs for past prints).  Historical option chains
are not freely available, so the options block for past events is
*reconstructed* from realised-vol dynamics + a per-ticker calibrated event
premium.  This is flagged in `OptionsBlock.expiry is None` so the LLM / replay
engine know those fields are proxies, not observed quotes.

Point-in-time discipline: every feature is computed from bars strictly before
`report_date` (close of T-1).  Tests enforce this.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..schema import (EarningsSchema, HistoryBlock, OptionsBlock, PastEarnings, StreetBlock,
                      TapeBlock)
from . import features as F

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# cached raw fetches
# ----------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _prices(ticker: str, years: int = 5) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna(subset=["Close"])


@lru_cache(maxsize=64)
def _earnings_table(ticker: str, limit: int = 40) -> pd.DataFrame:
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception as e:  # pragma: no cover
        log.warning("earnings dates unavailable for %s: %s", ticker, e)
        return pd.DataFrame(columns=["EPS Estimate", "Reported EPS", "Surprise(%)"])
    ed = ed.copy()
    ed["dt"] = pd.to_datetime(ed.index)
    ed["date"] = ed["dt"].dt.tz_localize(None).dt.normalize()
    ed["time"] = np.where(ed["dt"].dt.hour < 12, "bmo", "amc")
    return ed.reset_index(drop=True)


@lru_cache(maxsize=8)
def _spy() -> pd.Series:
    return _prices("SPY")["Close"]


@lru_cache(maxsize=8)
def _vix() -> pd.Series:
    try:
        return _prices("^VIX")["Close"]
    except Exception:  # pragma: no cover
        return pd.Series(dtype=float)


# ----------------------------------------------------------------------------
# earnings-reaction history
# ----------------------------------------------------------------------------
def earnings_reactions(ticker: str, before: Optional[date] = None, n: int = 8) -> List[PastEarnings]:
    """Close(T-1) -> Close(T+1 for AMC / T for BMO) reactions of past prints."""
    px = _prices(ticker)["Close"]
    ed = _earnings_table(ticker)
    out: List[PastEarnings] = []
    cutoff = pd.Timestamp(before) if before else pd.Timestamp.today().normalize()
    for _, row in ed.iterrows():
        d = row["date"]
        if d >= cutoff or pd.isna(row.get("Reported EPS")):
            continue
        # position of report date in the price index
        pos = px.index.searchsorted(d)
        if pos == 0 or pos >= len(px):
            continue
        if row["time"] == "amc":
            i_before = pos if px.index[pos] == d else pos - 1
            i_after = i_before + 1
        else:  # bmo: reaction is on the report day itself
            i_before = pos - 1
            i_after = pos
        if i_after >= len(px) or i_before < 0:
            continue
        cb, ca = float(px.iloc[i_before]), float(px.iloc[i_after])
        mv = (ca / cb - 1) * 100
        est, act = row.get("EPS Estimate"), row.get("Reported EPS")
        out.append(PastEarnings(
            report_date=d.date(), close_before=cb, close_after=ca, move_pct=mv, abs_move_pct=abs(mv),
            eps_actual=None if pd.isna(act) else float(act),
            eps_estimate=None if pd.isna(est) else float(est),
            beat=None if (pd.isna(act) or pd.isna(est)) else bool(act >= est),
        ))
        if len(out) >= n:
            break
    return out


def history_block(reactions: List[PastEarnings]) -> HistoryBlock:
    if not reactions:
        return HistoryBlock(last_n=[], hist_median_abs_move=5.0, hist_mean_signed_move=0.0,
                            beat_rate=None, up_rate=0.5)
    moves = np.array([r.move_pct for r in reactions])
    beats = [r.beat for r in reactions if r.beat is not None]
    return HistoryBlock(
        last_n=reactions,
        hist_median_abs_move=float(np.median(np.abs(moves))),
        hist_mean_signed_move=float(moves.mean()),
        beat_rate=(sum(beats) / len(beats)) if beats else None,
        up_rate=float((moves > 0).mean()),
    )


# ----------------------------------------------------------------------------
# tape block (point-in-time safe)
# ----------------------------------------------------------------------------
def tape_block(ticker: str, asof: date) -> TapeBlock:
    df = _prices(ticker)
    df = df[df.index <= pd.Timestamp(asof)]
    close, vol = df["Close"], df["Volume"]
    spy = _spy()
    spy = spy[spy.index <= pd.Timestamp(asof)]
    rs, ph = F.rs_phase(close, spy)
    hi52 = float(close.tail(252).max())
    vix = _vix()
    vix = vix[vix.index <= pd.Timestamp(asof)]
    return TapeBlock(
        ret_5d=F.pct_ret(close, 5), ret_20d=F.pct_ret(close, 20), ret_60d=F.pct_ret(close, 60),
        rsi_14=F.rsi(close), rs_vs_spy_20d=rs, rs_phase=ph, flow_z_5d=F.flow_z(close, vol),
        dist_from_52w_high_pct=float((close.iloc[-1] / hi52 - 1) * 100),
        realized_vol_20d=F.realized_vol(close), vix=float(vix.iloc[-1]) if len(vix) else None,
    )


# ----------------------------------------------------------------------------
# options block -- live
# ----------------------------------------------------------------------------
def _pick_expiry(expiries: List[str], report_date: date) -> Optional[str]:
    for e in expiries:
        if date.fromisoformat(e) >= report_date:
            return e
    return expiries[-1] if expiries else None


def _iv_at(df: pd.DataFrame, target_strike: float) -> float:
    d = df.dropna(subset=["impliedVolatility"])
    if d.empty:
        return float("nan")
    i = (d["strike"] - target_strike).abs().idxmin()
    return float(d.loc[i, "impliedVolatility"])


def options_block_live(ticker: str, report_date: date) -> OptionsBlock:
    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["last_price"])
    expiries = list(tk.options)
    exp = _pick_expiry(expiries, report_date)
    if exp is None:
        raise RuntimeError(f"no option expiries for {ticker}")
    oc = tk.option_chain(exp)
    calls, puts = oc.calls, oc.puts
    days = max((date.fromisoformat(exp) - date.today()).days, 1)

    atm_iv = float(np.nanmean([_iv_at(calls, spot), _iv_at(puts, spot)]))
    # 25-delta proxy: strikes ~ +-1 sigma*sqrt(T)*0.67 from spot
    k = spot * atm_iv * math.sqrt(days / 365) * 0.67
    put_iv_25 = _iv_at(puts, spot - k)
    call_iv_25 = _iv_at(calls, spot + k)
    skew_pp = (put_iv_25 - call_iv_25) * 100

    # straddle-implied move: use actual ATM straddle mid if quotes exist
    def _mid(df):
        d = df.copy()
        d["mid"] = np.where((d["bid"] > 0) & (d["ask"] > 0), (d["bid"] + d["ask"]) / 2, d["lastPrice"])
        i = (d["strike"] - spot).abs().idxmin()
        return float(d.loc[i, "mid"]), float(d.loc[i, "strike"])

    cm, ks = _mid(calls)
    pm, _ = _mid(puts)
    straddle = cm + pm
    implied_move = straddle / spot * 100 if straddle > 0 else F.straddle_implied_move(atm_iv, days)
    # straddle price over-estimates the pure earnings move when expiry is far away;
    # bound by BS approximation
    implied_move = min(implied_move, F.straddle_implied_move(atm_iv, days) * 1.15)

    cv, pv = int(np.nansum(calls["volume"])), int(np.nansum(puts["volume"]))
    coi, poi = int(np.nansum(calls["openInterest"])), int(np.nansum(puts["openInterest"]))
    return OptionsBlock(
        expiry=date.fromisoformat(exp), spot=spot, atm_iv=atm_iv, implied_move_pct=implied_move,
        iv_skew_pp=skew_pp, pc_volume=F.safe_div(pv, cv, 1.0), pc_oi=F.safe_div(poi, coi, 1.0),
        fresh_positioning_v_oi=F.safe_div(cv + pv, coi + poi, 0.5),
        call_volume=cv, put_volume=pv, call_oi=coi, put_oi=poi,
    )


# ----------------------------------------------------------------------------
# options block -- reconstructed for a past date (no historical chains)
# ----------------------------------------------------------------------------
def options_block_reconstructed(ticker: str, asof: date, reactions_before: List[PastEarnings],
                                tape: TapeBlock) -> OptionsBlock:
    """Proxy option metrics for a past date.

    * implied move  : median of |moves| of prints *before* asof, scaled by the
                      ratio of current realised vol to its 1y mean (markets
                      price event vol relative to ambient vol).  Options
                      historically over-price earnings moves by ~15-20%
                      (variance risk premium) -> multiply 1.15.
    * skew          : proxied by recent drawdown & momentum. After a sell-off
                      puts get bid (skew up); after a run-up calls get bid.
    * P/C volume    : proxied from 5d return / RSI (mean-reversion of demand).
    * fresh v/OI    : proxied from flow_z (dollar volume surge).
    These are *weak* proxies; their purpose is to let the replay engine run
    end-to-end, not to be treated as observed quotes.
    """
    df = _prices(ticker)
    df = df[df.index <= pd.Timestamp(asof)]
    close = df["Close"]
    rv20 = F.realized_vol(close, 20)
    rv250 = float(np.log(close).diff().dropna().tail(250).std() * math.sqrt(252)) if len(close) > 60 else rv20
    vol_ratio = float(np.clip(rv20 / rv250, 0.6, 1.8)) if rv250 > 0 else 1.0
    base_move = float(np.median([r.abs_move_pct for r in reactions_before])) if reactions_before else 5.0
    implied = base_move * 1.15 * (0.5 + 0.5 * vol_ratio)

    skew = float(np.clip(2.0 - 0.6 * tape.ret_20d + 0.25 * (-tape.dist_from_52w_high_pct), -6, 15))
    pc_vol = float(np.clip(0.75 - 0.01 * (tape.rsi_14 - 50) - 0.01 * tape.ret_5d, 0.3, 1.5))
    fresh = float(np.clip(0.8 + 0.35 * tape.flow_z_5d, 0.3, 2.5))
    spot = float(close.iloc[-1])
    atm_iv = float(math.sqrt(rv20**2 + (implied / 100 / 0.8) ** 2 * 252))  # event var folded into 1d
    return OptionsBlock(expiry=None, spot=spot, atm_iv=atm_iv, implied_move_pct=implied,
                        iv_skew_pp=skew, pc_volume=pc_vol, pc_oi=0.8, fresh_positioning_v_oi=fresh)


# ----------------------------------------------------------------------------
# street block
# ----------------------------------------------------------------------------
def street_block(ticker: str, report_date: date, live: bool) -> StreetBlock:
    ed = _earnings_table(ticker)
    row = ed[ed["date"] == pd.Timestamp(report_date)]
    est = float(row["EPS Estimate"].iloc[0]) if len(row) and not pd.isna(row["EPS Estimate"].iloc[0]) else None
    sb = StreetBlock(eps_consensus=est)
    if live:
        try:
            info = yf.Ticker(ticker).info
            sb.eps_n_analysts = info.get("numberOfAnalystOpinions")
            sb.revenue_consensus = None
        except Exception:  # pragma: no cover
            pass
    if est is not None:
        # yfinance has no low/high; use a typical +-8% dispersion band
        sb.eps_low, sb.eps_high = est * 0.90, est * 1.10
    return sb


# ----------------------------------------------------------------------------
# public builders
# ----------------------------------------------------------------------------
def next_report_date(ticker: str) -> tuple[date, str]:
    ed = _earnings_table(ticker)
    today = pd.Timestamp.today().normalize()
    fut = ed[ed["date"] >= today].sort_values("date")
    if fut.empty:
        raise RuntimeError(f"no upcoming earnings date found for {ticker}")
    r = fut.iloc[0]
    return r["date"].date(), r["time"]


def build_live(ticker: str) -> EarningsSchema:
    rd, rt = next_report_date(ticker)
    asof = date.today()
    reactions = earnings_reactions(ticker, before=rd)
    return EarningsSchema(
        ticker=ticker, as_of=datetime.now(), report_date=rd, report_time=rt,
        options=options_block_live(ticker, rd), history=history_block(reactions),
        street=street_block(ticker, rd, live=True), tape=tape_block(ticker, asof),
    )


def build_point_in_time(ticker: str, report_date: date) -> EarningsSchema:
    """Schema as of the close *before* `report_date`, plus realised outcome."""
    px = _prices(ticker)["Close"]
    ed = _earnings_table(ticker)
    row = ed[ed["date"] == pd.Timestamp(report_date)]
    rt = row["time"].iloc[0] if len(row) else "amc"
    pos = px.index.searchsorted(pd.Timestamp(report_date))
    asof_idx = pos if (rt == "amc" and pos < len(px) and px.index[pos] == pd.Timestamp(report_date)) else pos - 1
    asof = px.index[asof_idx].date()
    reactions_before = earnings_reactions(ticker, before=report_date)
    tape = tape_block(ticker, asof)
    s = EarningsSchema(
        ticker=ticker, as_of=datetime.combine(asof, datetime.min.time().replace(hour=15, minute=55)),
        report_date=report_date, report_time=rt,
        options=options_block_reconstructed(ticker, asof, reactions_before, tape),
        history=history_block(reactions_before), street=street_block(ticker, report_date, live=False),
        tape=tape,
    )
    # realised outcome (kept out of prompts by replay engine until scored)
    this = [r for r in earnings_reactions(ticker, before=report_date + timedelta(days=1), n=1)
            if r.report_date == report_date]
    if this:
        s.realized_move_pct = this[0].move_pct
    return s


def past_report_dates(ticker: str, n: int = 8) -> List[date]:
    return [r.report_date for r in earnings_reactions(ticker, n=n)]
