"""
Intraday agent loop.

    "原理和这个帖子差不多，就是写一个有一定结构的 skill，配合自进化以及每 5 分钟询问一次，
     而不是像财报一样一个季度只询问一次。然后疯狂的 replay 和 training"

Same architecture as the earnings agent, different clock:

    IntradaySchema (seed)  ->  factor core  ->  rule memory  ->  (optional) LLM judge
                              every `interval` seconds, for every ticker

Two bots, as described on the timeline
    * short-term bot  ("短线机器人"): momentum/flow continuation, tight stops,
      posts '✅ 平仓获利 / ❌ 平仓亏损 / 🔵 持仓' style tapes.
    * long-term  bot  ("长线机器人 ... 只会抄底"): only buys dips -- RSI < 40
      style oversold in a name whose 60d relative strength is still healthy,
      or a > -8% single-day gap in a habitual beater (the WMT -10% 'RSI<39
      可以观察' post).

Both bots log every decision to a JSONL journal so that `replay_intraday`
can re-score them and the same rule-evolution machinery can be reused.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel

from ..data import features as F

log = logging.getLogger(__name__)
JOURNAL = Path(__file__).resolve().parents[2] / "out" / "intraday_journal.jsonl"


# ----------------------------------------------------------------------------
# schema seed for the 5-minute clock
# ----------------------------------------------------------------------------
class IntradaySchema(BaseModel):
    ticker: str
    ts: datetime
    last: float
    ret_5m: float
    ret_30m: float
    ret_day: float
    vwap_dev_pct: float
    vol_ratio_30m: float          # last 30m volume vs same-window 20d average
    rsi_14_5m: float
    rsi_14_daily: float
    ret_20d: float
    rs_vs_spy_20d: float
    dist_from_52w_high_pct: float
    gap_pct: float                # today's open vs prior close
    minutes_since_open: float
    vix: Optional[float] = None
    next_earnings_days: Optional[int] = None


class BotSignal(BaseModel):
    bot: str                      # 'short' | 'long'
    ticker: str
    ts: datetime
    action: str                   # 'enter_long' | 'exit' | 'hold' | 'watch' | 'none'
    score: float
    reasons: List[str]
    price: float


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def _intraday_bars(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="5d", interval="5m", prepost=False)
    if df.empty:
        raise RuntimeError(f"no intraday bars for {ticker}")
    return df


def _daily(ticker: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period="1y", interval="1d")


def build_intraday_schema(ticker: str, spy_daily: pd.Series, vix: Optional[float] = None,
                          next_earn_days: Optional[int] = None) -> IntradaySchema:
    b = _intraday_bars(ticker)
    d = _daily(ticker)
    today = b.index[-1].date()
    tb = b[b.index.date == today]
    if tb.empty:
        tb = b.tail(78)
    last = float(b["Close"].iloc[-1])
    ret_5m = float((last / b["Close"].iloc[-2] - 1) * 100) if len(b) > 1 else 0.0
    ret_30m = float((last / b["Close"].iloc[-7] - 1) * 100) if len(b) > 7 else 0.0
    prev_close = float(d["Close"].iloc[-2]) if len(d) > 1 and d.index[-1].date() == today else float(d["Close"].iloc[-1])
    ret_day = (last / prev_close - 1) * 100
    gap = (float(tb["Open"].iloc[0]) / prev_close - 1) * 100
    vwap = float((tb["Close"] * tb["Volume"]).sum() / max(tb["Volume"].sum(), 1))
    # 30m volume vs 20d same-time average (approx: mean 30m volume over 5 days)
    v30 = float(b["Volume"].tail(6).sum())
    v30_avg = float(b["Volume"].rolling(6).sum().dropna().mean()) or 1.0
    dclose = d["Close"]
    spy = spy_daily.loc[spy_daily.index.tz_localize(None).normalize() <= pd.Timestamp(today)] if spy_daily.index.tz is not None else spy_daily
    dclose_n = dclose.copy(); dclose_n.index = dclose_n.index.tz_localize(None).normalize()
    spy_n = spy.copy(); spy_n.index = spy_n.index.tz_localize(None).normalize() if spy_n.index.tz is not None else spy_n.index.normalize()
    rs, _ = F.rs_phase(dclose_n, spy_n)
    return IntradaySchema(
        ticker=ticker, ts=b.index[-1].to_pydatetime(), last=last, ret_5m=ret_5m, ret_30m=ret_30m, ret_day=ret_day,
        vwap_dev_pct=(last / vwap - 1) * 100, vol_ratio_30m=v30 / v30_avg, rsi_14_5m=F.rsi(b["Close"]),
        rsi_14_daily=F.rsi(dclose), ret_20d=F.pct_ret(dclose, 20), rs_vs_spy_20d=rs,
        dist_from_52w_high_pct=(last / float(dclose.tail(252).max()) - 1) * 100, gap_pct=gap,
        minutes_since_open=len(tb) * 5.0, vix=vix, next_earnings_days=next_earn_days,
    )


# ----------------------------------------------------------------------------
# bots (factor cores). Coefficients are the 'skill' -- kept explicit so the
# rule-evolution layer can mutate them.
# ----------------------------------------------------------------------------
@dataclass
class ShortTermBot:
    """Continuation/momentum on the 5-minute clock, event-aware."""
    name: str = "short"
    enter_thresh: float = 0.60
    exit_thresh: float = -0.20
    positions: Dict[str, float] = field(default_factory=dict)   # ticker -> entry

    def score(self, s: IntradaySchema) -> tuple[float, List[str]]:
        r: List[str] = []
        sc = 0.0
        if s.vwap_dev_pct > 0.3:
            sc += 0.25; r.append(f"above VWAP {s.vwap_dev_pct:+.2f}%")
        if s.vol_ratio_30m > 1.5:
            sc += 0.25; r.append(f"volume surge x{s.vol_ratio_30m:.1f}")
        if 0 < s.ret_30m < 2.5:
            sc += 0.20; r.append(f"30m momentum {s.ret_30m:+.2f}%")
        if s.rs_vs_spy_20d > 0:
            sc += 0.15; r.append(f"RS vs SPY {s.rs_vs_spy_20d:+.1f}%")
        if s.rsi_14_5m > 80:
            sc -= 0.30; r.append(f"5m RSI overbought {s.rsi_14_5m:.0f}")
        if s.minutes_since_open < 15:
            sc -= 0.25; r.append("first 15 minutes, noisy")
        if s.next_earnings_days is not None and s.next_earnings_days <= 1:
            sc -= 0.40; r.append("earnings tonight: no naked intraday continuation")
        if s.vix and s.vix > 28:
            sc -= 0.20; r.append(f"VIX {s.vix:.0f} elevated")
        return sc, r

    def decide(self, s: IntradaySchema) -> BotSignal:
        sc, r = self.score(s)
        held = s.ticker in self.positions
        if held:
            pnl = (s.last / self.positions[s.ticker] - 1) * 100
            if sc < self.exit_thresh or pnl < -2.5 or s.vwap_dev_pct < -0.5:
                del self.positions[s.ticker]
                return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="exit", score=sc,
                                 reasons=r + [f"pnl {pnl:+.2f}%"], price=s.last)
            return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="hold", score=sc,
                             reasons=[f"pnl {pnl:+.2f}%"], price=s.last)
        if sc >= self.enter_thresh:
            self.positions[s.ticker] = s.last
            return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="enter_long", score=sc, reasons=r, price=s.last)
        return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="none", score=sc, reasons=r, price=s.last)


@dataclass
class LongTermBot:
    """'只会抄底' -- buys dips in strong names, nothing else."""
    name: str = "long"
    rsi_max: float = 40.0
    positions: Dict[str, float] = field(default_factory=dict)

    def score(self, s: IntradaySchema) -> tuple[float, List[str]]:
        r: List[str] = []
        sc = 0.0
        if s.rsi_14_daily < self.rsi_max:
            sc += 0.40; r.append(f"daily RSI {s.rsi_14_daily:.0f} < {self.rsi_max:.0f}")
        if s.ret_day < -5 or s.gap_pct < -5:
            sc += 0.25; r.append(f"capitulation day {s.ret_day:+.1f}% (gap {s.gap_pct:+.1f}%)")
        if s.rs_vs_spy_20d > -5:
            sc += 0.15; r.append("relative strength intact")
        else:
            sc -= 0.15; r.append("RS broken")
        if s.dist_from_52w_high_pct > -25:
            sc += 0.10
        if s.vol_ratio_30m > 2:
            sc += 0.10; r.append("high volume flush")
        if s.next_earnings_days is not None and s.next_earnings_days <= 2:
            sc -= 0.30; r.append("earnings imminent -> wait for print")
        return sc, r

    def decide(self, s: IntradaySchema) -> BotSignal:
        sc, r = self.score(s)
        if s.ticker in self.positions:
            pnl = (s.last / self.positions[s.ticker] - 1) * 100
            if s.rsi_14_daily > 65 and pnl > 0:
                del self.positions[s.ticker]
                return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="exit", score=sc, reasons=[f"RSI recovered, pnl {pnl:+.2f}%"], price=s.last)
            return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="hold", score=sc, reasons=[f"pnl {pnl:+.2f}%"], price=s.last)
        if sc >= 0.65:
            self.positions[s.ticker] = s.last
            return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="enter_long", score=sc, reasons=r, price=s.last)
        if sc >= 0.40:
            return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="watch", score=sc, reasons=r, price=s.last)
        return BotSignal(bot=self.name, ticker=s.ticker, ts=s.ts, action="none", score=sc, reasons=r, price=s.last)


# ----------------------------------------------------------------------------
# loop
# ----------------------------------------------------------------------------
class IntradayLoop:
    def __init__(self, tickers: List[str], interval: int = 300, use_llm: bool = False):
        self.tickers = tickers
        self.interval = interval
        self.short = ShortTermBot()
        self.long = LongTermBot()
        self.use_llm = use_llm
        self._spy = yf.Ticker("SPY").history(period="1y")["Close"]
        JOURNAL.parent.mkdir(exist_ok=True)

    def _vix(self) -> Optional[float]:
        try:
            return float(yf.Ticker("^VIX").fast_info["last_price"])
        except Exception:
            return None

    def _earn_days(self, t: str) -> Optional[int]:
        try:
            from ..data.provider import next_report_date
            d, _ = next_report_date(t)
            return (d - datetime.now().date()).days
        except Exception:
            return None

    def tick(self) -> List[BotSignal]:
        vix = self._vix()
        out: List[BotSignal] = []
        for t in self.tickers:
            try:
                s = build_intraday_schema(t, self._spy, vix=vix, next_earn_days=self._earn_days(t))
            except Exception as e:
                log.warning("%s: %s", t, e)
                continue
            for bot in (self.short, self.long):
                sig = bot.decide(s)
                out.append(sig)
                with JOURNAL.open("a") as f:
                    f.write(json.dumps({"schema": s.model_dump(mode="json"), "signal": sig.model_dump(mode="json")}, default=str) + "\n")
        return out

    @staticmethod
    def format_tape(sigs: List[BotSignal]) -> str:
        """Balder-style tape: ✅ 平仓获利 / ❌ 平仓亏损 / 🔵 持仓 / 🆕 建仓 / 👀 观察."""
        parts = []
        for s in sigs:
            tag = {"enter_long": "🆕 建仓", "exit": "平仓", "hold": "🔵 持仓", "watch": "👀 观察"}.get(s.action)
            if not tag:
                continue
            if s.action == "exit":
                pnl = next((r for r in s.reasons if r.startswith("pnl")), "")
                tag = ("✅ 平仓获利" if "+" in pnl else "❌ 平仓亏损")
            parts.append(f"[{s.bot}] {tag} ${s.ticker} {s.price:.2f}  ({'; '.join(s.reasons[:3])})")
        return "\n".join(parts) if parts else "(no actionable signals this tick)"

    def run(self, once: bool = False, max_iter: int = 0):
        i = 0
        while True:
            i += 1
            sigs = self.tick()
            print(f"\n[{datetime.now():%H:%M:%S}] tick {i}")
            print(self.format_tape(sigs))
            for s in sigs:
                if s.action == "none":
                    print(f"   · {s.bot:5} {s.ticker:5} score {s.score:+.2f}  {'; '.join(s.reasons[:2])}")
            if once or (max_iter and i >= max_iter):
                break
            time.sleep(self.interval)
