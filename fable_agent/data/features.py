"""Pure feature functions (no I/O).  Used by both live and replay paths."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff().dropna()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return float(out.iloc[-1]) if len(out) else 50.0


def pct_ret(close: pd.Series, n: int) -> float:
    if len(close) <= n:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-1 - n] - 1) * 100)


def realized_vol(close: pd.Series, n: int = 20) -> float:
    r = np.log(close).diff().dropna().tail(n)
    return float(r.std() * math.sqrt(252)) if len(r) > 2 else 0.3


def flow_z(close: pd.Series, volume: pd.Series, short: int = 5, long: int = 60) -> float:
    """z-score of recent dollar volume vs trailing distribution."""
    dv = (close * volume).dropna()
    if len(dv) < long + short:
        return 0.0
    recent = dv.tail(short).mean()
    base = dv.tail(long + short).head(long)
    sd = base.std()
    return float((recent - base.mean()) / sd) if sd > 0 else 0.0


def rs_phase(stock: pd.Series, spy: pd.Series, n: int = 20) -> tuple[float, float]:
    """Relative strength vs SPY and a discretised 'phase' in [-1, 1].

    Phase buckets (matching the dashboard's coarse +0.3 style values):
        RS > +6%  -> +1.0 (leadership)
        RS > +2%  -> +0.6
        RS > 0    -> +0.3 (mild outperformance)  <- AMZN/MSFT anchors
        RS > -2%  -> -0.3
        RS > -6%  -> -0.6
        else      -> -1.0
    """
    j = stock.index.intersection(spy.index)
    s, b = stock.loc[j], spy.loc[j]
    if len(j) <= n:
        return 0.0, 0.0
    rs = float(((s.iloc[-1] / s.iloc[-1 - n]) / (b.iloc[-1] / b.iloc[-1 - n]) - 1) * 100)
    if rs > 6:
        ph = 1.0
    elif rs > 2:
        ph = 0.6
    elif rs > 0:
        ph = 0.3
    elif rs > -2:
        ph = -0.3
    elif rs > -6:
        ph = -0.6
    else:
        ph = -1.0
    return rs, ph


def straddle_implied_move(atm_iv: float, days: float) -> float:
    """Approx straddle-implied move: 0.8 * sigma * sqrt(T)  (Brenner-Subrahmanyam)."""
    days = max(days, 0.5)
    return float(0.8 * atm_iv * math.sqrt(days / 365.0) * 100)


def implied_move_from_iv_excess(atm_iv: float, base_iv: float, days_to_expiry: float) -> float:
    """Earnings-only implied move from the IV *excess* over the non-event level.

    Front IV^2 * T = base IV^2 * T + sigma_event^2 * (1/252)
    """
    T = max(days_to_expiry, 0.5) / 365.0
    var_event = max(atm_iv**2 * T - base_iv**2 * T, 0.0)
    return float(0.8 * math.sqrt(var_event) * 100)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default
