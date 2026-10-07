"""Statistical baselines, cold-start fallback and business-rule layer.

All functions take a forecast origin ``o`` (last observed week index) and return an array of
unit forecasts ``[S, H]`` using **only data at weeks <= o**.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config
from .features import nmean
from .weekly_dataset import Panel


def naive_ma(P: Panel, o: int, weeks: int = 4) -> np.ndarray:
    """Flat forecast = mean units of the last ``weeks`` observed weeks."""
    base = nmean(P.Y, o - weeks + 1, o + 1)
    return np.repeat(base[:, None], P.H, axis=1)


def seasonal_naive_growth(P: Panel, o: int) -> np.ndarray:
    """Seasonal naive (3-week smoothed value one year earlier) x trailing-13-week YoY growth.

    Falls back to the 8-week moving average when no seasonal history exists."""
    Y = P.Y
    cur = np.nansum(np.where(np.isnan(Y[:, o - 12:o + 1]), 0, Y[:, o - 12:o + 1]), axis=1)
    lo = o - 12 - 52
    if lo >= 0:
        prev = np.nansum(np.where(np.isnan(Y[:, lo:o - 51]), 0, Y[:, lo:o - 51]), axis=1)
        valid = (~np.isnan(Y[:, lo:o - 51])).sum(1) >= 10
        g = np.where(valid & (prev > 0), cur / np.maximum(prev, 1e-9), 1.0)
    else:
        g = np.ones(P.S)
    g = np.clip(g, 0.6, 1.6)
    fb = nmean(Y, o - 7, o + 1)
    out = np.zeros((P.S, P.H), np.float32)
    for h in range(1, P.H + 1):
        j = o + h
        w = np.stack([Y[:, j - 53], Y[:, j - 52], Y[:, j - 51]], axis=1)
        wts = np.array([0.25, 0.5, 0.25])
        m = ~np.isnan(w)
        num = np.where(m, w, 0) @ wts
        den = m @ wts
        base = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        f = base * g
        out[:, h - 1] = np.where(np.isnan(f), fb, f)
    return out


def damped_holt(P: Panel, o: int, alpha=0.35, beta=0.05, phi=0.9, window=104) -> np.ndarray:
    """Damped-trend exponential smoothing on log1p units (fixed, conventional parameters),
    vectorised across series.  Statistical, non-seasonal reference model."""
    a = max(0, o - window + 1)
    X = pd.DataFrame(P.L[:, a:o + 1].T).ffill().bfill().T.values
    l = X[:, 0].copy()
    b = np.zeros(P.S)
    for tt in range(1, X.shape[1]):
        lp, bp = l, b
        l = alpha * X[:, tt] + (1 - alpha) * (lp + phi * bp)
        b = beta * (l - lp) + (1 - beta) * phi * bp
    out = np.zeros((P.S, P.H), np.float32)
    for h in range(1, P.H + 1):
        out[:, h - 1] = l + b * sum(phi ** i for i in range(1, h + 1))
    return np.clip(np.expm1(out), 0, None).astype(np.float32)


def age_at(P: Panel, o: int) -> np.ndarray:
    age = (o - P.first_pos).astype(np.float32)
    age[(P.first_pos < 0) | (P.first_pos > o)] = np.nan
    return age


def coldstart_peer(P: Panel, o: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _coldstart_peer(P, o)


def _coldstart_peer(P: Panel, o: int) -> np.ndarray:
    """Peer-group forecast for sparse / newly launched series.

    peer level  = mean units (last 8 wks) of *mature* series (>=52 weeks old) of the same family
                  in stores of the same type;
    traffic adj = store traffic / peer-store traffic (last 4 weeks), clipped to [0.3, 3];
    seasonality = peer-group year-ago ratio for each target week.
    If the series has some own history it is blended in with weight min(age/13, 1)."""
    age = age_at(P, o)
    mature = np.nan_to_num(age, nan=-1) >= 52
    Y = P.Y
    own = nmean(Y, o - 7, o + 1)
    meta = P.meta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = np.nanmean(P.TR[:, o - 3:o + 1], axis=1)
    out = np.full((P.S, P.H), np.nan, np.float32)
    keys = meta["fam_idx"].values * 100 + meta["type_code"].values
    for key in np.unique(keys):
        members = np.where(keys == key)[0]
        peers = members[mature[members]]
        if len(peers) == 0:
            continue
        peer_lvl = np.nanmean(own[peers])
        peer_stores = np.unique(meta["store_idx"].values[peers])
        peer_tr = np.nanmean(tr[peer_stores]) if len(peer_stores) else np.nan
        denom = nmean(Y[peers], o - 59, o - 51)
        denom = np.nanmean(denom)
        for s in members:
            if mature[s]:
                continue
            ratio = tr[meta["store_idx"].values[s]] / peer_tr if peer_tr and not np.isnan(peer_tr) else 1.0
            ratio = 1.0 if np.isnan(ratio) else float(np.clip(ratio, 0.3, 3))
            for h in range(1, P.H + 1):
                j = o + h
                num = np.nanmean(nmean(Y[peers], j - 53, j - 50)) if j - 53 >= 0 else np.nan
                seas = num / denom if denom and not np.isnan(denom) and not np.isnan(num) and denom > 0 else 1.0
                out[s, h - 1] = peer_lvl * ratio * float(np.clip(seas, 0.5, 2.0))
    a = np.nan_to_num(age, nan=0)
    w = np.clip(a / config.COLDSTART_MIN_ACTIVE_WEEKS, 0, 1)
    own_f = np.repeat(own[:, None], P.H, axis=1)
    blend = np.where(np.isnan(own_f), out, w[:, None] * own_f + (1 - w[:, None]) * np.where(np.isnan(out), own_f, out))
    return np.where(mature[:, None], np.nan, blend).astype(np.float32)


def apply_rules(P: Panel, o: int, F: np.ndarray, return_flags: bool = False):
    """Business-rule layer applied identically to every model.

    1. never sold / not yet launched at the origin  -> 0 (no basis for a forecast);
    2. delisted: >= DISCONTINUE_ZERO_WEEKS trailing zero-sales weeks -> 0, *unless* the series sold
       around the same calendar week a year earlier (seasonal products such as school supplies are
       legitimately zero off-season; a first version of this rule without the exception zeroed 311
       forecasts that later had sales and hurt RMSLE);
    3. non-negative and capped at 3x the 52-week maximum weekly sales of the series.
    """
    F = np.array(F, dtype=np.float64)
    never = (P.first_pos < 0) | (P.first_pos > o)
    k = config.DISCONTINUE_ZERO_WEEKS
    tail = P.Y[:, o - k + 1:o + 1]
    stale = (~never) & np.all(tail == 0, axis=1)
    delisted = np.zeros(F.shape, bool)
    for h in range(1, P.H + 1):
        j = o + h
        if j - 54 >= 0:
            w = P.Y[:, j - 54:j - 50]
            seasonal = np.nan_to_num(np.nanmax(np.where(np.isnan(w), -1, w), axis=1), nan=-1) > 0
        else:
            seasonal = np.zeros(P.S, bool)
        delisted[:, h - 1] = stale & ~seasonal
    cap = 3 * np.nanmax(np.where(np.isnan(P.Y[:, max(o - 51, 0):o + 1]), 0, P.Y[:, max(o - 51, 0):o + 1]), axis=1)
    F = np.minimum(np.clip(F, 0, None), cap[:, None])
    F[never[:, None] | delisted] = 0
    if return_flags:
        return F.astype(np.float32), never, delisted
    return F.astype(np.float32)
