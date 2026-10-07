"""Leakage-safe, direct multi-horizon feature builder.

Design
------
For a forecast *origin* week ``t`` (the last observed week) and horizon ``h = 1..H`` the model
predicts ``log1p(units)`` of week ``j = t + h`` **relative to the series' own recent run-rate**

        target = log1p(y[j]) - lvl8[t],        lvl8[t] = mean(log1p(y[t-7..t]))

so the tree model learns *drivers of deviation* (seasonality, promotions, holidays, momentum)
rather than having to memorise 1,782 different volume scales.

Leakage rules (enforced in ``tests/test_leakage.py``):
  * historical arrays (sales, zero-share, traffic, oil) are only read at columns <= t;
  * seasonal lags read column j-52 <= t-39;
  * the only things allowed to look past the origin are *known-in-advance* covariates of the
    target week: calendar / holidays, and the promotion plan (``promo_tgt``).
"""
from __future__ import annotations

import warnings

import numpy as np

from .weekly_dataset import Panel

GROUPS: dict[str, list[str]] = {
    "AR": [f"d_lag{k}" for k in range(8)] + ["d_m4", "d_m13", "d_m26", "sd8", "sd26", "trend4"],
    "SEAS": ["d_ly", "d_ly_sm", "d_ly2_sm", "yoy8"],
    "PROMO": ["promo_tgt", "promo_rel", "promo_ly", "promo_l4"],
    "AVAIL": ["zs1", "zs4", "zero_streak", "age_w", "active52"],
    "TRAFFIC": ["ltr_d1", "ltr_d4", "ltr_yoy", "ltr_lvl"],
    "MACRO": ["oil_lvl", "oil_chg13", "oil_yoy"],
    "CAL": ["woy", "month", "nat_hol", "reg_hol", "loc_hol", "bridge", "workday", "event_cnt",
            "mothers", "blackfri", "cybermon", "quake", "payday", "xmas", "ny", "ndays"],
    "ATTR": ["store_nbr", "family_code", "type_code", "cluster", "city_code", "state_code",
             "lvl8", "vol52"],
    "HORIZON": ["h"],
}
FEATURES: list[str] = [f for g in GROUPS.values() for f in g]
CAT_FEATURES = ["store_nbr", "family_code", "type_code", "cluster", "city_code", "state_code"]
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}


# ------------------------------------------------------------------------------------------
def _nan(n):
    return np.full(n, np.nan, np.float32)


def nmean(A: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """NaN-aware mean of A[..., lo:hi] along the last axis (rows -> vector). Safe for lo<0."""
    n = A.shape[0]
    lo = max(lo, 0)
    if hi <= lo:
        return _nan(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(A[:, lo:hi], axis=1).astype(np.float32)


def nstd(A: np.ndarray, lo: int, hi: int) -> np.ndarray:
    n = A.shape[0]
    lo = max(lo, 0)
    if hi <= lo:
        return _nan(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanstd(A[:, lo:hi], axis=1).astype(np.float32)


def _vmean(v: np.ndarray, lo: int, hi: int) -> float:
    lo = max(lo, 0)
    return float(np.mean(v[lo:hi])) if hi > lo else np.nan


# ------------------------------------------------------------------------------------------
def origin_features(P: Panel, t: int, promo: np.ndarray, trans_shift: float = 0.0) -> dict:
    """Features that depend on data up to and including week ``t`` only."""
    L = P.L
    st = P.store_idx
    F: dict[str, np.ndarray] = {}
    lvl = nmean(L, t - 7, t + 1)
    for k in range(8):
        F[f"d_lag{k}"] = L[:, t - k] - lvl
    m4, m13, m26 = nmean(L, t - 3, t + 1), nmean(L, t - 12, t + 1), nmean(L, t - 25, t + 1)
    F["d_m4"], F["d_m13"], F["d_m26"] = m4 - lvl, m13 - lvl, m26 - lvl
    F["sd8"], F["sd26"] = nstd(L, t - 7, t + 1), nstd(L, t - 25, t + 1)
    F["trend4"] = m4 - nmean(L, t - 7, t - 3)
    F["yoy8"] = lvl - nmean(L, t - 59, t - 51)

    # availability / distribution proxies
    F["zs1"] = P.Z[:, t]
    F["zs4"] = nmean(P.Z, t - 3, t + 1)
    win = P.Y[:, max(t - 25, 0): t + 1][:, ::-1]
    F["zero_streak"] = np.cumprod(win == 0, axis=1).sum(1).astype(np.float32)
    age = (t - P.first_pos).astype(np.float32)
    age[(P.first_pos < 0) | (P.first_pos > t)] = np.nan
    F["age_w"] = age
    a = P.Y[:, max(t - 51, 0): t + 1]
    valid = (~np.isnan(a)).sum(1)
    F["active52"] = np.where(valid > 0, (a > 0).sum(1) / np.maximum(valid, 1), np.nan).astype(np.float32)

    # traffic (store transactions): history only, optional what-if shift on the last 13 weeks
    tr = np.log1p(P.TR[:, : t + 1]).astype(np.float32)
    if trans_shift:
        tr[:, max(t - 12, 0):] += trans_shift
    t13, t4 = nmean(tr, t - 12, t + 1), nmean(tr, t - 3, t + 1)
    F["ltr_d1"] = (tr[:, t] - t13)[st]
    F["ltr_d4"] = (t4 - t13)[st]
    F["ltr_yoy"] = (t4 - nmean(tr, t - 55, t - 51))[st]
    F["ltr_lvl"] = t13[st]

    # macro (oil): history only
    o4 = _vmean(P.OIL, t - 3, t + 1)
    F["oil_lvl"] = np.full(P.S, o4, np.float32)
    F["oil_chg13"] = np.full(P.S, o4 - _vmean(P.OIL, t - 15, t - 11), np.float32)
    F["oil_yoy"] = np.full(P.S, o4 - _vmean(P.OIL, t - 55, t - 51), np.float32)

    # promo-history summaries at origin
    F["_lp13"] = np.log1p(nmean(promo[:, : t + 1], t - 12, t + 1))
    F["promo_l4"] = np.log1p(nmean(promo[:, : t + 1], t - 3, t + 1))

    # attributes
    m = P.meta
    F["store_nbr"] = m["store_nbr"].values.astype(np.float32)
    F["family_code"] = m["fam_idx"].values.astype(np.float32)
    F["type_code"] = m["type_code"].values.astype(np.float32)
    F["cluster"] = m["cluster"].values.astype(np.float32)
    F["city_code"] = m["city_code"].values.astype(np.float32)
    F["state_code"] = m["state_code"].values.astype(np.float32)
    F["lvl8"] = lvl
    F["vol52"] = nmean(L, t - 51, t + 1)
    F["_lvl"] = lvl
    return F


def target_features(P: Panel, t: int, h: int, promo: np.ndarray, base: dict) -> dict:
    """Features of target week j=t+h that are known in advance (calendar, promo plan) or are
    seasonal lags of the *past* (j-52 <= t-39)."""
    j = t + h
    st = P.store_idx
    L = P.L
    lvl = base["_lvl"]
    F = {"h": np.full(P.S, h, np.float32)}
    F["d_ly"] = L[:, j - 52] - lvl
    F["d_ly_sm"] = nmean(L, j - 53, j - 50) - lvl
    F["d_ly2_sm"] = nmean(L, j - 105, j - 102) - lvl if j - 105 >= 0 else _nan(P.S)
    pj = promo[:, j]
    F["promo_tgt"] = np.log1p(pj)
    F["promo_rel"] = F["promo_tgt"] - base["_lp13"]
    F["promo_ly"] = np.log1p(promo[:, j - 52])
    C = P.CAL
    F["woy"] = np.full(P.S, C["woy"][j], np.float32)
    F["month"] = np.full(P.S, C["month"][j], np.float32)
    F["nat_hol"] = C["nat"][st, j]
    F["reg_hol"] = C["reg"][st, j]
    F["loc_hol"] = C["loc"][st, j]
    F["bridge"] = C["bridge"][st, j]
    F["workday"] = C["workday"][st, j]
    for name, key in (("event_cnt", "event"), ("mothers", "mothers"), ("blackfri", "blackfri"),
                      ("cybermon", "cybermon"), ("quake", "quake"), ("payday", "payday"),
                      ("xmas", "xmas"), ("ny", "ny"), ("ndays", "ndays")):
        F[name] = np.full(P.S, C[key][j], np.float32)
    return F


def build_block(P: Panel, t: int, hs, promo: np.ndarray | None = None, trans_shift: float = 0.0):
    """Feature matrix for origin ``t`` and horizons ``hs`` (rows ordered h-major, series-minor)."""
    promo = P.PROMO if promo is None else promo
    base = origin_features(P, t, promo, trans_shift)
    mats, hh = [], []
    for h in hs:
        F = {**base, **target_features(P, t, h, promo, base)}
        mats.append(np.column_stack([F[c] for c in FEATURES]).astype(np.float32))
        hh.append(np.full(P.S, h))
    X = np.vstack(mats)
    hh = np.concatenate(hh)
    sidx = np.tile(np.arange(P.S), len(hs))
    j = t + hh
    lvl = np.tile(base["_lvl"], len(hs))
    return X, dict(sidx=sidx, h=hh, j=j, lvl=lvl, t=t)


def targets_for(P: Panel, info: dict):
    """(log-target-relative-to-level, raw units, closed-flag). NaN where unobserved."""
    sidx, j = info["sidx"], info["j"]
    obs = j < P.Wh
    jj = np.where(obs, j, 0)
    yl = np.where(obs, P.L[sidx, jj], np.nan)
    yr = np.where(obs, P.Y_raw[sidx, jj], np.nan)
    cl = np.where(obs, P.closed_series[sidx, jj], False)
    return yl, yr, cl


def build_training(P: Panel, o: int, step: int, max_rows: int, rng: np.random.Generator,
                   half_life: float, first_origin: int = 52, gaps: np.ndarray | None = None):
    """Training rows for a model whose information set ends at week ``o`` (inclusive).

    Only rows with target week j <= o are used, so no training target ever lies beyond the fold
    origin (purged, expanding-window).  Origins are sampled every ``step`` weeks, anchored on the
    most recent one, and rows are randomly sub-sampled to cap memory/time.
    """
    origins = list(range(o - 1, first_origin - 1, -step))
    H = P.H
    est = sum(min(H, o - t) for t in origins) * P.S * 0.9
    p = min(1.0, max_rows / max(est, 1))
    Xs, ys, ws, meta = [], [], [], []
    for t in origins:
        hs = [h for h in range(1, H + 1) if t + h <= o]
        X, info = build_block(P, t, hs)
        yl, yr, cl = targets_for(P, info)
        dev = yl - info["lvl"]
        ok = ~np.isnan(dev) & ~cl & ~np.isnan(info["lvl"])
        if gaps is not None:      # reporting-gap weeks that were already known to be over at the origin
            ok &= ~gaps[info["sidx"], np.minimum(info["j"], P.Wh - 1)]
        if p < 1:
            ok &= rng.random(len(ok)) < p
        Xs.append(X[ok])
        ys.append(dev[ok].astype(np.float32))
        ws.append((0.5 ** ((o - info["j"][ok]) / half_life)).astype(np.float32))
        meta.append(np.column_stack([info["sidx"][ok], info["h"][ok], info["j"][ok], info["lvl"][ok]]))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.vstack(meta)
