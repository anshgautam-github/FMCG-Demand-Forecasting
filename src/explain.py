"""Explainability without external libraries (SHAP is not required).

1. ``group_permutation_importance`` - global: how much does out-of-sample error rise when one
   *driver group* (promo, calendar, traffic, ...) is shuffled across rows of a held-out fold?
2. ``group_shapley`` - local: **exact Shapley values over driver groups** for individual forecasts
   (interventional / background-sampling value function).  With 8 groups this needs 2^8 = 256
   coalitions, which is cheap, and the attributions sum exactly to
   ``prediction - mean background prediction`` (efficiency axiom, verified in tests).

Because the model predicts the log deviation from the series' own recent run-rate, the final
forecast is read as

        forecast = exp(run-rate) x exp(base + sum of driver effects) - 1
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from . import metrics
from .features import FEATURES, GROUPS, GROUP_OF

# Drivers shown to the business.  HORIZON is merged into the recent-momentum group.
EXPLAIN_GROUPS: dict[str, list[str]] = {
    "Recent momentum (lags/trend)": GROUPS["AR"] + GROUPS["HORIZON"],
    "Seasonality (year-ago, YoY)": GROUPS["SEAS"],
    "Promotions": GROUPS["PROMO"],
    "Availability (zero-sales / age proxy)": GROUPS["AVAIL"],
    "Store traffic (transactions)": GROUPS["TRAFFIC"],
    "Macro (oil price)": GROUPS["MACRO"],
    "Calendar & holidays": GROUPS["CAL"],
    "Store/family identity & scale": GROUPS["ATTR"],
}


def group_permutation_importance(gbm, X, lvl, y_raw, eligible, seed: int = 0) -> pd.DataFrame:
    """Increase in WMAPE / RMSLE when a driver group is shuffled (rows permuted jointly)."""
    rng = np.random.default_rng(seed)
    ok = eligible & ~np.isnan(lvl)

    def score(Xm):
        f = np.clip(np.expm1(lvl[ok] + gbm.predict_dev(Xm[ok])), 0, None)
        return metrics.wmape(y_raw[ok], f), metrics.rmsle(y_raw[ok], f)

    base_w, base_r = score(X)
    rows = []
    for g, cols in EXPLAIN_GROUPS.items():
        idx = [FEATURES.index(c) for c in cols]
        if np.nanmax(np.nanstd(X[ok][:, idx], axis=0)) == 0:
            # e.g. oil: identical for every row of a single forecast origin, so shuffling rows is a no-op
            rows.append(dict(driver_group=g, WMAPE_shuffled=np.nan, d_WMAPE=np.nan, RMSLE_shuffled=np.nan, d_RMSLE=np.nan))
            continue
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, idx] = X[perm][:, idx]
        w, r = score(Xp)
        rows.append(dict(driver_group=g, WMAPE_shuffled=w, d_WMAPE=w - base_w, RMSLE_shuffled=r, d_RMSLE=r - base_r))
    out = pd.DataFrame(rows).sort_values("d_RMSLE", ascending=False)
    out.insert(1, "WMAPE_base", base_w)
    out.insert(2, "RMSLE_base", base_r)
    return out.reset_index(drop=True)


def group_shapley(gbm, X_inst: np.ndarray, X_bg: np.ndarray, groups: dict[str, list[str]] | None = None):
    """Exact Shapley values over feature groups.

    Returns (phi [n, G], base [n], full [n], group_names).  ``full - base == phi.sum(1)``."""
    groups = groups or EXPLAIN_GROUPS
    names = list(groups)
    G = len(names)
    cols = [[FEATURES.index(c) for c in groups[g] if c in gbm.cols] for g in names]
    n, b = len(X_inst), len(X_bg)
    vals = np.zeros((2 ** G, n))
    for mask in range(2 ** G):
        Z = np.broadcast_to(X_bg[None], (n, b, X_bg.shape[1])).copy()
        for g in range(G):
            if mask >> g & 1:
                Z[:, :, cols[g]] = X_inst[:, None, cols[g]]
        vals[mask] = gbm.predict_dev(Z.reshape(n * b, -1)).reshape(n, b).mean(1)
    phi = np.zeros((n, G))
    fact = [math.factorial(i) for i in range(G + 1)]
    for g in range(G):
        for mask in range(2 ** G):
            if mask >> g & 1:
                continue
            k = bin(mask).count("1")
            wgt = fact[k] * fact[G - k - 1] / fact[G]
            phi[:, g] += wgt * (vals[mask | (1 << g)] - vals[mask])
    return phi, vals[0], vals[-1], names


def select_showcase_series(P, X_final, info, lvl_sh, age, fc_units) -> pd.DataFrame:
    """Pick a handful of illustrative series automatically (no manual cherry-picking)."""
    S = P.S
    meta = P.meta[["series_id", "store_nbr", "family", "type"]].copy()
    vol = np.nan_to_num(np.nanmean(P.Y[:, -52:], axis=1))
    meta["vol52"] = vol
    H = P.H
    fidx = {c: i for i, c in enumerate(FEATURES)}
    Xh = X_final.reshape(H, S, -1)
    meta["promo_l4"] = Xh[0][:, fidx["promo_l4"]]
    meta["seas_dev"] = np.nanmean(np.abs(Xh[:, :, fidx["d_ly_sm"]]), axis=0)
    meta["age"] = age
    top300 = meta.sort_values("vol52", ascending=False).head(300)
    picks = {}
    picks["Highest-volume series"] = int(meta["vol52"].idxmax())
    picks["Most promotion-exposed (top-300 volume)"] = int(top300["promo_l4"].idxmax())
    picks["Strongest seasonal swing (top-300 volume)"] = int(top300["seas_dev"].idxmax())
    cs = meta[(meta["age"] < 26) & (meta["vol52"] > 0)]
    if len(cs):
        picks["Cold-start (<26 weeks of history)"] = int(cs["vol52"].idxmax())
    mid = meta[(meta["vol52"] > meta["vol52"].quantile(0.5)) & (meta["vol52"] < meta["vol52"].quantile(0.6))]
    picks["Mid-volume series"] = int(mid.sort_values("vol52").iloc[len(mid) // 2]["series_id"])
    rows = [{"showcase": k, **meta.loc[v].to_dict()} for k, v in picks.items()]
    return pd.DataFrame(rows)


def local_attribution(P, gbm, X_final: np.ndarray, info: dict, series_ids, X_bg: np.ndarray) -> pd.DataFrame:
    """Group-Shapley attribution of the 13 weekly forecasts of each selected series."""
    H, S = P.H, P.S
    rows_idx = np.array([h * S + s for s in series_ids for h in range(H)])
    phi, base, full, names = group_shapley(gbm, X_final[rows_idx], X_bg)
    out = []
    for n, ri in enumerate(rows_idx):
        s, h = ri % S, ri // S + 1
        out.append(dict(series_id=int(s), h=h, run_rate_log=float(info["lvl"][ri]), base_dev=float(base[n]),
                        model_dev=float(full[n]), **{nm: float(phi[n, k]) for k, nm in enumerate(names)}))
    return pd.DataFrame(out)
