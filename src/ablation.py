"""Feature-impact / ablation study (task 8).

Each variant removes one driver group from the *global* model, re-trains it from scratch on the
same rows (light configuration, identical seed) and re-scores it on the same rolling-origin
folds.  Positive delta-WMAPE = the removed group was helping."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from . import metrics, models_stat as ms
from .cv import fill_nan, fit_fold_model, gbm_units
from .validation import Fold

VARIANTS = {
    "full model": (),
    "- promotions": ("PROMO",),
    "- calendar & holidays": ("CAL",),
    "- seasonal lags": ("SEAS",),
    "- availability proxy (distribution)": ("AVAIL",),
    "- store traffic": ("TRAFFIC",),
    "- macro (oil; weather proxy)": ("MACRO",),
    "autoregressive only": ("PROMO", "CAL", "SEAS", "AVAIL", "TRAFFIC", "MACRO"),
}


def run_ablation(P, folds: list[Fold]) -> pd.DataFrame:
    cache = config.CACHE_DIR / "ablation_raw.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    rows = []
    for f in folds:
        o = f.origin
        cold = ms.coldstart_peer(P, o)
        never = (P.first_pos < 0) | (P.first_pos > o)
        gapfull = P.gap_mask(P.Wh - 1)
        ev = np.zeros((P.S, P.H), bool)
        yy = np.zeros((P.S, P.H))
        for h in range(1, P.H + 1):
            yy[:, h - 1] = P.Y_raw[:, o + h]
            ev[:, h - 1] = ~never & ~P.closed_series[:, o + h] & ~gapfull[:, o + h]
        for name, drop in VARIANTS.items():
            t0 = time.time()
            gbm, _ = fit_fold_model(P, o, drop_groups=drop, max_rows=config.ABL_MAX_ROWS, max_iter=config.ABL_ITER)
            gb, lvl, _, _ = gbm_units(P, gbm, o)
            F = ms.apply_rules(P, o, fill_nan(np.where(np.isnan(lvl), cold, gb), P, o))
            m = metrics.summarize(yy[ev], F[ev])
            rows.append(dict(fold=f.k, variant=name, **m))
            print(f"  ablation fold {f.k} | {name:<38} WMAPE={m['WMAPE']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_pickle(cache)
    return out


def summarize_ablation(raw: pd.DataFrame) -> pd.DataFrame:
    full = raw[raw["variant"] == "full model"].set_index("fold")
    r = raw.copy()
    for c in ("WMAPE", "RMSE", "RMSLE", "Bias"):
        r[f"d_{c}"] = r[c].values - full.loc[r["fold"], c].values
    agg = r.groupby("variant", sort=False).agg(
        WMAPE=("WMAPE", "mean"), RMSE=("RMSE", "mean"), RMSLE=("RMSLE", "mean"), Bias=("Bias", "mean"),
        d_WMAPE=("d_WMAPE", "mean"), d_RMSLE=("d_RMSLE", "mean"),
        folds_worse=("d_WMAPE", lambda s: int((s > 0).sum())), n_folds=("d_WMAPE", "size")).reset_index()
    agg["d_WMAPE_pct_rel"] = 100 * agg["d_WMAPE"] / agg.loc[agg["variant"] == "full model", "WMAPE"].values[0]
    return agg
