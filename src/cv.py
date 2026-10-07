"""Rolling-origin cross-validation of all candidate forecasting approaches."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

import config
from . import models_stat as ms
from .features import build_block, build_training, targets_for
from .models_gbm import GlobalGBM
from .validation import Fold, make_folds
from .weekly_dataset import Panel

MODELS = ["naive_ma4", "snaive_growth", "damped_holt", "gbm"]


def to_SH(vec: np.ndarray, P: Panel) -> np.ndarray:
    """h-major vector (length H*S) -> [S, H]."""
    return vec.reshape(P.H, P.S).T


def fill_nan(F: np.ndarray, P: Panel, o: int) -> np.ndarray:
    """Replace NaN forecasts by the series' own 26-week mean (or 0 if it has no history)."""
    from .features import nmean
    fb = np.nan_to_num(nmean(P.Y, o - 25, o + 1), nan=0.0)
    return np.where(np.isnan(F), fb[:, None], F)


def gbm_units(P: Panel, gbm: GlobalGBM, o: int, promo=None, trans_shift=0.0):
    X, info = build_block(P, o, range(1, P.H + 1), promo=promo, trans_shift=trans_shift)
    dev = gbm.predict_dev(X)
    lvl = to_SH(info["lvl"], P)
    pred = np.expm1(to_SH(dev, P) + lvl)
    return np.clip(pred, 0, None), lvl, X, info


def assumed_promo(P: Panel, o: int) -> np.ndarray:
    """What-if 'promo plan unknown': carry the last-4-week average forward."""
    pr = P.PROMO.copy()
    with np.errstate(all="ignore"):
        carry = np.nanmean(P.PROMO[:, o - 3:o + 1], axis=1)
    pr[:, o + 1:o + 1 + P.H] = carry[:, None]
    return pr


def fit_fold_model(P: Panel, o: int, drop_groups=(), max_rows=None, max_iter=None, seed=config.SEED, vol_power=None):
    rng = np.random.default_rng(seed)
    Xtr, ytr, wtr, mt = build_training(
        P, o, config.TRAIN_ORIGIN_STEP, max_rows or config.TRAIN_MAX_ROWS, rng, config.RECENCY_HALF_LIFE_W,
        gaps=P.gap_mask(o))
    if vol_power is None:
        vol_power = config.VOL_WEIGHT_POWER
    if vol_power:
        # volume weighting aligns the log-scale loss with the volume-weighted business metric (WMAPE)
        vw = np.exp(vol_power * mt[:, 3]); wtr = wtr * (vw / vw.mean()).astype(np.float32)
    params = {"max_iter": max_iter} if max_iter else None
    return GlobalGBM(drop_groups, params).fit(Xtr, ytr, wtr), len(ytr)


def evaluate_fold(P: Panel, fold: Fold, gbm: GlobalGBM, extras: bool = True) -> pd.DataFrame:
    o, H, S = fold.origin, P.H, P.S
    gb, lvl, _, _ = gbm_units(P, gbm, o)
    comp = {
        "naive_ma4": ms.naive_ma(P, o, 4),
        "snaive_growth": ms.seasonal_naive_growth(P, o),
        "damped_holt": ms.damped_holt(P, o),
    }
    cold = ms.coldstart_peer(P, o)
    age = ms.age_at(P, o)
    cold_mask = (np.nan_to_num(age, nan=1e9) < config.COLDSTART_MIN_ACTIVE_WEEKS)
    gb_f = np.where(np.isnan(lvl), cold, gb)                     # un-forecastable by GBM -> peers
    comp["gbm"] = gb_f
    if extras:
        gb_pa, _, _, _ = gbm_units(P, gbm, o, promo=assumed_promo(P, o))
        comp["gbm_promo_unknown"] = np.where(np.isnan(lvl), cold, gb_pa)
    out = {}
    for k, v in comp.items():
        out[k] = ms.apply_rules(P, o, fill_nan(v, P, o))
    # cold-start variants (reported on the cold segment only)
    cold_f = ms.apply_rules(P, o, fill_nan(np.where(np.isnan(cold), gb_f, cold), P, o))
    out["coldstart_peer"] = cold_f
    out["gbm_norules"] = np.clip(fill_nan(gb_f, P, o), 0, None)      # to quantify the business-rule layer
    out["gbm_cold_blend"] = ms.apply_rules(P, o, fill_nan(0.5 * gb_f + 0.5 * np.where(np.isnan(cold), gb_f, cold), P, o))

    rows = []
    meta = P.meta
    never = (P.first_pos < 0) | (P.first_pos > o)
    gapfull = P.gap_mask(P.Wh - 1)        # evaluation-universe flag (uses hindsight; see report)
    for h in range(1, H + 1):
        j = o + h
        d = pd.DataFrame({
            "fold": fold.k, "series_id": np.arange(S), "h": h, "week_idx": j,
            "week": P.weeks[j], "store_nbr": meta["store_nbr"].values, "family": meta["family"].values,
            "type": meta["type"].values, "y": P.Y_raw[:, j], "lvl": lvl[:, h - 1],
            "age_at_origin": age, "cold": cold_mask & ~never, "launched": ~never,
            "store_closed": P.closed_series[:, j], "in_gap": gapfull[:, j],
        })
        for k, v in out.items():
            d[k] = v[:, h - 1]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    # evaluation universe: launched at origin, store open and series not in a reporting gap that week
    df["eval_incl_gaps"] = df["launched"] & ~df["store_closed"]
    df["eval"] = df["eval_incl_gaps"] & ~df["in_gap"]
    return df


def run_cv(P: Panel, folds: list[Fold], cache_name: str = "cv_eval") -> pd.DataFrame:
    cache = config.CACHE_DIR / f"{cache_name}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    dfs, log = [], []
    for f in folds:
        t0 = time.time()
        gbm, n = fit_fold_model(P, f.origin)
        joblib.dump(gbm, config.CACHE_DIR / f"gbm_fold{f.k}.joblib")
        df = evaluate_fold(P, f, gbm)
        dfs.append(df)
        log.append(dict(fold=f.k, train_rows=n, seconds=round(time.time() - t0, 1)))
        print(f"  fold {f.k}: origin={P.weeks[f.origin].date()} rows={n:,} ({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(dfs, ignore_index=True)
    out["family"] = out["family"].astype(str)
    out.to_pickle(cache)
    pd.DataFrame(log).to_csv(config.TAB_DIR / "cv_training_log.csv", index=False)
    return out


def gbm_forecast(P: Panel, gbm: GlobalGBM, o: int, promo=None, trans_shift: float = 0.0, cold=None) -> np.ndarray:
    """Rule-adjusted GBM unit forecasts [S, H] (cold-start peers where the GBM has no run-rate)."""
    gb, lvl, _, _ = gbm_units(P, gbm, o, promo=promo, trans_shift=trans_shift)
    if cold is None:
        cold = ms.coldstart_peer(P, o)
    return ms.apply_rules(P, o, fill_nan(np.where(np.isnan(lvl), cold, gb), P, o))
