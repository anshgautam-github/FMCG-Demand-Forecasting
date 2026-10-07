"""Final 13-week forecast for every SKU-market series, with prediction intervals."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import config
from . import ensemble as ens, models_stat as ms
from .cv import fill_nan, fit_fold_model, gbm_units
from .intervals import ConformalIntervals


def make_final_forecast(P, cv_eval: pd.DataFrame, partner: str, w: float, cold_col: str):
    """Train on ALL observed weeks, forecast weeks Wh .. Wh+H-1, attach calibrated intervals."""
    o = P.Wh - 1
    path = config.CACHE_DIR / "gbm_final.joblib"
    if path.exists():
        gbm = joblib.load(path)
    else:
        gbm, n = fit_fold_model(P, o)
        joblib.dump(gbm, path)
        print(f"  final model trained on {n:,} rows", flush=True)

    cold = ms.coldstart_peer(P, o)
    gb, lvl, X, info = gbm_units(P, gbm, o)
    gb_f = ms.apply_rules(P, o, fill_nan(np.where(np.isnan(lvl), cold, gb), P, o))
    stat = {"snaive_growth": ms.seasonal_naive_growth, "damped_holt": ms.damped_holt,
            "naive_ma4": lambda P_, o_: ms.naive_ma(P_, o_, 4)}[partner](P, o)
    stat_f = ms.apply_rules(P, o, fill_nan(stat, P, o))
    age = ms.age_at(P, o)
    cold_mask = np.nan_to_num(age, nan=1e9) < config.COLDSTART_MIN_ACTIVE_WEEKS
    cold_cols = {
        "gbm": gb_f,
        "coldstart_peer": ms.apply_rules(P, o, fill_nan(np.where(np.isnan(cold), gb_f, cold), P, o)),
        "gbm_cold_blend": ms.apply_rules(P, o, fill_nan(0.5 * gb_f + 0.5 * np.where(np.isnan(cold), gb_f, cold), P, o)),
        "snaive_growth": ms.apply_rules(P, o, fill_nan(ms.seasonal_naive_growth(P, o), P, o)),
        "damped_holt": ms.apply_rules(P, o, fill_nan(ms.damped_holt(P, o), P, o)),
    }
    final = ens.blend(gb_f, stat_f, w)
    final = np.where(cold_mask[:, None], cold_cols[cold_col], final)
    _, never, delisted = ms.apply_rules(P, o, final, return_flags=True)
    final = ms.apply_rules(P, o, final)

    # calibrated intervals from all out-of-sample CV residuals of the ensemble
    d = cv_eval[cv_eval["eval"]]
    ci = ConformalIntervals(levels=config.INTERVAL_LEVELS).fit(d["h"], d["lvl"], d["y"], d["ensemble"])
    rows = []
    meta = P.meta
    for h in range(1, P.H + 1):
        rows.append(pd.DataFrame({
            "date": P.weeks[o + h], "week_end": P.weeks[o + h] + pd.Timedelta(days=6), "horizon_week": h,
            "market_id": meta["store_nbr"].values, "sku_id": meta["family"].values,
            "forecast": final[:, h - 1], "forecast_gbm": gb_f[:, h - 1], "forecast_stat": stat_f[:, h - 1],
            "run_rate_units": np.expm1(lvl[:, h - 1]),
            "_lvl": lvl[:, h - 1],
            "promo_plan_known_share": P.promo_known_share[h - 1],
            "flag_cold_start": cold_mask, "flag_delisted_or_unlaunched": delisted[:, h - 1] | never,
        }))
    fc = pd.concat(rows, ignore_index=True)
    iv = ci.predict(fc["horizon_week"].values, fc["_lvl"].values, fc["forecast"].values)
    zero = fc["flag_delisted_or_unlaunched"].values
    for c in iv.columns:
        iv.loc[zero, c] = 0.0
    fc = pd.concat([fc, iv], axis=1).drop(columns="_lvl")
    num = fc.select_dtypes("number").columns
    fc[num] = fc[num].round(2)
    cols = ["date", "week_end", "horizon_week", "market_id", "sku_id", "forecast", "lower_50", "upper_50",
            "lower_90", "upper_90", "forecast_gbm", "forecast_stat", "run_rate_units", "promo_plan_known_share",
            "flag_cold_start", "flag_delisted_or_unlaunched"]
    return fc[cols], gbm, ci, (X, info, lvl, age)
