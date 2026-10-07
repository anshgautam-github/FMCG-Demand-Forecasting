#!/usr/bin/env python
"""End-to-end pipeline.

    python run_pipeline.py                       # everything (about 15-20 min on 1 CPU)
    python run_pipeline.py --stages eda,cv       # selected stages
    python run_pipeline.py --raw-dir /path/to/csvs --force

Stages (each is cached under outputs/cache, so re-runs are fast):
    eda       data-quality checks, EDA figures, report 01
    cv        rolling-origin CV of all models, ensemble selection, interval calibration check
    ablation  driver-group ablation study
    forecast  final 13-week forecast + prediction intervals  (outputs/forecast_13w.csv)
    explain   global permutation importance + local Shapley driver attribution
    scenarios promo / traffic scenarios and promo response curves
    report    figures + auto-generated results summary
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALL = ["eda", "cv", "ablation", "forecast", "explain", "scenarios", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    a = ap.parse_args()
    if a.raw_dir:
        os.environ["FMCG_RAW_DIR"] = a.raw_dir
    import numpy as np
    import pandas as pd
    import config
    if a.force:
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from src import (ablation, cv, ensemble, eda, explain, features, forecast, quality, reporting,
                     scenarios, validation)
    from src.data_loading import load_raw
    from src.weekly_dataset import build_panel

    stages = [s.strip() for s in a.stages.split(",")]
    t00 = time.time()

    def say(msg):
        print(f"[{time.time() - t00:6.0f}s] {msg}", flush=True)

    say(f"loading raw data from {config.RAW_DIR}")
    raw = load_raw(config.RAW_DIR)
    P = build_panel(raw, config.HORIZON)
    folds = validation.make_folds(P, config.N_FOLDS, config.HORIZON)
    validation.describe_folds(P, folds).to_csv(config.TAB_DIR / "cv_fold_definition.csv", index=False)
    say(f"weekly panel: {P.S} series x {P.Wh} weeks; folds={[(f.k, str(P.weeks[f.origin].date())) for f in folds]}")

    if "eda" in stages:
        say("stage eda")
        S, T = quality.run_quality_checks(raw, P)
        info = eda.make_eda_figures(raw, P, S, T)
        eda.write_dq_report(S, T, info)

    cv_df = None
    if any(s in stages for s in ("cv", "forecast", "explain", "scenarios", "report")):
        say("stage cv (rolling-origin)")
        cv_df = cv.run_cv(P, folds)
        partner, w, wtab = ensemble.select_blend(cv_df, select_folds=[f.k for f in folds[:-1]])
        cold_col, ctab = ensemble.select_cold(cv_df)
        cv_df = ensemble.add_ensemble(cv_df, partner, w, cold_col)
        wtab.to_csv(config.TAB_DIR / "ensemble_weight_search.csv", index=False)
        ctab.to_csv(config.TAB_DIR / "coldstart_strategy_comparison.csv", index=False)
        cfg = dict(partner=partner, w_gbm=w, cold_strategy=cold_col)
        (config.TAB_DIR / "ensemble_config.json").write_text(json.dumps(cfg, indent=2))
        say(f"ensemble: {w:.1f} x GBM + {1 - w:.1f} x {partner}; cold-start strategy: {cold_col}")
        T = reporting.cv_tables(cv_df, P)
        say("overall CV:\n" + T["overall"].round(4).to_string(index=False))
        summ, by_h, by_t, _ = reporting.interval_evaluation(cv_df, folds[-1].k)
        say("interval check on last fold:\n" + summ.round(4).to_string(index=False))
        reporting.fig_model_comparison(T); reporting.fig_horizon(T); reporting.fig_segments(T)
        reporting.fig_backtest_aggregate(cv_df); reporting.fig_calibration(by_h, summ); reporting.fig_scatter(cv_df)

    if "ablation" in stages or "report" in stages:
        say("stage ablation")
        raw_ab = ablation.run_ablation(P, folds)
        ab = ablation.summarize_ablation(raw_ab)
        ab.to_csv(config.TAB_DIR / "ablation_summary.csv", index=False)
        raw_ab.to_csv(config.TAB_DIR / "ablation_by_fold.csv", index=False)
        reporting.fig_ablation(ab)
        say("ablation:\n" + ab.round(4).to_string(index=False))

    fc = None
    if any(s in stages for s in ("forecast", "explain", "scenarios", "report")):
        say("stage forecast")
        fc, gbm, ci, (Xf, inf, lvl_sh, age) = forecast.make_final_forecast(P, cv_df, partner, w, cold_col)
        fc.to_csv(config.OUT_DIR / "forecast_13w.csv", index=False)
        agg = (fc.assign(date=fc["date"].astype(str)).groupby(["date", "horizon_week"])
               [["forecast", "forecast_gbm", "forecast_stat"]].sum().reset_index())
        agg.to_csv(config.TAB_DIR / "forecast_total_by_week.csv", index=False)
        (fc.groupby("sku_id")[["forecast"]].sum().sort_values("forecast", ascending=False)
           .to_csv(config.TAB_DIR / "forecast_total_by_family.csv"))
        say(f"forecast rows={len(fc):,}; total 13-week units={fc['forecast'].sum():,.0f}")

    show = None
    if any(s in stages for s in ("explain", "report")):
        say("stage explain")
        X_all = Xf
        # global importance on two held-out folds (pooled)
        pis = []
        for k in (2, 4):
            f = folds[k - 1]
            gk = __import__("joblib").load(config.CACHE_DIR / f"gbm_fold{k}.joblib")
            X, info = features.build_block(P, f.origin, range(1, P.H + 1))
            yl, yr, cl = features.targets_for(P, info)
            never = (P.first_pos < 0) | (P.first_pos > f.origin)
            elig = ~cl & ~never[info["sidx"]]
            pis.append(explain.group_permutation_importance(gk, X, info["lvl"], yr, elig, seed=k))
        pi = (pd.concat(pis).groupby("driver_group", as_index=False).mean()
              .sort_values("d_RMSLE", ascending=False, na_position="last"))
        pi.to_csv(config.TAB_DIR / "explain_permutation_importance.csv", index=False)
        reporting.fig_perm(pi)
        # local attribution
        show = explain.select_showcase_series(P, Xf, inf, lvl_sh, age, None)
        rng = np.random.default_rng(config.SEED)
        Xtr_bg, _, _, _ = features.build_training(P, P.Wh - 1, 12, 20000, rng, config.RECENCY_HALF_LIFE_W)
        X_bg = Xtr_bg[rng.choice(len(Xtr_bg), 40, replace=False)]
        attr = explain.local_attribution(P, gbm, Xf, inf, show["series_id"].tolist(), X_bg)
        attr = attr.merge(P.meta[["series_id", "store_nbr", "family"]], on="series_id")
        attr.to_csv(config.TAB_DIR / "explain_local_attribution_by_week.csv", index=False)
        show.to_csv(config.TAB_DIR / "explain_showcase_series.csv", index=False)
        names = list(explain.EXPLAIN_GROUPS)
        # efficiency check: contributions sum to (model - base)
        err = np.abs(attr[names].sum(1) - (attr["model_dev"] - attr["base_dev"])).max()
        say(f"Shapley efficiency check max |error| = {err:.2e}")
        reporting.fig_shapley(P, attr, show, fc)
        reporting.fig_forecast_examples(P, fc, show)
        summ_attr = attr.groupby(["series_id", "store_nbr", "family"])[names].mean().reset_index()
        summ_attr.to_csv(config.TAB_DIR / "explain_local_attribution_summary.csv", index=False)

    if any(s in stages for s in ("scenarios", "report")):
        say("stage scenarios")
        res, fam, weekly, curve, base = scenarios.run_scenarios(P, gbm, P.Wh - 1)
        res.to_csv(config.TAB_DIR / "scenario_summary.csv", index=False)
        fam.to_csv(config.TAB_DIR / "scenario_by_family.csv", index=False)
        weekly.to_csv(config.TAB_DIR / "scenario_by_week.csv", index=False)
        curve.to_csv(config.TAB_DIR / "scenario_promo_response_curves.csv", index=False)
        reporting.fig_scenarios(res, fam, curve)
        say("scenarios:\n" + res[["scenario", "delta_units", "delta_pct", "share_series_up"]].round(3).to_string(index=False))

    say("done. Outputs in outputs/ (forecast_13w.csv, tables/, figures/, reports/)")


if __name__ == "__main__":
    main()
