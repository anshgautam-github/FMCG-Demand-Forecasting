"""Tables and figures for model comparison, intervals, explainability and scenarios."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from . import metrics
from .intervals import ConformalIntervals, coverage, interval_score
from .explain import EXPLAIN_GROUPS

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9})
C = {"naive_ma4": "#9aa5b1", "snaive_growth": "#c9a227", "damped_holt": "#7b6fb0", "gbm": "#1f5f8b",
     "ensemble": "#e07a1f"}
LABEL = {"naive_ma4": "Naive (4-wk mean)", "snaive_growth": "Seasonal naive x growth", "damped_holt": "Damped Holt (ETS)",
         "gbm": "Global GBM", "ensemble": "Hybrid ensemble (GBM + stat.)"}
MODELS = ["naive_ma4", "snaive_growth", "damped_holt", "gbm", "ensemble"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _csv(df, name):
    df.to_csv(config.TAB_DIR / name, index=False)
    return df


# ---------------------------------------------------------------- cross-validation tables
def cv_tables(df: pd.DataFrame, P) -> dict:
    e = df[df["eval"]].copy()
    T = {}
    T["overall"] = _csv(pd.DataFrame([{"model": m, **metrics.summarize(e["y"], e[m])} for m in MODELS]),
                        "cv_model_comparison_overall.csv")
    T["by_fold"] = _csv(metrics.table_by(e, MODELS, ["fold"]), "cv_by_fold.csv")
    T["by_horizon"] = _csv(metrics.table_by(e, MODELS, ["h"]), "cv_by_horizon.csv")
    T["by_family"] = _csv(metrics.table_by(e, MODELS, ["family"]), "cv_by_family.csv")
    T["by_store"] = _csv(metrics.table_by(e, MODELS, ["store_nbr"]), "cv_by_store.csv")
    T["by_type"] = _csv(metrics.table_by(e, MODELS, ["type"]), "cv_by_store_type.csv")
    # volume tier of the series at the origin
    e["tier"] = pd.qcut(e["lvl"].rank(method="first"), 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
    T["by_tier"] = _csv(metrics.table_by(e, MODELS, ["tier"]), "cv_by_volume_tier.csv")
    T["by_segment"] = _csv(pd.concat([
        metrics.table_by(e.assign(seg="Cold-start (<13 wks history)").query("cold"), MODELS, ["seg"]),
        metrics.table_by(e.assign(seg="Established").query("~cold"), MODELS, ["seg"])]), "cv_by_segment.csv")
    # aggregate (bottom-up) accuracy
    agg = []
    for lvl in ("total", "family", "store"):
        a = metrics.aggregate_level_wmape(e, MODELS, lvl)
        agg.append(a)
    T["aggregate"] = _csv(pd.concat(agg), "cv_aggregate_levels.csv")
    # series-level distribution of WMAPE
    sw = e.groupby(["series_id"]).apply(lambda g: pd.Series({m: metrics.wmape(g["y"], g[m]) for m in MODELS}),
                                       include_groups=False)
    sw = sw[e.groupby("series_id")["y"].sum() > 0]
    T["series_wmape_dist"] = _csv(sw.describe(percentiles=[.25, .5, .75, .9]).T.reset_index().rename(columns={"index": "model"}),
                                  "cv_series_level_wmape_distribution.csv")
    # sensitivity: promotion plan unknown, business rules
    sens = pd.DataFrame([
        {"variant": "GBM, promo plan known (planned/trade calendar)", **metrics.summarize(e["y"], e["gbm"])},
        {"variant": "GBM, promo plan unknown (last-4-wk average carried forward)", **metrics.summarize(e["y"], e["gbm_promo_unknown"])},
        {"variant": "GBM without business-rule layer", **metrics.summarize(e["y"], e["gbm_norules"])},
        {"variant": "GBM with business-rule layer", **metrics.summarize(e["y"], e["gbm"])},
        {"variant": "Ensemble, scored on the primary universe (excl. reporting-gap weeks)", **metrics.summarize(e["y"], e["ensemble"])},
        {"variant": "Ensemble, scored INCLUDING reporting-gap weeks", **metrics.summarize(
            df.loc[df["eval_incl_gaps"], "y"], df.loc[df["eval_incl_gaps"], "ensemble"])},
        {"variant": "Naive, scored INCLUDING reporting-gap weeks", **metrics.summarize(
            df.loc[df["eval_incl_gaps"], "y"], df.loc[df["eval_incl_gaps"], "naive_ma4"])},
    ])
    T["sensitivity"] = _csv(sens, "cv_sensitivity_promo_plan_and_rules.csv")
    # skill vs naive per family
    fam = T["by_family"].pivot(index="family", columns="model", values="WMAPE")
    fam["skill_vs_naive_%"] = 100 * (1 - fam["ensemble"] / fam["naive_ma4"])
    vol = e.groupby("family")["y"].sum()
    fam["volume_share_%"] = 100 * vol / vol.sum()
    T["family_skill"] = _csv(fam.sort_values("volume_share_%", ascending=False).reset_index(), "cv_family_skill.csv")
    return T


# ---------------------------------------------------------------- interval evaluation
def interval_evaluation(df: pd.DataFrame, last_fold: int):
    train = df[df["eval"] & (df["fold"] < last_fold)]
    test = df[df["eval"] & (df["fold"] == last_fold)].copy()
    ci = ConformalIntervals(levels=config.INTERVAL_LEVELS).fit(train["h"], train["lvl"], train["y"], train["ensemble"])
    iv = ci.predict(test["h"].values, test["lvl"].values, test["ensemble"].values)
    test = pd.concat([test.reset_index(drop=True), iv], axis=1)
    rows = []
    for lev in config.INTERVAL_LEVELS:
        p = int(round(lev * 100))
        rows.append(dict(level=p, nominal=lev, coverage=coverage(test["y"], test[f"lower_{p}"], test[f"upper_{p}"]),
                         mean_width_units=float((test[f"upper_{p}"] - test[f"lower_{p}"]).mean()),
                         interval_score=interval_score(test["y"], test[f"lower_{p}"], test[f"upper_{p}"], 1 - lev)))
    summ = pd.DataFrame(rows)
    by_h = []
    for h, g in test.groupby("h"):
        for lev in config.INTERVAL_LEVELS:
            p = int(round(lev * 100))
            by_h.append(dict(h=h, level=p, coverage=coverage(g["y"], g[f"lower_{p}"], g[f"upper_{p}"])))
    by_h = pd.DataFrame(by_h)
    test["tier"] = pd.qcut(test["lvl"].rank(method="first"), 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
    by_t = test.groupby("tier", observed=True).apply(
        lambda g: pd.Series({f"cov_{int(round(l*100))}": coverage(g["y"], g[f"lower_{int(round(l*100))}"], g[f"upper_{int(round(l*100))}"])
                             for l in config.INTERVAL_LEVELS}), include_groups=False).reset_index()
    _csv(summ, "interval_coverage_holdout_fold.csv"); _csv(by_h, "interval_coverage_by_horizon.csv")
    _csv(by_t, "interval_coverage_by_volume_tier.csv")
    return summ, by_h, by_t, test


# ---------------------------------------------------------------- figures
def fig_model_comparison(T):
    bf = T["by_fold"].pivot(index="fold", columns="model", values="WMAPE")[MODELS]
    ov = T["overall"].set_index("model")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(bf)); w = .16
    for i, m in enumerate(MODELS):
        axs[0].bar(x + (i - 2) * w, bf[m] * 100, w, color=C[m], label=LABEL[m])
    axs[0].set_xticks(x); axs[0].set_xticklabels([f"Fold {k}" for k in bf.index])
    axs[0].set_ylabel("WMAPE %"); axs[0].set_title("WMAPE by rolling-origin fold (13-week horizon)")
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].barh([LABEL[m] for m in MODELS][::-1], (ov.loc[MODELS, "WMAPE"] * 100)[::-1], color=[C[m] for m in MODELS][::-1])
    for i, m in enumerate(MODELS[::-1]):
        axs[1].text(ov.loc[m, "WMAPE"] * 100 + .1, i, f"{ov.loc[m, 'WMAPE']*100:.1f}%", va="center", fontsize=8)
    axs[1].set_title("Pooled over 4 folds"); axs[1].set_xlim(0, ov["WMAPE"].max() * 100 * 1.2)
    _save(fig, "res01_model_comparison.png")


def fig_horizon(T):
    h = T["by_horizon"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for m in MODELS:
        g = h[h["model"] == m]
        axs[0].plot(g["h"], g["WMAPE"] * 100, color=C[m], label=LABEL[m], lw=1.8 if m in ("gbm", "ensemble") else 1)
        axs[1].plot(g["h"], g["Bias"] * 100, color=C[m], lw=1.8 if m in ("gbm", "ensemble") else 1)
    axs[0].set_xlabel("horizon (weeks ahead)"); axs[0].set_ylabel("WMAPE %"); axs[0].legend(fontsize=7)
    axs[0].set_title("Accuracy decay with horizon")
    axs[1].axhline(0, color="k", lw=.6); axs[1].set_xlabel("horizon (weeks ahead)"); axs[1].set_ylabel("Bias % (f - y)/y")
    axs[1].set_title("Bias by horizon")
    _save(fig, "res02_by_horizon.png")


def fig_segments(T):
    fam = T["family_skill"].sort_values("volume_share_%", ascending=False).head(15)
    st = T["by_store"].pivot(index="store_nbr", columns="model", values="WMAPE")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.arange(len(fam))
    axs[0].barh(y + .2, fam["naive_ma4"] * 100, .4, color=C["naive_ma4"], label="Naive")
    axs[0].barh(y - .2, fam["ensemble"] * 100, .4, color=C["ensemble"], label="Ensemble")
    axs[0].set_yticks(y); axs[0].set_yticklabels(fam["family"], fontsize=7); axs[0].invert_yaxis()
    axs[0].set_xlabel("WMAPE %"); axs[0].set_title("15 largest families by volume"); axs[0].legend(fontsize=7)
    axs[1].scatter(st["naive_ma4"] * 100, st["ensemble"] * 100, s=18, color=C["ensemble"])
    lim = [0, max(st["naive_ma4"].max(), st["ensemble"].max()) * 100 * 1.05]
    axs[1].plot(lim, lim, "k--", lw=.7); axs[1].set_xlabel("Naive WMAPE % (per store)")
    axs[1].set_ylabel("Ensemble WMAPE %"); axs[1].set_title("Store-level accuracy (below diagonal = better than naive)")
    _save(fig, "res03_segments.png")


def fig_backtest_aggregate(df: pd.DataFrame):
    e = df[df["eval"]]
    a = e.groupby(["fold", "week"])[["y", "naive_ma4", "ensemble"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(a["week"], a["y"] / 1e6, color="k", lw=1.6, label="Actual")
    ax.plot(a["week"], a["ensemble"] / 1e6, color=C["ensemble"], lw=1.4, label="Ensemble (out-of-sample)")
    ax.plot(a["week"], a["naive_ma4"] / 1e6, color=C["naive_ma4"], lw=1, label="Naive")
    for f in a["fold"].unique():
        ax.axvline(a[a["fold"] == f]["week"].min(), color="grey", ls=":", lw=.8)
    ax.set_ylabel("Total units / week (M)")
    ax.set_title("Out-of-sample backtest, total of all series (dotted lines = fold origins)")
    ax.legend(fontsize=8)
    _save(fig, "res04_aggregate_backtest.png")


def fig_calibration(by_h, summ):
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for lev, col in ((50, "#3a9d5d"), (90, "#1f5f8b")):
        g = by_h[by_h["level"] == lev]
        ax.plot(g["h"], g["coverage"] * 100, "o-", color=col, label=f"{lev}% interval")
        ax.axhline(lev, color=col, ls="--", lw=.7)
    ax.set_xlabel("horizon (weeks ahead)"); ax.set_ylabel("empirical coverage %")
    ax.set_title("Interval calibration on the untouched last fold\n(dashed = nominal)")
    ax.legend(fontsize=8)
    _save(fig, "res05_interval_calibration.png")


def fig_ablation(ab):
    a = ab[ab["variant"] != "full model"].sort_values("d_WMAPE")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    cols = ["#c0392b" if v > 0 else "#7f8c8d" for v in a["d_WMAPE"]]
    axs[0].barh(a["variant"], a["d_WMAPE"] * 100, color=cols)
    axs[0].axvline(0, color="k", lw=.7)
    axs[0].set_xlabel("change in WMAPE (percentage points) vs full model")
    axs[0].set_title("Ablation: cost of REMOVING a driver group\n(positive = driver helps)")
    axs[1].barh(a["variant"], a["d_RMSLE"], color=cols)
    axs[1].axvline(0, color="k", lw=.7); axs[1].set_yticklabels([])
    axs[1].set_xlabel("change in RMSLE"); axs[1].set_title("Same, RMSLE (log-scale, all series)")
    _save(fig, "res06_ablation.png")


def fig_perm(pi):
    p = pi.dropna(subset=["d_RMSLE"]).sort_values("d_RMSLE")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(p["driver_group"], p["d_RMSLE"], color="#1f5f8b")
    ax.set_xlabel("increase in RMSLE when the driver group is shuffled")
    ax.set_title("Global driver importance (permutation, held-out folds)\nmacro/oil n/a: constant within one origin - see ablation")
    _save(fig, "res07_permutation_importance.png")


def fig_shapley(P, attr: pd.DataFrame, show: pd.DataFrame, fc: pd.DataFrame):
    names = list(EXPLAIN_GROUPS)
    n = len(show)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12.5, 3.6 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[n:]:
        ax.axis("off")
    for ax, r in zip(axs, show.itertuples()):
        g = attr[attr["series_id"] == r.series_id]
        phi = g[names].mean().values
        pct = (np.exp(phi) - 1) * 100
        order = np.argsort(np.abs(phi))
        cols = ["#3a9d5d" if v > 0 else "#c0392b" for v in phi[order]]
        ax.barh(np.array(names)[order], pct[order], color=cols)
        ax.axvline(0, color="k", lw=.7)
        ax.set_title(f"{r.showcase}\nstore {r.store_nbr} - {r.family}", fontsize=9)
        ax.set_xlabel("effect on 13-wk forecast vs a typical series (%)", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
    _save(fig, "res08_local_driver_attribution.png")


def fig_forecast_examples(P, fc: pd.DataFrame, show: pd.DataFrame):
    n = len(show)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12.5, 3.4 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[n:]:
        ax.axis("off")
    for ax, r in zip(axs, show.itertuples()):
        s = r.series_id
        hist = P.Y_raw[s, -52:]
        w_h = P.weeks[P.Wh - 52:P.Wh]
        f = fc[(fc["market_id"] == r.store_nbr) & (fc["sku_id"] == r.family)].sort_values("horizon_week")
        ax.plot(w_h, hist, color="k", lw=1, label="actual (last 52 wks)")
        ax.plot(f["date"], f["forecast"], color="#e07a1f", lw=1.8, label="forecast")
        ax.fill_between(f["date"], f["lower_90"], f["upper_90"], color="#e07a1f", alpha=.18, label="90% interval")
        ax.fill_between(f["date"], f["lower_50"], f["upper_50"], color="#e07a1f", alpha=.3, label="50% interval")
        ax.set_title(f"{r.showcase}\nstore {r.store_nbr} - {r.family}", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    axs[0].legend(fontsize=7)
    _save(fig, "res09_forecast_examples.png")


def fig_scenarios(res, fam, curve):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    r = res.copy()
    r["lab"] = r["scenario"].str.replace(r"^S\d ", "", regex=True)
    axs[0].barh(r["lab"][::-1], r["delta_pct"][::-1], color=["#3a9d5d" if v > 0 else "#c0392b" for v in r["delta_pct"][::-1]])
    for i, v in enumerate(r["delta_pct"][::-1]):
        axs[0].text(v, i, f" {v:+.1f}%", va="center", fontsize=8)
    axs[0].axvline(0, color="k", lw=.7); axs[0].set_xlabel("13-week demand vs baseline plan, %")
    axs[0].set_title("Scenario impact on total units")
    top = (curve[curve["multiplier"] == 1.0].sort_values("units", ascending=False).head(6)["family"])
    for fam_ in top:
        g = curve[curve["family"] == fam_]
        axs[1].plot(g["multiplier"], g["units_vs_base_pct"], "o-", label=fam_, lw=1.3)
    axs[1].axhline(0, color="k", lw=.6); axs[1].axvline(1, color="k", lw=.6, ls=":")
    axs[1].set_xlabel("promotion intensity multiplier (1 = plan)"); axs[1].set_ylabel("units vs plan, %")
    axs[1].set_title("Promo response curves, 6 largest families"); axs[1].legend(fontsize=7)
    _save(fig, "res10_scenarios.png")


def fig_scatter(df):
    e = df[df["eval"] & (df["y"] > 0)]
    e = e.sample(min(6000, len(e)), random_state=0)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(np.log1p(e["y"]), np.log1p(e["ensemble"]), s=3, alpha=.3, color=C["ensemble"])
    lim = [0, np.log1p(e["y"]).max()]
    ax.plot(lim, lim, "k--", lw=.7)
    ax.set_xlabel("log1p actual units"); ax.set_ylabel("log1p forecast")
    ax.set_title("Out-of-sample forecast vs actual (6,000 series-weeks)")
    _save(fig, "res11_actual_vs_forecast.png")
