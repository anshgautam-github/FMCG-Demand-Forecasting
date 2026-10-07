"""Exploratory data analysis: figures + a written EDA / data-quality report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from .quality import detect_anomalies

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9})
C1, C2, C3 = "#1f5f8b", "#e07a1f", "#3a9d5d"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def make_eda_figures(raw: dict, P, S: dict, T: dict) -> dict:
    wk = P.weeks[:P.Wh]
    tot = np.nansum(P.Y_raw, axis=0)
    info = {}

    # 1. total weekly units with events
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(wk, tot / 1e6, color=C1, lw=1.4)
    for d, lab in (("2016-04-16", "Ecuador earthquake"), ("2014-12-22", "Christmas dip"),
                   ("2015-12-21", ""), ("2016-12-19", ""), ("2013-12-23", "")):
        ax.axvline(pd.Timestamp(d), color=C2, ls="--", lw=.8)
        if lab:
            ax.text(pd.Timestamp(d), ax.get_ylim()[1] * .95, " " + lab, color=C2, fontsize=8, va="top")
    ax.set_ylabel("Units per week (millions)")
    ax.set_title("Total weekly unit sales, all 1,782 SKU-market series")
    _save(fig, "eda01_total_weekly_sales.png")

    # 2. store availability heat-map (share of families with sales)
    act = (P.Y_raw.reshape(P.n_store, P.n_fam, P.Wh) > 0).mean(1)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(act, aspect="auto", cmap="viridis", extent=[0, P.Wh, P.n_store, 0])
    ax.set_xticks(np.arange(0, P.Wh, 52))
    ax.set_xticklabels([str(wk[i].year) for i in np.arange(0, P.Wh, 52)])
    ax.set_yticks(np.arange(P.n_store)[::4] + .5)
    ax.set_yticklabels(P.stores["store_nbr"].values[::4])
    ax.set_ylabel("store_nbr"); ax.grid(False)
    ax.set_title("Share of product families with sales > 0, by store and week (dark = closed / not yet open)")
    fig.colorbar(im, ax=ax, shrink=.8)
    _save(fig, "eda02_store_activity_heatmap.png")

    # 3. series launches
    fp = P.first_pos[P.first_pos >= 0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].hist(fp, bins=60, color=C1)
    axs[0].set_title("Week of first sale per series (launch timing)")
    axs[0].set_xlabel("week index (0 = Jan 2013)")
    fl = (pd.DataFrame({"family": P.meta["family"], "fp": P.first_pos}).query("fp>=0")
          .groupby("family")["fp"].median().sort_values(ascending=False).head(8))
    axs[1].barh(fl.index[::-1], fl.values[::-1], color=C2)
    axs[1].set_title("Latest-launching families (median first-sale week)")
    _save(fig, "eda03_series_launches.png")

    # 4. promo coverage
    cov = np.nanmean(P.PROMO[:, :P.Wh] > 0, axis=0)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(wk, cov * 100, color=C3)
    ax.axvline(P.weeks[P.promo_start_week], color=C2, ls="--")
    ax.set_ylabel("% of series-weeks with a promotion")
    ax.set_title("Promotion coverage: tracking begins only on " + str(P.weeks[P.promo_start_week].date()))
    _save(fig, "eda04_promo_coverage.png")

    # 5. anomalies
    flag, _ = detect_anomalies(P)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(wk, flag.sum(0), width=6, color=C2)
    ax.set_title("Anomalous series-weeks per week (robust residual > 6 MAD)")
    _save(fig, "eda05_anomalies.png")

    # 6. seasonality profile
    L = pd.DataFrame(P.Y_raw.T, index=wk)
    fam_idx = P.meta["fam_idx"].values
    fig, ax = plt.subplots(figsize=(10, 3.8))
    top = (pd.Series(np.nansum(P.Y_raw, axis=1)).groupby(fam_idx).sum().sort_values(ascending=False).head(5).index)
    woy = pd.Series(wk.isocalendar().week.values, index=wk)
    for fi in top:
        s = pd.Series(np.nansum(P.Y_raw[fam_idx == fi], axis=0), index=wk)
        s = s[s.index >= "2014-01-01"]
        prof = s.groupby(woy[s.index].values).mean()
        ax.plot(prof.index, prof / prof.mean(), label=P.families[fi], lw=1.3)
    ax.set_xlabel("ISO week of year"); ax.set_ylabel("index (mean = 1)")
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Average seasonal profile of the five largest families (2014-2017)")
    _save(fig, "eda06_seasonality.png")

    # 7. concentration
    fam_tot = pd.Series(np.nansum(P.Y_raw[:, -52:], axis=1)).groupby(P.meta["family"].values).sum().sort_values(ascending=False)
    st_tot = pd.Series(np.nansum(P.Y_raw[:, -52:], axis=1)).groupby(P.meta["store_nbr"].values).sum().sort_values(ascending=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].bar(range(len(fam_tot)), fam_tot.cumsum() / fam_tot.sum() * 100, color=C1)
    axs[0].set_title("Cumulative % of last-52-week units by family")
    axs[0].set_xlabel("families ranked by volume")
    axs[1].bar(range(len(st_tot)), st_tot.cumsum() / st_tot.sum() * 100, color=C2)
    axs[1].set_title("Cumulative % by store")
    axs[1].set_xlabel("stores ranked by volume")
    _save(fig, "eda07_concentration.png")
    info["top_family_share"] = (fam_tot / fam_tot.sum()).head(5).round(3).to_dict()
    info["n_families_for_90pct"] = int((fam_tot.cumsum() / fam_tot.sum() < .9).sum() + 1)

    # 8. drivers: traffic & oil vs sales
    store_week = np.nansum(P.Y_raw.reshape(P.n_store, P.n_fam, P.Wh), axis=1)
    ok = ~np.isnan(P.TR) & (store_week > 0)
    corr_tr = np.corrcoef(np.log(store_week[ok]), np.log(P.TR[ok]))[0, 1]
    oil_w = P.OIL[:P.Wh]
    corr_oil = np.corrcoef(oil_w, np.log(tot))[0, 1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    pool = np.where(ok.ravel())[0]
    idx = np.random.default_rng(0).choice(pool, min(4000, len(pool)), replace=False)
    axs[0].scatter(np.log(P.TR.ravel()[idx]), np.log(store_week.ravel()[idx]), s=3, alpha=.3, color=C1)
    axs[0].set_xlabel("log weekly transactions"); axs[0].set_ylabel("log weekly store units")
    axs[0].set_title(f"Traffic vs sales (store-weeks), corr={corr_tr:.2f}")
    axs[1].scatter(oil_w, tot / 1e6, s=6, color=C2)
    axs[1].set_xlabel("WTI oil price ($)"); axs[1].set_ylabel("total weekly units (M)")
    axs[1].set_title(f"Oil vs total sales, corr={corr_oil:.2f}")
    _save(fig, "eda08_traffic_oil.png")
    info.update(corr_traffic=float(corr_tr), corr_oil=float(corr_oil))

    # 9. intermittency
    ic = T["intermittency_classes"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(ic.iloc[:, 0][::-1], ic["n_series"][::-1], color=C1)
    ax.set_title("Demand pattern of series (Syntetos-Boylan, last 2 years)")
    _save(fig, "eda09_intermittency.png")

    # 10. promo response (binned)
    Lg = P.L
    with np.errstate(all="ignore"):
        dev = Lg[:, 60:P.Wh] - pd.DataFrame(Lg.T).rolling(8, min_periods=4).mean().shift(1).values.T[:, 60:P.Wh]
        pr = np.log1p(P.PROMO[:, 60:P.Wh]) - np.log1p(pd.DataFrame(P.PROMO[:, :P.Wh].T).rolling(13, min_periods=6)
                                                       .mean().shift(1).values.T[:, 60:P.Wh])
    ok2 = ~np.isnan(dev) & ~np.isnan(pr)
    bins = np.linspace(-1.5, 1.5, 13)
    b = np.digitize(pr[ok2], bins)
    rr = pd.DataFrame({"b": b, "dev": dev[ok2]}).groupby("b")["dev"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ctr = [(bins[i - 1] + bins[i]) / 2 if 0 < i < len(bins) else np.nan for i in rr.index]
    ax.plot(ctr, rr["mean"], "o-", color=C3)
    ax.set_xlabel("promo intensity vs own trailing 13-week average (log)")
    ax.set_ylabel("sales vs own trailing 8-week average (log)")
    ax.set_title("Raw association: heavier promotion weeks sell more")
    _save(fig, "eda10_promo_response.png")
    info["promo_slope"] = float(np.polyfit(np.array(ctr)[~np.isnan(ctr)], rr["mean"].values[~np.isnan(ctr)], 1)[0])
    return info


def write_dq_report(S: dict, T: dict, info: dict, path=None) -> str:
    path = path or config.REP_DIR / "01_data_quality_and_eda_report.md"
    cl = T["store_closures"]
    anom = T["anomalies_top_weeks"].head(6)
    ic = T["intermittency_classes"]
    lines = f"""# 01 - Data-quality & EDA report

*Dataset:* {S['train_rows']:,} daily rows, {S['train_date_range'][0]} to {S['train_date_range'][1]}, {S['n_stores']} stores x {S['n_families']} product families = **{S['n_series']:,} SKU-market series** (market = store, SKU = product family; see README "Dataset mapping").

## 1. Structural checks

| Check | Result |
|---|---|
| Duplicate (date, store, family) keys - train / test | {S['duplicate_keys_train']} / {S['duplicate_keys_test']} |
| Duplicate (date, store) keys in transactions | {S['duplicate_keys_transactions']} |
| Null cells in train | {S['null_cells_train']} |
| Negative sales rows | {S['negative_sales_rows']} |
| Rows missing within observed days (should be 0) | {S['missing_rows_within_present_days']} |
| Calendar days absent from the data | {len(S['missing_calendar_days'])} - {', '.join(S['missing_calendar_days'])} (every 25 December: stores closed) |
| Share of rows with fractional units | {S['share_rows_fractional_units']:.1%} (weight-sold families such as PRODUCE, MEATS, POULTRY...) |
| Share of rows with zero sales | {S['zero_sales_row_share']:.1%} |
| Oil prices missing | {S['oil_missing_values']} of {S['oil_rows']} days (weekends/holidays) -> forward/back-filled |
| Store-days with sales but no transactions record | {S['store_days_with_sales_but_no_transactions']} |
| Holiday calendar ends | {S['holiday_calendar_end']} (covers the whole 13-week horizon) |

**Treatment.** The 25-Dec gap is kept (weeks have 6 days; a calendar feature ``ndays`` and ``xmas`` flag let the model learn the closure).
The first partial week (Jan 2013) and the last partial week (14-15 Aug 2017) are dropped from the history. Transferred holidays are
re-dated using the *Transfer* rows. Oil is only ever used as a lagged, origin-dated feature, so no future oil assumption is needed.

## 2. Weekly panel & missing weeks

History = **{S['n_weeks_history']} complete weeks** ({S['history_weeks'][0]} to {S['history_weeks'][1]} week-starts). Weeks with fewer than 7 days: {', '.join(S['weeks_with_missing_days'])}.
No full week is missing for any store-family row (the table is a complete panel); gaps are instead *business* gaps - closures, launches, delistings.

## 3. Launches, delistings and closures (affects forecastability)

* **{S['series_never_sold']} series never sold** in the whole history (forecast = 0 by rule).
* **{S['series_launched_after_week8']} series launched after the first two months** (new store or new family; e.g. BOOKS, BABY CARE, store 52). Pre-launch zeros are treated as *missing*, not demand.
* **{S['series_delisted_trailing_zero_8w']} series show >= 8 trailing zero weeks** -> treated as delisted (forecast 0).
* **Reporting gaps:** {S['gap_spells']:,} spells (in {S['gap_series_affected']:,} series) of 8+ consecutive zero weeks inside a series' life after which sales *resume*
  = {S['gap_series_weeks']:,} series-weeks ({S['gap_share_of_active']:.1%} of active data). By start year: {S['gap_series_weeks_by_year']}.
  Whole families vanish together (e.g. PRODUCE, HOME CARE, CELEBRATION in 2014-15), so these are data-recording gaps, not zero demand.
  **Treatment:** such weeks are removed from *training targets* (only for gaps already over at the fold origin, so no look-ahead) and from the scored universe
  (they are {S['gap_share_in_cv_windows']:.1%} of the CV windows); a sensitivity table reports accuracy with them included. Features still see the raw zeros, exactly as production would.
* **{S['stores_opened_after_start']} stores opened after Jan 2013** and {int(cl['weeks_closed'].sum()) if len(cl) else 0} store-weeks are whole-week closures
  (all families zero while the store is otherwise open): these weeks are masked from training and excluded from scoring because closures are unpredictable.

Median first-sale week by family and store openings are in `outputs/tables/dq_family_launch_week_median.csv`, `dq_store_openings.csv`, `dq_store_closures.csv`.

![](../figures/eda02_store_activity_heatmap.png)
![](../figures/eda03_series_launches.png)

## 4. Unusual sales movements

A robust detector (residual to a 9-week centred median, > 6 x MAD, > 0.8 log-points, > 20 units) flags **{S['anomalous_series_weeks']:,} series-weeks ({S['anomalous_share_of_active']:.2%} of active data)**.
Largest clusters: {', '.join(f"{r.week.date()} ({r.n_flagged})" for r in anom.itertuples())}.
The Feb-Mar 2014 clusters are the *return* of families after a reporting gap (levels jump back, flagged as spikes), the Dec-2014 cluster is the
Christmas/new-year effect, and April 2016 shows the earthquake. Genuine demand shocks (holidays, earthquake) are explainable by calendar features
and are **kept** in the targets (no winsorising); closures and reporting gaps are masked as described above.

![](../figures/eda01_total_weekly_sales.png)
![](../figures/eda05_anomalies.png)

## 5. Promotions and other drivers

* Promotion tracking **starts {S['promo_tracking_starts_week']}**; earlier values are all zero and are encoded as *missing* (not "no promo"). Afterwards {S['promo_share_series_weeks_active']:.0%} of series-weeks carry some promotion.
* Raw association: heavier promo weeks sell more (slope {info['promo_slope']:.2f} log-points per log-point of promo intensity).
* Store traffic is strongly tied to sales (corr of logs {info['corr_traffic']:.2f}) but is *not known in the future* -> lag-only.
* Oil price vs total weekly sales correlation {info['corr_oil']:.2f} (a trend artefact as much as a driver) -> lag-only, tested in the ablation.
* Concentration: top families {', '.join(f'{k} {v:.0%}' for k, v in info['top_family_share'].items())}; {info['n_families_for_90pct']} of 33 families make 90% of volume.

![](../figures/eda04_promo_coverage.png)
![](../figures/eda06_seasonality.png)
![](../figures/eda07_concentration.png)
![](../figures/eda08_traffic_oil.png)
![](../figures/eda10_promo_response.png)

## 6. Demand patterns (sparse / cold-start awareness)

{ic.to_markdown(index=False)}

Intermittent and sparse series are a minority by volume, which motivates (a) a log-scale model, (b) a peer-group cold-start fallback and (c) volume-weighted training.

![](../figures/eda09_intermittency.png)

## 7. Test file

`test.csv` covers {S['test_date_range'][0]} to {S['test_date_range'][1]} for {S['test_series']:,} series ({S['test_series_not_in_train']} not in train). It has **no sales**
(a held-out Kaggle-style file), so it cannot score the models; it provides the *promotion plan* for the first ~2.6 forecast weeks. Beyond that the
promotion plan is assumed (carry-forward of the last 28 days).
"""
    path.write_text(lines)
    return lines
