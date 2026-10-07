"""Data-quality checks (task 1): missing weeks/dates, duplicates, anomalies, launches/delistings."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from .weekly_dataset import Panel


def detect_anomalies(P: Panel, win: int = 9, k: float = 6.0, min_abs: float = 0.8, min_units: float = 20):
    """Robust level-shift/spike detector on log1p weekly units.

    residual = L - centred rolling median(9w); flag if |residual| > k * 1.4826 * MAD(series residual)
    and |residual| > min_abs (log points) and units > min_units.  Returns a boolean [S, Wh] array."""
    L = pd.DataFrame(P.L.T)
    med = L.rolling(win, center=True, min_periods=3).median()
    res = (L - med).values.T
    mad = np.nanmedian(np.abs(res - np.nanmedian(res, axis=1, keepdims=True)), axis=1)
    scale = np.maximum(1.4826 * mad, 0.05)
    flag = (np.abs(res) > k * scale[:, None]) & (np.abs(res) > min_abs) & (np.nan_to_num(P.Y, nan=0) > min_units)
    return flag, res


def intermittency_class(P: Panel, last_weeks: int = 104) -> pd.DataFrame:
    """Syntetos-Boylan classification on the last two years of *active* weekly demand."""
    Y = P.Y[:, -last_weeks:]
    rows = []
    for s in range(P.S):
        y = Y[s][~np.isnan(Y[s])]
        nz = y[y > 0]
        if len(nz) < 3:
            rows.append((s, np.nan, np.nan, "sparse (<3 active weeks)"))
            continue
        adi = len(y) / len(nz)
        cv2 = (nz.std() / nz.mean()) ** 2
        cls = ("smooth" if adi < 1.32 and cv2 < 0.49 else "erratic" if adi < 1.32 else
               "intermittent" if cv2 < 0.49 else "lumpy")
        rows.append((s, adi, cv2, cls))
    d = pd.DataFrame(rows, columns=["series_id", "ADI", "CV2", "class"])
    return d


def run_quality_checks(raw: dict, P: Panel) -> tuple[dict, dict[str, pd.DataFrame]]:
    tr, te, tx, oil, hol, st = (raw[k] for k in ("train", "test", "transactions", "oil", "holidays", "stores"))
    S: dict = {}
    T: dict[str, pd.DataFrame] = {}

    # --- structure / completeness -------------------------------------------------------------
    all_days = pd.date_range(tr["date"].min(), tr["date"].max())
    missing_days = all_days.difference(pd.DatetimeIndex(tr["date"].unique()))
    S["train_rows"] = int(len(tr))
    S["train_date_range"] = [str(tr["date"].min().date()), str(tr["date"].max().date())]
    S["n_stores"], S["n_families"] = int(tr["store_nbr"].nunique()), int(tr["family"].nunique())
    S["n_series"] = P.S
    S["expected_rows_if_complete"] = int((len(all_days) - len(missing_days)) * P.S)
    S["missing_rows_within_present_days"] = S["expected_rows_if_complete"] - S["train_rows"]
    S["missing_calendar_days"] = [str(d.date()) for d in missing_days]
    S["duplicate_keys_train"] = int(tr.duplicated(["date", "store_nbr", "family"]).sum())
    S["duplicate_keys_transactions"] = int(tx.duplicated(["date", "store_nbr"]).sum())
    S["duplicate_keys_test"] = int(te.duplicated(["date", "store_nbr", "family"]).sum())
    S["null_cells_train"] = int(tr.isna().sum().sum())
    S["negative_sales_rows"] = int((tr["sales"] < 0).sum())
    frac = (tr["sales"] % 1 != 0)
    S["share_rows_fractional_units"] = float(frac.mean())
    T["fractional_units_by_family"] = (frac.groupby(tr["family"], observed=True).mean()
                                       .sort_values(ascending=False).rename("share_fractional").reset_index())
    S["zero_sales_row_share"] = float((tr["sales"] == 0).mean())

    # --- week-level view ------------------------------------------------------------------------
    S["n_weeks_history"] = P.Wh
    S["history_weeks"] = [str(P.weeks[0].date()), str(P.weeks[P.Wh - 1].date())]
    nd = P.CAL["ndays"][: P.Wh]
    S["weeks_with_missing_days"] = [str(P.weeks[i].date()) for i in np.where(nd < 7)[0]]
    S["partial_weeks_dropped"] = ["first partial week (2013-01-01..06)",
                                  f"last partial week ({(P.weeks[P.Wh]).date()}..{P.last_date.date()})"]

    # --- store closures / openings -----------------------------------------------------------------
    clos = []
    for s_i in range(P.n_store):
        w = P.closed[s_i]
        if w.any():
            idx = np.where(w)[0]
            # contiguous runs
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for r in runs:
                clos.append(dict(store_nbr=int(P.stores["store_nbr"][s_i]), first_week=str(P.weeks[r[0]].date()),
                                 weeks_closed=len(r)))
    T["store_closures"] = pd.DataFrame(clos)
    S["store_closure_weeks_total"] = int(P.closed.sum())
    store_first = np.array([np.argmax((P.Y_raw.reshape(P.n_store, P.n_fam, P.Wh).sum(1)[i]) > 0)
                            for i in range(P.n_store)])
    late_open = [(int(P.stores["store_nbr"][i]), str(P.weeks[w].date())) for i, w in enumerate(store_first) if w > 8]
    T["store_openings"] = pd.DataFrame(late_open, columns=["store_nbr", "first_sales_week"])
    S["stores_opened_after_start"] = int(len(late_open))

    # --- launches / delistings ----------------------------------------------------------------------
    fp = P.first_pos
    never = fp < 0
    late = (fp > 8)
    tail0 = np.all(P.Y[:, -config.DISCONTINUE_ZERO_WEEKS:] == 0, axis=1) & ~never
    S["series_never_sold"] = int(never.sum())
    S["series_launched_after_week8"] = int((late & ~never).sum())
    S["series_delisted_trailing_zero_8w"] = int(tail0.sum())
    fam_launch = (pd.DataFrame({"family": P.meta["family"], "first_week": np.where(never, -1, fp)})
                  .query("first_week >= 0").groupby("family")["first_week"].median())
    T["family_launch_week_median"] = fam_launch.map(lambda i: str(P.weeks[int(i)].date())).rename("median_first_sale_week").reset_index()
    # --- reporting gaps: long zero spells inside a series' life after which sales resume ------------
    sp = P.gap_spells.copy()
    sp["family"] = P.meta["family"].values[sp["series"]]
    sp["store_nbr"] = P.meta["store_nbr"].values[sp["series"]]
    sp["length_w"] = sp["end"] - sp["start"] + 1
    sp["start_week"] = P.weeks[sp["start"].values]
    S["gap_spells"] = int(len(sp))
    S["gap_series_weeks"] = int(sp["length_w"].sum())
    S["gap_share_of_active"] = float(sp["length_w"].sum() / max(np.sum(~np.isnan(P.Y)), 1))
    S["gap_series_affected"] = int(sp["series"].nunique())
    S["gap_series_weeks_by_year"] = {int(k): int(v) for k, v in
                                     sp.groupby(sp["start_week"].dt.year)["length_w"].sum().items()}
    S["gap_share_in_cv_windows"] = float(P.gap_mask(P.Wh - 1)[:, P.Wh - 52:].sum() /
                                         max(np.sum(~np.isnan(P.Y[:, P.Wh - 52:])), 1))
    T["reporting_gaps_by_family"] = (sp.groupby("family")["length_w"].agg(["count", "sum"]).rename(
        columns={"count": "n_spells", "sum": "series_weeks"}).sort_values("series_weeks", ascending=False).reset_index())
    T["reporting_gap_spells"] = sp.drop(columns=["series"]).sort_values("length_w", ascending=False)
    S["series_with_8plus_zero_week_spell"] = S["gap_series_affected"]

    # --- anomalies ------------------------------------------------------------------------------------
    flag, res = detect_anomalies(P)
    S["anomalous_series_weeks"] = int(flag.sum())
    S["anomalous_share_of_active"] = float(flag.sum() / max(np.sum(~np.isnan(P.Y)), 1))
    by_week = pd.DataFrame({"week": P.weeks[:P.Wh], "n_flagged": flag.sum(0)})
    T["anomalies_by_week"] = by_week
    T["anomalies_top_weeks"] = by_week.sort_values("n_flagged", ascending=False).head(12)
    T["anomalies_by_family"] = (pd.DataFrame({"family": P.meta["family"], "n": flag.sum(1)})
                                .groupby("family")["n"].sum().sort_values(ascending=False).reset_index())

    # --- promotions ---------------------------------------------------------------------------------------
    S["promo_tracking_starts_week"] = str(P.weeks[P.promo_start_week].date())
    S["promo_share_series_weeks_active"] = float(np.nanmean(P.PROMO[:, P.promo_start_week:P.Wh] > 0))

    # --- transactions / oil / holidays ----------------------------------------------------------------------
    sd = tr.groupby(["date", "store_nbr"], observed=True)["sales"].sum().reset_index()
    sd = sd[sd["sales"] > 0]
    mg = sd.merge(tx, on=["date", "store_nbr"], how="left")
    S["store_days_with_sales_but_no_transactions"] = int(mg["transactions"].isna().sum())
    S["oil_missing_values"] = int(oil["dcoilwtico"].isna().sum())
    S["oil_rows"] = int(len(oil))
    S["holiday_rows"] = int(len(hol))
    S["holidays_transferred_true"] = int(hol["transferred"].astype(bool).sum())
    S["holiday_types"] = hol["type"].value_counts().to_dict()
    S["holiday_calendar_end"] = str(hol["date"].max().date())

    # --- test file ---------------------------------------------------------------------------------------------
    S["test_date_range"] = [str(te["date"].min().date()), str(te["date"].max().date())]
    S["test_series"] = int(te.groupby(["store_nbr", "family"], observed=True).ngroups)
    S["test_series_not_in_train"] = int(len(set(zip(te.store_nbr, te.family.astype(str))) - set(zip(tr.store_nbr, tr.family.astype(str)))))
    S["test_days"] = int(te["date"].nunique())
    S["test_covers_weeks_of_horizon"] = "weeks 1-2 fully, week 3 partially (4 of 7 days); weeks 4-13 unknown"

    cls = intermittency_class(P)
    T["intermittency_classes"] = cls["class"].value_counts().rename("n_series").reset_index()
    T["intermittency_detail"] = cls.merge(P.meta[["series_id", "store_nbr", "family"]], on="series_id")

    for k, v in T.items():
        v.to_csv(config.TAB_DIR / f"dq_{k}.csv", index=False)
    (config.TAB_DIR / "dq_summary.json").write_text(json.dumps(S, indent=2, default=str))
    return S, T
