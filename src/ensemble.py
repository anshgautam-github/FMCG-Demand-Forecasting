"""Hybrid (statistical + ML) ensemble and cold-start strategy selection.

Weights are chosen on the *earlier* folds only (1..K-1) so the last fold remains an honest,
untouched check of the ensemble.  Blends are arithmetic (volume-preserving)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

PARTNERS = ["snaive_growth", "damped_holt", "naive_ma4"]
COLD_CANDIDATES = ["gbm", "coldstart_peer", "gbm_cold_blend", "snaive_growth", "damped_holt"]


def blend(a, b, w):
    return w * np.asarray(a) + (1 - w) * np.asarray(b)


def select_blend(df: pd.DataFrame, select_folds) -> tuple[str, float, pd.DataFrame]:
    d = df[df["eval"] & ~df["cold"] & df["fold"].isin(select_folds)]
    rows = []
    for p in PARTNERS:
        for w in np.round(np.arange(0, 1.01, 0.1), 2):
            f = blend(d["gbm"], d[p], w)
            rows.append(dict(partner=p, w_gbm=w, WMAPE=metrics.wmape(d["y"], f), RMSLE=metrics.rmsle(d["y"], f)))
    t = pd.DataFrame(rows)
    best = t.sort_values(["WMAPE", "RMSLE"]).iloc[0]
    return str(best["partner"]), float(best["w_gbm"]), t


def select_cold(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    c = df[df["eval"] & df["cold"]]
    t = pd.DataFrame([dict(strategy=m, **metrics.summarize(c["y"], c[m])) for m in COLD_CANDIDATES])
    return str(t.sort_values("WMAPE").iloc[0]["strategy"]), t


def add_ensemble(df: pd.DataFrame, partner: str, w: float, cold_col: str) -> pd.DataFrame:
    df = df.copy()
    df["ensemble"] = blend(df["gbm"], df[partner], w)
    df.loc[df["cold"], "ensemble"] = df.loc[df["cold"], cold_col]
    return df
