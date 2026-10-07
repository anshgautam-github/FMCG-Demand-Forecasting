"""Forecast accuracy metrics (computed on raw units, never on the log scale only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(y - f).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, f):
    """Signed volume bias: (sum f - sum y) / sum y.  Negative = under-forecast."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f.sum() - y.sum()) / max(np.abs(y).sum(), 1e-9))


def rmse(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def rmsle(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((np.log1p(np.clip(f, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))


def summarize(y, f) -> dict:
    return dict(WMAPE=wmape(y, f), RMSE=rmse(y, f), Bias=bias(y, f), RMSLE=rmsle(y, f), n=int(len(y)))


def table_by(df: pd.DataFrame, models: list[str], by, ycol: str = "y") -> pd.DataFrame:
    """Metrics per group for every model; returns a long tidy frame."""
    rows = []
    for key, g in df.groupby(by, observed=True):
        for m in models:
            rows.append({**(dict(zip(by, key if isinstance(key, tuple) else (key,))) if isinstance(by, list)
                            else {by: key}), "model": m, **summarize(g[ycol], g[m])})
    return pd.DataFrame(rows)


def aggregate_level_wmape(df: pd.DataFrame, models: list[str], level: str) -> pd.DataFrame:
    """Bottom-up aggregation accuracy: sum forecasts and actuals to a coarser level first.
    level in {'total','family','store'}.  Forecasts are coherent by construction (bottom-up)."""
    keys = {"total": ["fold", "week"], "family": ["fold", "week", "family"],
            "store": ["fold", "week", "store_nbr"]}[level]
    agg = df.groupby(keys, observed=True)[["y"] + models].sum().reset_index()
    return pd.DataFrame([{"level": level, "model": m, **summarize(agg["y"], agg[m])} for m in models])
