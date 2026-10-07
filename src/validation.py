"""Time-aware validation: expanding-window, rolling-origin cross-validation.

Fold k forecasts weeks o_k+1 .. o_k+H using ONLY information up to the origin week o_k, and the
model for fold k is trained on rows whose *target* week is <= o_k (purged).  Random splits are
never used.  With N_FOLDS=4 and H=13 the folds tile the final 52 observed weeks, so every season
of the year (incl. Christmas and Carnival) is evaluated out-of-sample once.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .weekly_dataset import Panel


@dataclass(frozen=True)
class Fold:
    k: int
    origin: int           # last observed week index
    first_target: int
    last_target: int


def make_folds(P: Panel, n_folds: int, horizon: int) -> list[Fold]:
    folds = []
    for i, k in enumerate(range(n_folds, 0, -1), start=1):
        o = P.Wh - 1 - horizon * k
        folds.append(Fold(i, o, o + 1, o + horizon))
    return folds


def describe_folds(P: Panel, folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fold": f.k, "origin_week_start": P.weeks[f.origin].date(),
        "target_from": P.weeks[f.first_target].date(), "target_to": P.weeks[f.last_target].date(),
        "train_target_weeks_<=": P.weeks[f.origin].date(),
    } for f in folds])
