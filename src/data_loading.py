"""Load the raw CSV files.

The provided dataset is the *Store Sales* (Favorita-style) retail dataset.  It is mapped onto
the FreshBasket brief as follows (see README, "Dataset mapping"):

    market_id  -> store_nbr          (54 retail stores)
    sku_id     -> family            (33 product families)
    units      -> sales             (daily, aggregated to weeks)
    promo      -> onpromotion       (# items of the family on promotion in the store that day)
    traffic    -> transactions      (store-day receipts; used lag-only)
    macro      -> oil               (daily WTI price; used lag-only, proxy for external drivers)
    calendar   -> holidays_events   (national / regional / local holidays and events)

File names are matched by suffix so that upload-prefixed names (e.g. ``1791_train.csv``) work.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("train", "test", "stores", "transactions", "oil", "holidays_events")


def _find(raw_dir: Path, name: str) -> Path:
    exact = raw_dir / f"{name}.csv"
    if exact.exists():
        return exact
    cands = sorted(raw_dir.glob(f"*{name}.csv"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(
            f"Could not find '{name}.csv' (or '*{name}.csv') in {raw_dir}. "
            "Place the dataset files in data/raw/ or set FMCG_RAW_DIR."
        )
    return cands[-1]


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    d: dict[str, pd.DataFrame] = {}
    d["train"] = pd.read_csv(
        _find(raw_dir, "train"), parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", "sales": "float32", "onpromotion": "int32"},
    )
    d["test"] = pd.read_csv(
        _find(raw_dir, "test"), parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"},
    )
    d["stores"] = pd.read_csv(_find(raw_dir, "stores"))
    d["transactions"] = pd.read_csv(_find(raw_dir, "transactions"), parse_dates=["date"])
    d["oil"] = pd.read_csv(_find(raw_dir, "oil"), parse_dates=["date"])
    d["holidays"] = pd.read_csv(_find(raw_dir, "holidays_events"), parse_dates=["date"])
    return d
