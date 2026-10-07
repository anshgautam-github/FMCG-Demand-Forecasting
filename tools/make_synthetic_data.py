"""Create a SMALL synthetic dataset with the same schema as the provided files.

Used by the unit tests / smoke test so the whole pipeline can be exercised in seconds without the
120 MB train.csv.  Includes seasonality, promotions (tracked only after week ~10), a store opening,
a product launch, a reporting gap, a Dec-25 closure and holidays.

    python tools/make_synthetic_data.py data/synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(out: Path, n_stores=6, families=("GROCERY I", "BEVERAGES", "CLEANING", "BOOKS", "PRODUCE"),
         start="2014-01-01", days=1100, test_days=16, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    dates = dates[~((dates.month == 12) & (dates.day == 25))]
    all_dates = pd.date_range(start, periods=days + test_days)
    stores = pd.DataFrame({"store_nbr": np.arange(1, n_stores + 1),
                           "city": ["Quito", "Quito", "Guayaquil", "Cuenca", "Quito", "Guayaquil"][:n_stores],
                           "state": ["Pichincha", "Pichincha", "Guayas", "Azuay", "Pichincha", "Guayas"][:n_stores],
                           "type": list("DDBCAB")[:n_stores], "cluster": [13, 13, 8, 5, 14, 8][:n_stores]})
    rows, test_rows, tx = [], [], []
    for s in stores.store_nbr:
        scale_s = rng.uniform(.6, 1.6)
        for f_i, f in enumerate(families):
            base = rng.uniform(20, 400) * scale_s * (0.05 if f == "BOOKS" else 1)
            promo_lvl = rng.uniform(0, 8)
            for d in all_dates:
                t = (d - all_dates[0]).days
                seas = 1 + .25 * np.sin(2 * np.pi * d.dayofyear / 365) + (.15 if d.dayofweek >= 5 else 0)
                promo = int(rng.poisson(promo_lvl)) if t > 70 else 0
                mu = base * seas * (1 + .02 * promo) * (1 + t / 4000)
                y = rng.poisson(max(mu, 0.1)) if f != "PRODUCE" else float(np.round(mu * rng.lognormal(0, .15), 2))
                if f == "BOOKS" and t < 400:
                    y = 0
                if s == n_stores and t < 300:
                    y = 0                                    # store opens late
                if f == "PRODUCE" and 200 < t < 330:
                    y = 0                                    # reporting gap
                if d in dates:
                    rows.append((d, s, f, y, promo))
                if d >= dates[-1] + pd.Timedelta(days=1) and d.strftime("%m-%d") != "12-25":
                    test_rows.append((d, s, f, promo))
        for d in dates:
            tx.append((d, s, int(rng.integers(900, 2500) * scale_s)))
    tr = pd.DataFrame(rows, columns=["date", "store_nbr", "family", "sales", "onpromotion"])
    tr.insert(0, "id", np.arange(len(tr)))
    te = pd.DataFrame(test_rows, columns=["date", "store_nbr", "family", "onpromotion"])
    te.insert(0, "id", np.arange(len(tr), len(tr) + len(te)))
    tr = tr[tr["date"] <= dates[-1]]
    oil = pd.DataFrame({"date": all_dates, "dcoilwtico": 60 + np.cumsum(rng.normal(0, .8, len(all_dates)))})
    oil.loc[::9, "dcoilwtico"] = np.nan
    hol = pd.DataFrame([
        (pd.Timestamp(f"{y}-{m:02d}-{d:02d}"), "Holiday", "National", "Ecuador", n, False)
        for y in (2014, 2015, 2016, 2017) for m, d, n in ((1, 1, "Primer dia del ano"), (5, 1, "Dia del Trabajo"),
                                                           (8, 10, "Primer Grito"), (11, 2, "Difuntos"), (12, 25, "Navidad"))],
        columns=["date", "type", "locale", "locale_name", "description", "transferred"])
    stores.to_csv(out / "stores.csv", index=False)
    tr.to_csv(out / "train.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    pd.DataFrame(tx, columns=["date", "store_nbr", "transactions"]).to_csv(out / "transactions.csv", index=False)
    oil.to_csv(out / "oil.csv", index=False)
    hol.to_csv(out / "holidays_events.csv", index=False)
    return out


if __name__ == "__main__":
    make(Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"))
    print("synthetic data written")
