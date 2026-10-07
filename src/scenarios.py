"""Business-scenario simulator (task 9).

Scenarios re-run the *same* trained global model with modified known-in-advance inputs and report
the demand difference against the baseline plan.  Results are model-implied associations learnt
from history (promotions were not randomised), not causal experiments - see the report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import gbm_forecast
from . import models_stat as ms


def promo_cap(P, o):
    """Never extrapolate promotions beyond 1.5x the series' own 52-week maximum."""
    with np.errstate(all="ignore"):
        mx = np.nanmax(P.PROMO[:, max(o - 51, 0):o + 1], axis=1)
    return np.nan_to_num(mx, nan=0.0) * 1.5


def scale_promo(P, o, mult, fam_mask=None) -> np.ndarray:
    """Multiply the planned items-on-promotion in the forecast window (optionally for some series only)."""
    pr = P.PROMO.copy()
    fut = pr[:, o + 1:o + 1 + P.H]
    cap = np.maximum(promo_cap(P, o)[:, None], fut)          # cap, but never below the baseline plan
    new = fut * mult if mult <= 1 else np.minimum(fut * mult, cap)
    sel = np.ones(P.S, bool) if fam_mask is None else fam_mask
    out = fut.copy()
    out[sel] = new[sel]
    pr[:, o + 1:o + 1 + P.H] = out
    return pr


def run_scenarios(P, gbm, o, base_final: pd.DataFrame | None = None):
    cold = ms.coldstart_peer(P, o)
    base = gbm_forecast(P, gbm, o, cold=cold)
    meta = P.meta
    res_rows, fam_rows, weekly = [], [], []

    def record(name, F, desc):
        d = F - base
        res_rows.append(dict(scenario=name, description=desc, base_units=base.sum(), scenario_units=F.sum(),
                             delta_units=d.sum(), delta_pct=100 * d.sum() / base.sum(),
                             share_series_up=float(np.mean(d.sum(1) > 1e-6)),
                             share_series_down=float(np.mean(d.sum(1) < -1e-6))))
        f = pd.DataFrame({"family": meta["family"].values, "type": meta["type"].values,
                          "base": base.sum(1), "scn": F.sum(1)})
        for fam, g in f.groupby("family"):
            fam_rows.append(dict(scenario=name, family=fam, base_units=g.base.sum(), scenario_units=g.scn.sum(),
                                 delta_units=g.scn.sum() - g.base.sum(),
                                 delta_pct=100 * (g.scn.sum() - g.base.sum()) / max(g.base.sum(), 1e-9)))
        for h in range(P.H):
            weekly.append(dict(scenario=name, week=P.weeks[o + 1 + h], base_units=base[:, h].sum(), scenario_units=F[:, h].sum()))

    # S1: baseline vs promotion decomposition (promotions withdrawn)
    F0 = gbm_forecast(P, gbm, o, promo=scale_promo(P, o, 0.0), cold=cold)
    record("S0 No promotions (baseline decomposition)", F0, "all promotions set to 0 in the 13 forecast weeks")
    for mult, label in ((1.25, "S1 Promo intensity +25%"), (1.5, "S2 Promo intensity +50%"), (2.0, "S3 Promo intensity x2")):
        F = gbm_forecast(P, gbm, o, promo=scale_promo(P, o, mult), cold=cold)
        record(label, F, f"items-on-promotion x{mult} (capped at 1.5x series 52-week max)")
    # S4: footfall shock
    F = gbm_forecast(P, gbm, o, trans_shift=float(np.log(0.9)), cold=cold)
    record("S4 Store traffic -10%", F, "transactions in the last 13 observed weeks lowered 10% (footfall shock)")

    # promo response curves per family
    curve = []
    fams = np.array(P.families)
    for fi, fam in enumerate(fams):
        fm = (meta["fam_idx"].values == fi)
        if base[fm].sum() <= 0:
            continue
        for mult in (0.0, 0.5, 1.0, 1.5, 2.0):
            F = base if mult == 1.0 else gbm_forecast(P, gbm, o, promo=scale_promo(P, o, mult, fm), cold=cold)
            curve.append(dict(family=fam, multiplier=mult, units=float(F[fm].sum()),
                              units_vs_base_pct=100 * (F[fm].sum() / base[fm].sum() - 1)))
    curve = pd.DataFrame(curve)
    return (pd.DataFrame(res_rows), pd.DataFrame(fam_rows), pd.DataFrame(weekly), curve, base)
