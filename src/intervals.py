"""Prediction intervals via (Mondrian) split-conformal calibration on log-residuals.

Residuals r = log1p(actual) - log1p(forecast) are collected out-of-sample from the rolling-origin
folds, grouped by (horizon bin x series-volume tier), and their empirical quantiles give
multiplicative bands:  [expm1(log1p(f)+q_lo), expm1(log1p(f)+q_hi)].  Marginal coverage holds
under exchangeability of residuals within a cell; coverage is *checked* on a held-out fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H_BINS = [(1, 4), (5, 9), (10, 13)]


def _hbin(h):
    h = np.asarray(h)
    out = np.zeros(len(h), int)
    for i, (a, b) in enumerate(H_BINS):
        out[(h >= a) & (h <= b)] = i
    return out


class ConformalIntervals:
    def __init__(self, levels=(0.5, 0.9), min_cell=150, n_tiers=4):
        self.levels, self.min_cell, self.n_tiers = levels, min_cell, n_tiers

    def _tier(self, lvl):
        return np.clip(np.searchsorted(self.edges_, lvl, side="right") - 1, 0, self.n_tiers - 1)

    def fit(self, h, lvl, y, f):
        h, lvl, y, f = map(np.asarray, (h, lvl, y, f))
        ok = ~np.isnan(lvl)
        h, lvl, y, f = h[ok], lvl[ok], y[ok], f[ok]
        self.edges_ = np.quantile(lvl, np.linspace(0, 1, self.n_tiers + 1))
        self.edges_[0], self.edges_[-1] = -np.inf, np.inf
        r = np.log1p(np.clip(y, 0, None)) - np.log1p(np.clip(f, 0, None))
        hb, tb = _hbin(h), self._tier(lvl)
        self.q_ = {}
        for lev in self.levels:
            qs = ((1 - lev) / 2, 1 - (1 - lev) / 2)
            glob = np.quantile(r, qs)
            for a in range(len(H_BINS)):
                pool = r[hb == a]
                pq = np.quantile(pool, qs) if len(pool) >= self.min_cell else glob
                for b in range(self.n_tiers):
                    cell = r[(hb == a) & (tb == b)]
                    self.q_[(lev, a, b)] = np.quantile(cell, qs) if len(cell) >= self.min_cell else pq
        return self

    def predict(self, h, lvl, f):
        h, lvl, f = map(np.asarray, (h, lvl, f))
        hb, tb = _hbin(h), self._tier(np.where(np.isnan(lvl), self.edges_[1], lvl))
        out = {}
        lf = np.log1p(np.clip(f, 0, None))
        for lev in self.levels:
            lo = np.empty(len(f)); hi = np.empty(len(f))
            for i in range(len(f)):
                ql, qh = self.q_[(lev, hb[i], tb[i])]
                lo[i], hi[i] = np.expm1(lf[i] + ql), np.expm1(lf[i] + qh)
            pct = int(round(lev * 100))
            out[f"lower_{pct}"] = np.clip(lo, 0, None)
            out[f"upper_{pct}"] = np.maximum(hi, np.clip(f, 0, None))
        return pd.DataFrame(out)


def coverage(y, lo, hi):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo - 1e-9) & (y <= hi + 1e-9)))


def interval_score(y, lo, hi, alpha):
    """Winkler interval score (lower is better)."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)))
