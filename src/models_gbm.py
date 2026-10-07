"""Global gradient-boosting model (scikit-learn HistGradientBoostingRegressor).

One model is shared by all 1,782 SKU-market series and all 13 horizons (horizon is a feature),
trained on the *relative* log target  log1p(y[t+h]) - lvl8[t].  Missing values (pre-launch lags,
pre-2014 promo tracking, no 2-year seasonal lag) are handled natively by the trees.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from .features import CAT_FEATURES, FEATURES, GROUP_OF


class GlobalGBM:
    def __init__(self, drop_groups=(), params: dict | None = None):
        self.drop_groups = tuple(drop_groups)
        self.params = {**config.GBM_PARAMS, **(params or {})}
        self.cols = [c for c in FEATURES if GROUP_OF[c] not in self.drop_groups]
        self.idx = np.array([FEATURES.index(c) for c in self.cols])
        self.cat_idx = [i for i, c in enumerate(self.cols) if c in CAT_FEATURES]
        self.model = None

    def fit(self, X, y, w=None):
        self.model = HistGradientBoostingRegressor(
            categorical_features=self.cat_idx or None, **self.params)
        self.model.fit(X[:, self.idx], y, sample_weight=w)
        return self

    def predict_dev(self, X):
        return self.model.predict(X[:, self.idx])
