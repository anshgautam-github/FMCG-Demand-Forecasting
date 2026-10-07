"""Unit / smoke tests.  Run with:  python -m pytest -q   (or python tests/test_pipeline.py)

They use a tiny synthetic dataset (tools/make_synthetic_data.py), so they take ~1 minute and need
no real data.  The most important test is `test_no_future_leakage_in_features`."""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import features, validation  # noqa: E402
from src.data_loading import load_raw  # noqa: E402
from src.explain import group_shapley  # noqa: E402
from src.intervals import ConformalIntervals, coverage  # noqa: E402
from src.models_gbm import GlobalGBM  # noqa: E402
from tools.make_synthetic_data import make  # noqa: E402
from src.weekly_dataset import build_panel  # noqa: E402

_TMP = Path(tempfile.mkdtemp())
make(_TMP)
RAW = load_raw(_TMP)
P = build_panel(RAW, 13)


def test_panel_shapes_and_masks():
    assert P.Y.shape == (P.S, P.Wh) and P.PROMO.shape == (P.S, P.Wh + 13)
    assert np.isnan(P.Y).any()                      # pre-launch / closure weeks are masked
    assert np.nanmin(P.Y_raw) >= 0
    assert P.promo_start_week > 0 and np.isnan(P.PROMO[:, :P.promo_start_week]).all()
    assert len(P.gap_spells) > 0                    # the synthetic PRODUCE reporting gap is found


def test_folds_do_not_overlap_and_cover_last_year():
    folds = validation.make_folds(P, 4, 13)
    assert folds[-1].last_target == P.Wh - 1
    for a, b in zip(folds[:-1], folds[1:]):
        assert a.last_target + 1 == b.first_target


def test_training_targets_never_after_origin():
    o = P.Wh - 14
    rng = np.random.default_rng(0)
    X, y, w, meta = features.build_training(P, o, 3, 50_000, rng, 104, gaps=P.gap_mask(o))
    assert meta[:, 2].max() <= o                    # purged: all target weeks <= fold origin


def test_no_future_leakage_in_features():
    """Features at origin t must be identical if EVERYTHING observed after t is scrambled."""
    t = P.Wh - 30
    X0, _ = features.build_block(P, t, range(1, 14))
    Q = copy.deepcopy(P)
    rng = np.random.default_rng(1)
    for name in ("Y", "L", "Y_raw", "Z"):
        a = getattr(Q, name)
        a[:, t + 1:] = rng.random(a[:, t + 1:].shape) * 100
    Q.TR[:, t + 1:] = rng.random(Q.TR[:, t + 1:].shape) * 5000
    Q.OIL[t + 1:P.Wh] = 1e3
    X1, _ = features.build_block(Q, t, range(1, 14))
    assert np.allclose(X0, X1, equal_nan=True)


def test_shapley_efficiency():
    rng = np.random.default_rng(0)
    X, y, w, _ = features.build_training(P, P.Wh - 14, 6, 8000, rng, 104)
    g = GlobalGBM(params=dict(max_iter=15)).fit(X, y)
    inst, bg = X[:5], X[100:112]
    phi, base, full, _ = group_shapley(g, inst, bg)
    assert np.allclose(phi.sum(1), full - base, atol=1e-6)


def test_conformal_interval_coverage():
    rng = np.random.default_rng(0)
    n = 20000
    f = rng.uniform(5, 500, n); y = f * np.exp(rng.normal(0, .3, n))
    lvl = np.log1p(f); h = rng.integers(1, 14, n)
    ci = ConformalIntervals().fit(h[:10000], lvl[:10000], y[:10000], f[:10000])
    iv = ci.predict(h[10000:], lvl[10000:], f[10000:])
    assert abs(coverage(y[10000:], iv["lower_90"], iv["upper_90"]) - 0.9) < 0.02


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok  ", k)
