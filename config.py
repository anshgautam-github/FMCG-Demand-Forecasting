"""Central configuration for the FreshBasket demand-forecasting project."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("FMCG_RAW_DIR", ROOT / "data" / "raw"))
OUT_DIR = Path(os.environ.get("FMCG_OUT_DIR", ROOT / "outputs"))
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
REP_DIR = OUT_DIR / "reports"
CACHE_DIR = OUT_DIR / "cache"

SEED = 42
HORIZON = 13            # forecast horizon in weeks
N_FOLDS = 4             # rolling-origin folds, each forecasting HORIZON weeks
MIN_HISTORY_WEEKS = 52  # first usable forecast origin (needs a full seasonal lag)
SEASON = 52

# ---- gradient boosting (kept light: the project must run on a 1-CPU laptop) ----
GBM_PARAMS = dict(
    learning_rate=0.08, max_iter=160, max_leaf_nodes=48, min_samples_leaf=150,
    l2_regularization=2.0, max_bins=128, early_stopping=False, random_state=SEED,
)
TRAIN_ORIGIN_STEP = 3       # use every 3rd weekly origin when building training rows
TRAIN_MAX_ROWS = 380_000    # random row sub-sample (training only, never evaluation)
RECENCY_HALF_LIFE_W = 104   # sample-weight half-life (weeks)
VOL_WEIGHT_POWER = 0.3      # exponent on series volume in sample weights (0.0/0.3/0.6 compared on the CV folds)

ABL_MAX_ROWS = 170_000
ABL_ITER = 70

# ---- business rules ----
DISCONTINUE_ZERO_WEEKS = 8  # >= this many trailing zero weeks => delisted => forecast 0
COLDSTART_MIN_ACTIVE_WEEKS = 13
INTERVAL_LEVELS = (0.50, 0.90)

for _d in (RAW_DIR, FIG_DIR, TAB_DIR, REP_DIR, CACHE_DIR, OUT_DIR / "models"):
    _d.mkdir(parents=True, exist_ok=True)
