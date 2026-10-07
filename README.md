# FreshBasket FMCG Demand Forecasting - 13-week SKU x market forecasts

A reproducible Python solution to *Assessment 1: FMCG Demand Forecasting*.
It forecasts weekly units for **1,782 SKU-market series for the next 13 weeks**, with calibrated prediction
intervals, driver attribution, an ablation study and a scenario simulator.

> **Read this first - dataset mapping.** The files supplied with the assessment are the *Store Sales* (Favorita-style)
> retail dataset (`train, test, stores, transactions, oil, holidays_events`), **not** the FMCG schema listed in the brief
> (`weekly_sales, price, distribution, media, weather ...`). The project therefore maps the data as follows, and the
> report states the consequences openly:
>
> | Brief concept | Used here | Note |
> |---|---|---|
> | market_id | `store_nbr` (54 stores) | |
> | sku_id | `family` (33 product families) | families, not individual SKUs |
> | units | `sales`, daily -> weekly (Mon-Sun) | |
> | promotion | `onpromotion` = # items of the family on promo | no discount depth / promo type |
> | **price** | **not available** | cannot be modelled or ablated |
> | distribution | zero-sales-day share, series age, zero streak ("availability proxy") | proxy only |
> | weather | **not available**; oil price used as the only external macro driver | proxy only |
> | media | **not available** | |
> | calendar | `holidays_events` (national/regional/local, transfers, events) + paydays + Christmas/New Year | |
> | traffic | `transactions` (lag-only, unknown in the future) | extra driver |

## 1. Environment

Python 3.10+ (developed on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No LightGBM / statsmodels / SHAP is needed: gradient boosting uses scikit-learn's `HistGradientBoostingRegressor`,
the statistical models are implemented in NumPy, and driver attribution uses an exact group-Shapley implementation
(`src/explain.py`). Everything runs on a single CPU core with ~3 GB RAM.

## 2. Run

Put the six CSV files in `data/raw/` (names may carry a prefix, e.g. `1791_train.csv`; they are matched by suffix) or point to them:

```bash
python run_pipeline.py                         # all stages, ~8-10 min on 1 CPU
python run_pipeline.py --raw-dir /path/to/csv  # data elsewhere
python run_pipeline.py --stages eda,cv         # selected stages (cached in outputs/cache)
python run_pipeline.py --force                 # ignore caches
python tests/test_pipeline.py                  # unit + leakage tests on tiny synthetic data (~1 min)
python tools/make_synthetic_data.py data/synthetic   # tiny dataset with the same schema (no real data needed)
```

Stages: `eda -> cv -> ablation -> forecast -> explain -> scenarios -> report`.

## 3. Outputs

| Deliverable | File |
|---|---|
| Data-quality & EDA report | `outputs/reports/01_data_quality_and_eda_report.md`, `outputs/figures/eda*.png`, `outputs/tables/dq_*.csv` |
| **Forecast file** (date, market_id, sku_id, forecast, 50% / 90% intervals) | `outputs/forecast_13w.csv` |
| Model comparison & time-series CV | `outputs/tables/cv_*.csv`, `outputs/figures/res01-res04,res11*.png` |
| Interval calibration | `outputs/tables/interval_coverage_*.csv`, `res05*.png` |
| Explainability / driver attribution | `outputs/tables/explain_*.csv`, `res07*, res08*.png` |
| Ablation / feature impact | `outputs/tables/ablation_*.csv`, `res06*.png` |
| Scenario & sensitivity analysis | `outputs/tables/scenario_*.csv`, `res10*.png` |
| Project report | `reports/PROJECT_REPORT.md` |
| Presentation (10-15 min) | `reports/FreshBasket_Forecasting_Presentation.pptx` |

`forecast_13w.csv` columns: `date` (Monday week start), `week_end`, `horizon_week`, `market_id`, `sku_id`, `forecast`
(point = median-type forecast, units), `lower_50/upper_50`, `lower_90/upper_90`, `forecast_gbm`, `forecast_stat`,
`run_rate_units`, `promo_plan_known_share`, `flag_cold_start`, `flag_delisted_or_unlaunched`.

## 4. Code map

```
config.py               all settings (horizon, folds, model params, rules)
run_pipeline.py         orchestrator
src/data_loading.py     file discovery + typed loading
src/weekly_dataset.py   daily -> weekly panel [series x week] arrays, calendar/holiday/promo/traffic/oil, masks, gap detection
src/quality.py, eda.py  data-quality checks, EDA figures and report
src/features.py         leakage-safe direct multi-horizon feature builder (+ training-row sampler)
src/validation.py       rolling-origin folds
src/models_stat.py      naive, seasonal-naive x growth, damped Holt, cold-start peers, business rules
src/models_gbm.py       global gradient-boosting model
src/cv.py               cross-validation engine
src/ensemble.py         hybrid blend + cold-start strategy selection
src/intervals.py        conformal prediction intervals + coverage / interval score
src/ablation.py         driver-group ablation
src/explain.py          permutation importance + exact group-Shapley attribution
src/scenarios.py        scenario simulator and promo response curves
src/forecast.py         final forecast
src/reporting.py        result tables and figures
tests/, tools/          tests; synthetic-data generator
```

## 5. Method in one page

* **Panel.** Daily rows -> Monday-start weeks (240 complete weeks, 2013-01-07 .. 2017-08-07). First/last partial weeks dropped.
  Pre-launch zeros, whole-week store closures are *missing* (not demand). Promotion tracking starts 2014-03-31 -> earlier values are missing, not zero.
* **Model.** One **global** gradient-boosting model for all series and all horizons (direct strategy, horizon is a feature).
  Target = `log1p(units[t+h]) - run-rate[t]`, run-rate = mean log1p of the last 8 weeks. Features: lags/rolling stats, year-ago
  seasonal values and YoY growth, promotion plan for the target week, availability proxies, store-traffic and oil lags,
  holiday/event/payday/Christmas calendar flags, store/family attributes. Training rows are volume-weighted (power 0.3) and recency-weighted.
* **Baselines.** naive 4-week mean, seasonal-naive x trailing growth, damped-trend Holt. **Hybrid:** 0.9 GBM + 0.1 seasonal-naive (weight picked on folds 1-3 only).
* **Validation.** 4 expanding-window rolling-origin folds x 13 weeks tiling the last 52 weeks (all seasons, incl. Christmas & Carnival). The model of fold *k*
  only sees targets <= its origin. No random splits anywhere. `tests/` contains a test that scrambles all post-origin data and checks features are unchanged.
* **Uncertainty.** Mondrian split-conformal intervals on log-residuals (horizon bin x volume tier), calibrated on out-of-sample fold residuals, checked on the untouched last fold.
* **Explainability.** Permutation importance (global) and exact Shapley values over 8 driver groups (local; efficiency verified).
* **Ablation.** Re-train without each driver group, re-score on all four folds.
* **Scenarios.** Re-run the trained model with modified promotion intensity / traffic, report the demand difference; promo response curves per family.
* **Cold start / sparse.** Series with <13 weeks of history use the best-validated fallback; never-sold -> 0; delisted (8+ trailing zero weeks, unless seasonal) -> 0.

## 6. Assumptions (also in the report)

1. Future exogenous inputs: **calendar/holidays are known** (holiday file runs to Dec 2017); the **promotion plan** is known for 2.6 weeks from `test.csv`
   and **assumed = last-28-day average** afterwards; **traffic and oil are never needed in the future** (lagged, origin-dated features only).
2. In cross-validation the promotion plan of the target weeks is treated as *known* (a trade plan); a sensitivity run with the plan unknown is reported.
3. Whole-week store closures and 8+ week "reporting gaps" (zero spells that later resume) are excluded from *scoring*; a sensitivity including them is reported.
4. `test.csv` has no sales, so it cannot score the forecasts; the 13 forecast weeks are 2017-08-14 .. 2017-11-12.

## 7. Limitations

See `reports/PROJECT_REPORT.md` section 9: proxies replace price/distribution/weather/media; families rather than SKUs; promo effects are associations, not causal; small-volume series remain hard.
