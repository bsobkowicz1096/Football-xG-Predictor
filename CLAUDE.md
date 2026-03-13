# Football xG Predictor — CLAUDE.md

## Project Overview

Machine learning project predicting Expected Goals (xG) in football using StatsBomb open data.
Trains and compares Logistic Regression, Random Forest, and XGBoost.

## Project Structure

```
Football-xG-Predictor/
├── notebooks/
│   └── xg_model.ipynb             # Main pipeline: EDA → features → training → evaluation
├── src/
│   ├── data_collector.py          # StatsBomb API collection (CLI, includes n_passes_in_sequence)
│   ├── data_processing.py         # load_data, clean_data, select_features
│   ├── feature_engineering.py     # Geometry, freeze-frame, body part, play pattern transforms
│   ├── models.py                  # train_logistic_regression/random_forest/xgboost + CV objective
│   ├── evaluation.py              # evaluate_model (raw metrics) + calibrate_best_model
│   └── visualization.py           # All plot helpers
├── assets/
│   ├── models/                    # Saved plot PNGs per model
│   └── xg_scatter.png             # Predicted xG per shot (XGBoost, test set)
├── data/                          # Not in repo — run src/data_collector.py first
├── requirements.txt
└── README.md
```

## Data

- **Training/Calibration**: StatsBomb open dataset (2015/16, 5 top leagues) → `data/shots_combined_2015_2016.csv`
- **Test set**: FIFA World Cup 2022 → `data/shots_fifa_world_cup_2022.csv` (true out-of-sample holdout)
- **Not included in repo** — collect with:
  ```bash
  python src/data_collector.py                          # club data (2015/16)
  python src/data_collector.py --fifa-2022 --skip-club  # FIFA 2022 only
  ```
- Both datasets include `n_passes_in_sequence` by default

## Key Modules

### `src/data_collector.py`
- `collect_shots_data(...)` — 2015/16 club leagues
- `collect_fifa_world_cup_2022(...)` — FIFA World Cup 2022 (test holdout)
- `--no-passes-in-sequence` flag to skip pass counting (faster)
- `--fifa-2022` / `--skip-club` CLI flags
- `include_passes_in_sequence=True` by default — counts passes per possession via `(match_id, possession)` groupby

### `src/feature_engineering.py`
- `extract_xy(df)` — parses location string → x, y coords
- `calculate_angles_distances(df)` — shot angle + distance from goal
- `transform_body_part(df)` — better_foot / worse_foot / head
- `analyze_freeze_frame(df)` — defenders_in_path, goalkeeper_in_path, goalkeeper_distance_ratio
- `transform_play_pattern(df)` — groups play_pattern into corner / set_piece / counter / open_play (reference), creates dummies
- `create_dummies(df, column, drop_category)` — one-hot encoding
- `standardize_features(df, continuous_vars)` — StandardScaler

### `src/models.py`
- All training functions use **Hyperopt** (TPE) with **4-fold stratified CV** as the objective (`_cv_objective`)
- `prepare_train_calibration_split(X, y)` — 80/20 stratified split (train / calibration), **no resampling**
- Training functions return `(model, {'CV ROC AUC': ..., 'CV ROC AUC std': ...})`
- No StandardScaler, no resampling in pipeline

### `src/evaluation.py`
- `evaluate_model(model, X_test, y_test, save_plots, model_name)` — raw metrics only (ROC AUC, Brier, ECE, xG/Goals)
- `calibrate_best_model(model, X_calib, y_calib, X_test, y_test, ...)` — fits Beta / Isotonic / Platt on calib set, evaluates on test set, prints comparison table + plots
- Three calibration methods: **Beta** (Nelder-Mead), **Isotonic**, **Platt**
- Metrics: **Brier Score** + **ECE** (quantile-based bins), no log loss
- Best calibration selected by lowest Brier Score

### `src/visualization.py`
- `plot_reliability_diagram` — quantile-based bins, zoomed to [0, 0.4]
- `plot_expected_vs_actual_goals(total_goals, xg_raw, calibrated)` — bar chart with all calibration methods
- `plot_xg_scatter(x, y, xg_values)` — VerticalPitch half-pitch, inferno_r palette
- `plot_passes_in_sequence_effectiveness(df)` — stacked bar, bins: 0/1/2-3/4-5/6-7/8-9/10+
- `plot_stacked_bar(df, col, title, xlabel)` — goals vs non-goals per category

## Features (15 total)

| Feature | Type | Notes |
|---|---|---|
| `distance_scaled` | continuous | StandardScaler |
| `log_angle_scaled` | continuous | log transform + StandardScaler |
| `log_passes_in_sequence_scaled` | continuous | log1p + StandardScaler |
| `under_pressure` | binary | |
| `shot_first_time` | binary | |
| `better_foot` | binary | from refined_body_part |
| `worse_foot` | binary | from refined_body_part |
| `goalkeeper_in_path` | binary | freeze-frame |
| `open_play_shot` | binary | |
| `normal_shot` | binary | |
| `defenders_in_path` | ordinal (0–3) | capped at 3 |
| `goalkeeper_distance_ratio` | continuous | freeze-frame, not standardized |
| `corner` | binary | play_pattern group |
| `set_piece` | binary | play_pattern group |
| `counter` | binary | play_pattern group |

`continuous_vars = ['distance', 'log_angle', 'log_passes_in_sequence']`

## Split Strategy

| Set | Source | Size | Purpose |
|---|---|---|---|
| Train | 2015/16 club data | 80% | Hyperopt + 4-fold CV tuning |
| Calibration | 2015/16 club data | 20% | Fit calibration methods |
| Test | FIFA World Cup 2022 | all | True out-of-sample evaluation |

## Current Results (March 2026)

CV results (4-fold stratified, 2015/16 club data):

| Model | CV ROC AUC | CV ROC AUC std |
|---|---|---|
| Logistic Regression | 0.8065 | 0.0043 |
| Random Forest | 0.8094 | 0.0054 |
| **XGBoost** | **0.8134** | **0.0040** |

FIFA World Cup 2022 test set (XGBoost — best model, raw probabilities):

| Metric | Value |
|---|---|
| ROC AUC | 0.8051 |
| Brier Score | 0.0765 |
| ECE | 0.0178 |
| xG/Goals | 0.9162 |

Raw probabilities selected — Isotonic achieved lower Brier (0.0701 vs 0.0708) but ECE = 0.0000 indicates overfitting on the calibration set.

## Next Steps

1. **Update notebook markdown cells**
   - Split section: reflect 80/20 train/calib + FIFA 2022 as test holdout
   - Evaluation definition: reflect new `evaluate_model` / `calibrate_best_model` separation
   - Model parameter sections: update with actual CV results
   - Project summary: update results table

## Setup

```bash
pip install -r requirements.txt
python src/data_collector.py                          # collect 2015/16 club data
python src/data_collector.py --fifa-2022 --skip-club  # collect FIFA 2022 test data
# then run notebooks/xg_model.ipynb
```

## Notes

- `src/` modules use direct imports — requires `src/` on `sys.path` (handled in notebook)
- StatsBomb coords: x ∈ [0, 120], y ∈ [0, 80]; goal at x=120, y ∈ [36, 44]
- `df_spatial = df_shots[['x','y']].copy()` saved before final feature selection — used for xG scatter plot
- `play_pattern = 'Other'` in raw data contains ~486 penalties (already filtered) + ~54 misc shots → mapped to open_play
- FIFA 2022 data goes through the same feature engineering pipeline as club data
