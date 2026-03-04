# Football xG Predictor — CLAUDE.md

## Project Overview

Machine learning project predicting Expected Goals (xG) in football using StatsBomb open data (2015/16, 5 top leagues).
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
│   ├── models.py                  # train_logistic_regression/random_forest/xgboost
│   ├── evaluation.py              # evaluate_model: ROC AUC + calibration comparison table
│   └── visualization.py           # All plot helpers
├── assets/
│   ├── models/                    # Saved plot PNGs per model
│   └── xg_scatter.png             # Predicted xG per shot (XGBoost, test set)
├── data/                          # Not in repo — run src/data_collector.py first
├── requirements.txt
└── README.md
```

## Data

- Source: StatsBomb open dataset (2015/16, 5 top leagues)
- **Not included in repo** — run `python src/data_collector.py` to collect
- Outputs `data/shots_combined_2015_2016.csv` with `n_passes_in_sequence` included

## Key Modules

### `src/data_collector.py`
- CLI script, run with `python src/data_collector.py`
- `--no-passes-in-sequence` flag to skip pass counting (faster)
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
- All training functions use **Hyperopt** (TPE) for hyperparameter search
- `prepare_train_test_split(X, y)` — 60/20/20 stratified split, **no resampling** (natural ~10% goal rate)
- No StandardScaler, no resampling in pipeline

### `src/evaluation.py`
- `evaluate_model(model, X_test, y_test, X_val, y_val, save_plots, model_name)`
- Prints: ROC AUC → calibration comparison table → plots
- Three calibration methods compared: **Beta**, **Isotonic**, **Platt**
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

## Current Results (January 2026)

| Model | ROC AUC | Brier (Raw) | Best Calibration |
|---|---|---|---|
| Logistic Regression | 0.8007 | 0.0720 | Beta |
| Random Forest | 0.8020 | 0.0722 | Beta |
| **XGBoost** | **0.8034** | **0.0714** | Beta |

Raw probabilities are already well-calibrated (Beta barely improves Brier).

## Next Steps

1. **Cross-validation (Option A)**
   - Keep Hyperopt using val split for hyperparameter tuning (unchanged)
   - After finding best params, re-evaluate final model with **4–5 fold stratified CV** on train+val combined
   - Gives stable metric estimates without full nested CV overhead

2. **Calibration refactor**
   - Train all models → compare by ROC AUC + raw Brier → pick best model
   - Run calibration comparison (Beta / Isotonic / Platt) **only on the best model**
   - Use it as a verification step — assert raw probabilities are well-calibrated
   - Remove per-model calibration from `evaluate_model`

3. **Update markdown cells in notebook**
   - Reflect new features (play_pattern, n_passes_in_sequence)
   - Reflect no-resampling decision and reasoning
   - Update model comparison section with current results
   - Do last, when CV results are final

## Setup

```bash
pip install -r requirements.txt
python src/data_collector.py  # collect data first
# then run notebooks/xg_model.ipynb
```

## Notes

- `src/` modules use direct imports — requires `src/` on `sys.path` (handled in notebook)
- StatsBomb coords: x ∈ [0, 120], y ∈ [0, 80]; goal at x=120, y ∈ [36, 44]
- `df_spatial = df_shots[['x','y']].copy()` saved before final feature selection — used for xG scatter plot
- `play_pattern = 'Other'` in raw data contains ~486 penalties (already filtered) + ~54 misc shots → mapped to open_play
