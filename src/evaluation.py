import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from visualization import plot_roc_curve, plot_expected_vs_actual_goals, plot_reliability_diagram


def get_model_viz_path(model_name, viz_type):
    """Generates standard path for model visualizations."""
    path = f"assets/models/{model_name}/{viz_type}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def compute_ece(y_true, y_pred, n_bins=10):
    """
    Expected Calibration Error — weighted average of |mean predicted - mean actual|
    across quantile-based bins (each bin contains equal number of samples).
    """
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])
    n = len(y_true)
    ece = 0.0
    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_pred[mask].mean() - y_true[mask].mean())
    return ece


def evaluate_model(model, X_test, y_test, X_val, y_val, save_plots=False, model_name=None):
    """
    Evaluates model with three calibration methods (Beta, Isotonic, Platt).
    Metrics: ROC AUC, Brier Score, ECE (Expected Calibration Error).
    Best calibration selected by lowest Brier Score on the test set.
    """
    y_val_pred  = model.predict_proba(X_val)[:, 1]
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # --- Beta Calibration ---
    def beta_calibration(p, a, b, c):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        x = np.log(p / (1 - p))
        return expit(a * x + b * x**2 + c)

    result = minimize(
        lambda params, x, y: log_loss(y, beta_calibration(x, *params)),
        [1.0, 0.0, 0.0],
        args=(y_val_pred, y_val),
        method='Nelder-Mead'
    )
    y_pred_beta = beta_calibration(y_pred_proba, *result.x)

    # --- Isotonic Regression ---
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_val_pred, y_val)
    y_pred_isotonic = iso.predict(y_pred_proba)

    # --- Platt Scaling ---
    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(y_val_pred.reshape(-1, 1), y_val)
    y_pred_platt = platt.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]

    calibrated = {
        'Beta':     y_pred_beta,
        'Isotonic': y_pred_isotonic,
        'Platt':    y_pred_platt,
    }

    # --- Metrics ---
    y_test_arr  = np.array(y_test)
    roc_auc     = roc_auc_score(y_test_arr, y_pred_proba)
    brier_raw   = brier_score_loss(y_test_arr, y_pred_proba)
    ece_raw     = compute_ece(y_test_arr, y_pred_proba)
    total_goals = y_test_arr.sum()
    xg_raw      = y_pred_proba.sum()

    cal_metrics = {}
    for name, preds in calibrated.items():
        cal_metrics[name] = {
            'brier': brier_score_loss(y_test_arr, preds),
            'ece':   compute_ece(y_test_arr, preds),
            'xg':    preds.sum(),
            'ratio': preds.sum() / total_goals,
        }

    best_name = min(cal_metrics, key=lambda k: cal_metrics[k]['brier'])

    print(f"\nROC AUC: {roc_auc:.4f}")

    # Calibration comparison table
    table = pd.DataFrame(
        {
            'Brier Score': {
                'Raw': round(brier_raw, 4),
                **{name: round(m['brier'], 4) for name, m in cal_metrics.items()}
            },
            'ECE': {
                'Raw': round(ece_raw, 4),
                **{name: round(m['ece'], 4) for name, m in cal_metrics.items()}
            },
            'xG/Goals': {
                'Raw': round(xg_raw / total_goals, 4),
                **{name: round(m['ratio'], 4) for name, m in cal_metrics.items()}
            },
        }
    ).T
    table.index.name = 'Metric'
    print(f"\nCalibration comparison (best: {best_name}):")
    print(table.to_string())

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_roc_curve(y_test_arr, y_pred_proba, roc_auc, ax=axes[0])
    plot_expected_vs_actual_goals(
        total_goals, xg_raw,
        {name: (m['xg'], m['ratio']) for name, m in cal_metrics.items()},
        ax=axes[1]
    )
    plot_reliability_diagram(y_test_arr, y_pred_proba, calibrated, ax=axes[2])

    plt.tight_layout()

    if save_plots and model_name:
        plt.savefig(get_model_viz_path(model_name, "full_evaluation"), dpi=300, bbox_inches='tight')
        plt.close()

        plot_roc_curve(y_test_arr, y_pred_proba, roc_auc,
                       save_path=get_model_viz_path(model_name, "roc_curve"))
        plt.close()
        plot_expected_vs_actual_goals(
            total_goals, xg_raw,
            {name: (m['xg'], m['ratio']) for name, m in cal_metrics.items()},
            save_path=get_model_viz_path(model_name, "expected_goals")
        )
        plt.close()
        plot_reliability_diagram(
            y_test_arr, y_pred_proba, calibrated,
            save_path=get_model_viz_path(model_name, "reliability")
        )
        plt.close()
    else:
        plt.show()

    metrics = {
        'ROC AUC':          round(roc_auc, 4),
        'Brier (Raw)':      round(brier_raw, 4),
        'ECE (Raw)':        round(ece_raw, 4),
        'xG/Goals (Raw)':   round(xg_raw / total_goals, 4),
        'Best Calibration': best_name,
    }
    for name, m in cal_metrics.items():
        metrics[f'Brier ({name})']    = round(m['brier'], 4)
        metrics[f'ECE ({name})']      = round(m['ece'], 4)
        metrics[f'xG/Goals ({name})'] = round(m['ratio'], 4)

    return metrics
