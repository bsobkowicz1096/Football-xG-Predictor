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


def evaluate_model(model, X_test, y_test, save_plots=False, model_name=None):
    """
    Evaluates model on test set using raw probabilities only.
    Metrics: ROC AUC, Brier Score, ECE.
    Plots: ROC curve.
    """
    y_test_arr = np.array(y_test)
    y_pred = model.predict_proba(X_test)[:, 1]

    roc_auc   = roc_auc_score(y_test_arr, y_pred)
    brier     = brier_score_loss(y_test_arr, y_pred)
    ece       = compute_ece(y_test_arr, y_pred)
    xg_goals  = y_pred.sum() / y_test_arr.sum()

    print(f"  ROC AUC:      {roc_auc:.4f}")
    print(f"  Brier Score:  {brier:.4f}")
    print(f"  ECE:          {ece:.4f}")
    print(f"  xG/Goals:     {xg_goals:.4f}")

    if save_plots and model_name:
        plot_roc_curve(y_test_arr, y_pred, roc_auc,
                       save_path=get_model_viz_path(model_name, "roc_curve"))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_roc_curve(y_test_arr, y_pred, roc_auc, ax=ax)
        plt.tight_layout()
        plt.show()

    return {
        'ROC AUC':     round(roc_auc, 4),
        'Brier (Raw)': round(brier, 4),
        'ECE (Raw)':   round(ece, 4),
        'xG/Goals':    round(xg_goals, 4),
    }


def calibrate_best_model(model, X_calib, y_calib, X_test, y_test, save_plots=False, model_name=None):
    """
    Fits Beta, Isotonic, and Platt calibration on X_calib,
    evaluates all three on X_test, prints comparison table,
    and plots reliability diagram + expected vs actual goals.
    Best method selected by lowest Brier Score.
    """
    y_calib_pred = model.predict_proba(X_calib)[:, 1]
    y_test_pred  = model.predict_proba(X_test)[:, 1]
    y_test_arr   = np.array(y_test)

    # --- Beta Calibration ---
    def beta_calibration(p, a, b, c):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        x = np.log(p / (1 - p))
        return expit(a * x + b * x**2 + c)

    result = minimize(
        lambda params, x, y: log_loss(y, beta_calibration(x, *params)),
        [1.0, 0.0, 0.0],
        args=(y_calib_pred, y_calib),
        method='Nelder-Mead'
    )
    y_pred_beta = beta_calibration(y_test_pred, *result.x)

    # --- Isotonic Regression ---
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_calib_pred, y_calib)
    y_pred_isotonic = iso.predict(y_test_pred)

    # --- Platt Scaling ---
    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(y_calib_pred.reshape(-1, 1), y_calib)
    y_pred_platt = platt.predict_proba(y_test_pred.reshape(-1, 1))[:, 1]

    calibrated = {
        'Beta':     y_pred_beta,
        'Isotonic': y_pred_isotonic,
        'Platt':    y_pred_platt,
    }

    # --- Metrics ---
    brier_raw = brier_score_loss(y_test_arr, y_test_pred)
    ece_raw   = compute_ece(y_test_arr, y_test_pred)
    total_goals = y_test_arr.sum()
    xg_raw      = y_test_pred.sum()

    cal_metrics = {}
    for name, preds in calibrated.items():
        cal_metrics[name] = {
            'brier': brier_score_loss(y_test_arr, preds),
            'ece':   compute_ece(y_test_arr, preds),
            'xg':    preds.sum(),
            'ratio': preds.sum() / total_goals,
        }

    best_name = min(cal_metrics, key=lambda k: cal_metrics[k]['brier'])

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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_expected_vs_actual_goals(
        total_goals, xg_raw,
        {name: (m['xg'], m['ratio']) for name, m in cal_metrics.items()},
        ax=axes[0]
    )
    plot_reliability_diagram(y_test_arr, y_test_pred, calibrated, ax=axes[1])
    plt.tight_layout()

    if save_plots and model_name:
        plt.savefig(get_model_viz_path(model_name, "calibration"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    metrics = {
        'Best Calibration': best_name,
        'Brier (Raw)':      round(brier_raw, 4),
        'ECE (Raw)':        round(ece_raw, 4),
    }
    for name, m in cal_metrics.items():
        metrics[f'Brier ({name})'] = round(m['brier'], 4)
        metrics[f'ECE ({name})']   = round(m['ece'], 4)

    return metrics
