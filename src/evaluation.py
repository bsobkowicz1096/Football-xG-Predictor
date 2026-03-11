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
    path = f"../assets/models/{model_name}/{viz_type}.png"
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


def evaluate_model(model, X, y, calibrator=None, save_plots=False, model_name=None):
    """
    Evaluates model on a given set. If calibrator is provided, applies it to
    raw probabilities before computing metrics.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X : features
    y : true labels
    calibrator : callable or None
        Takes raw probabilities, returns calibrated probabilities.
        If None, raw model probabilities are used.

    Metrics: ROC AUC, Brier Score, ECE, xG/Goals ratio.
    """
    y_arr  = np.array(y)
    y_raw  = model.predict_proba(X)[:, 1]
    y_pred = calibrator(y_raw) if calibrator is not None else y_raw

    roc_auc  = roc_auc_score(y_arr, y_pred)
    brier    = brier_score_loss(y_arr, y_pred)
    ece      = compute_ece(y_arr, y_pred)
    xg_goals = y_pred.sum() / y_arr.sum()

    print(f"  ROC AUC:      {roc_auc:.4f}")
    print(f"  Brier Score:  {brier:.4f}")
    print(f"  ECE:          {ece:.4f}")
    print(f"  xG/Goals:     {xg_goals:.4f}")

    if save_plots and model_name:
        plot_roc_curve(y_arr, y_pred, roc_auc,
                       save_path=get_model_viz_path(model_name, "roc_curve"))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_roc_curve(y_arr, y_pred, roc_auc, ax=ax)
        plt.tight_layout()
        plt.show()

    return {
        'ROC AUC':     round(roc_auc, 4),
        'Brier Score': round(brier, 4),
        'ECE':         round(ece, 4),
        'xG/Goals':    round(xg_goals, 4),
    }


def calibrate_best_model(model, X_calib, y_calib, save_plots=False, model_name=None):
    """
    Fits Beta, Isotonic, and Platt calibration on X_calib and evaluates all
    methods on the same set. Picks best by lowest Brier Score.

    Note: fitting and evaluating on the same set introduces a small optimistic
    bias, particularly for Isotonic regression. This is acceptable given limited
    data — the raw model is already well-calibrated so calibration is a
    verification step rather than a critical correction.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X_calib : calibration features
    y_calib : calibration labels

    Returns
    -------
    best_name : str
        Name of best calibration method (or 'Raw').
    calibrators : dict
        Maps method name to a callable(raw_probs) -> calibrated_probs.
        Includes 'Raw' as identity function.
    metrics : dict
        Brier Score and ECE for Raw and each calibration method.
    """
    y_calib_pred = model.predict_proba(X_calib)[:, 1]
    y_calib_arr  = np.array(y_calib)

    # --- Beta Calibration ---
    def beta_calibration(p, a, b, c):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        x = np.log(p / (1 - p))
        return expit(a * x + b * x**2 + c)

    result = minimize(
        lambda params, x, y: log_loss(y, beta_calibration(x, *params)),
        [1.0, 0.0, 0.0],
        args=(y_calib_pred, y_calib_arr),
        method='Nelder-Mead'
    )
    beta_params = result.x

    # --- Isotonic Regression ---
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_calib_pred, y_calib_arr)

    # --- Platt Scaling ---
    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(y_calib_pred.reshape(-1, 1), y_calib_arr)

    calibrators = {
        'Raw':      lambda p: p,
        'Beta':     lambda p: beta_calibration(p, *beta_params),
        'Isotonic': lambda p: iso.predict(p),
        'Platt':    lambda p: platt.predict_proba(p.reshape(-1, 1))[:, 1],
    }

    # --- Evaluate all methods on calib set ---
    all_metrics = {}
    for name, fn in calibrators.items():
        preds = fn(y_calib_pred)
        all_metrics[name] = {
            'brier': brier_score_loss(y_calib_arr, preds),
            'ece':   compute_ece(y_calib_arr, preds),
            'xg':    preds.sum(),
            'ratio': preds.sum() / y_calib_arr.sum(),
        }

    table = pd.DataFrame({
        'Brier Score': {n: round(m['brier'], 4) for n, m in all_metrics.items()},
        'ECE':         {n: round(m['ece'],   4) for n, m in all_metrics.items()},
        'xG/Goals':    {n: round(m['ratio'], 4) for n, m in all_metrics.items()},
    }).T
    table.index.name = 'Metric'
    print(f"\nCalibration comparison on calib set")
    print(table.to_string())

    # --- Plots ---
    calibrated_only = {n: calibrators[n](y_calib_pred) for n in ['Beta', 'Isotonic', 'Platt']}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_expected_vs_actual_goals(
        y_calib_arr.sum(), y_calib_pred.sum(),
        {n: (all_metrics[n]['xg'], all_metrics[n]['ratio']) for n in ['Beta', 'Isotonic', 'Platt']},
        ax=axes[0]
    )
    plot_reliability_diagram(y_calib_arr, y_calib_pred, calibrated_only, ax=axes[1])
    plt.tight_layout()

    if save_plots and model_name:
        plt.savefig(get_model_viz_path(model_name, "calibration"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    metrics = {}
    for name, m in all_metrics.items():
        metrics[f'Brier ({name})'] = round(m['brier'], 4)
        metrics[f'ECE ({name})']   = round(m['ece'],   4)

    return calibrators, metrics
