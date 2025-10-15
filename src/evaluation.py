import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve

from .visualization import plot_roc_curve, plot_expected_vs_actual_goals, plot_reliability_diagram


def get_model_viz_path(model_name, viz_type):
    """Generates standard path for model visualizations."""
    path = f"assets/models/{model_name}/{viz_type}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def evaluate_model(model, X_test, y_test, X_val, y_val, save_plots=False, model_name=None):
    """
    Model evaluation using various metrics and calibration.
    """
    # Predictions
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Beta Calibration
    def beta_calibration(p, a, b, c):
        """
        Beta calibration.
        """
        p = np.clip(p, 1e-7, 1-1e-7)
        x = np.log(p / (1 - p))
        return expit(a*x + b*x**2 + c)
    
    def beta_calibration_objective(params, x, y):
        """
        Objective function for beta calibration optimization.
        """
        a, b, c = params
        p_calibrated = beta_calibration(x, a, b, c)
        return log_loss(y, p_calibrated)
    
    # Beta calibration parameter optimization
    initial_params = [1.0, 0.0, 0.0]
    result = minimize(
        beta_calibration_objective, 
        initial_params, 
        args=(y_val_pred, y_val),
        method='Nelder-Mead'
    )
    
    a_opt, b_opt, c_opt = result.x
    
    # Beta calibration for test set
    y_pred_proba_beta = beta_calibration(y_pred_proba, a_opt, b_opt, c_opt)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    brier_beta = brier_score_loss(y_test, y_pred_proba_beta)
    logloss_beta = log_loss(y_test, y_pred_proba_beta)
    
    # xG and actual goals sums
    total_xg = y_pred_proba.sum()
    total_xg_beta = y_pred_proba_beta.sum()
    total_goals = y_test.sum()
    
    xg_ratio = total_xg / total_goals
    xg_beta_ratio = total_xg_beta / total_goals
    
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_roc_curve(y_test, y_pred_proba, roc_auc, ax=axes[0])
    plot_expected_vs_actual_goals(total_xg, total_xg_beta, total_goals, xg_ratio, xg_beta_ratio, ax=axes[1])
    plot_reliability_diagram(y_test, y_pred_proba, y_pred_proba_beta, ax=axes[2])
    
    plt.tight_layout()
    
    if save_plots and model_name:
        plt.savefig(get_model_viz_path(model_name, "full_evaluation"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Returned metrics
    metrics = {
        'ROC AUC': round(roc_auc, 4),
        'Brier Score (Raw)': round(brier, 4),
        'Brier Score (Beta)': round(brier_beta, 4),
        'Log Loss (Raw)': round(logloss, 4),
        'Log Loss (Beta)': round(logloss_beta, 4),
        'xG/Goals Ratio': round(xg_ratio, 4),
        'Beta xG/Goals Ratio': round(xg_beta_ratio, 4)
    }
    
    return metrics