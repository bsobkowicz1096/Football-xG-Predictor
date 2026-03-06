import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

N_CV_FOLDS = 4


def prepare_train_calibration_split(X, y, calib_size=0.2, random_state=42):
    """Prepares 80/20 train/calibration split. Test set is FIFA 2022 (loaded separately)."""
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=calib_size, stratify=y, random_state=random_state
    )
    print(f"Train size: {len(X_train)} | Calibration size: {len(X_calib)}")
    print(f"Goal proportion in training set: {sum(y_train)/len(y_train):.2f}")
    return X_train, X_calib, y_train, y_calib


def _cv_objective(model_cls, params, X_train, y_train):
    """Runs 4-fold stratified CV and returns mean ROC AUC."""
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = model_cls(**params)
        model.fit(X_cv_train, y_cv_train)
        preds = model.predict_proba(X_cv_val)[:, 1]
        auc_scores.append(roc_auc_score(y_cv_val, preds))
    return np.mean(auc_scores), np.std(auc_scores)


def train_logistic_regression(X_train, y_train, max_evals=5):
    """Trains Logistic Regression using Hyperopt with 4-fold CV objective.
    Returns model and CV metrics (no test set used)."""
    space = {
        'C': hp.loguniform('C', np.log(0.01), np.log(10)),
        'penalty': hp.choice('penalty', ['l2']),
        'solver': hp.choice('solver', ['lbfgs', 'liblinear']),
        'max_iter': hp.choice('max_iter', [100, 500])
    }

    def objective(params):
        mean_auc, _ = _cv_objective(
            lambda **p: LogisticRegression(**p, random_state=42),
            params, X_train, y_train
        )
        return {'loss': -mean_auc, 'status': STATUS_OK}

    print("Optimizing Logistic Regression...")
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))

    best_params['penalty'] = ['l2'][best_params['penalty']]
    best_params['solver'] = ['lbfgs', 'liblinear'][best_params['solver']]
    best_params['max_iter'] = [100, 500][best_params['max_iter']]

    mean_cv_auc, std_cv_auc = _cv_objective(
        lambda **p: LogisticRegression(**p, random_state=42),
        best_params, X_train, y_train
    )

    print(f"Best parameters: {best_params}")
    print(f"CV ROC AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")

    best_model = LogisticRegression(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, {'CV ROC AUC': round(mean_cv_auc, 4), 'CV ROC AUC std': round(std_cv_auc, 4)}


def train_random_forest(X_train, y_train, max_evals=5):
    """Trains Random Forest using Hyperopt with 4-fold CV objective.
    Returns model and CV metrics (no test set used)."""
    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 500]),
        'max_depth': hp.choice('max_depth', [None, 5, 10, 15, 20, 30, 50]),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
    }

    def objective(params):
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        mean_auc, _ = _cv_objective(
            lambda **p: RandomForestClassifier(**p, random_state=42, n_jobs=-1),
            params, X_train, y_train
        )
        return {'loss': -mean_auc, 'status': STATUS_OK}

    print("Optimizing Random Forest...")
    trials = Trials()
    best_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))

    best_params = {
        'n_estimators': [50, 100, 200, 300, 500][best_idx['n_estimators']],
        'max_depth': [None, 5, 10, 15, 20, 30, 50][best_idx['max_depth']],
        'min_samples_split': int(best_idx['min_samples_split']),
        'min_samples_leaf': int(best_idx['min_samples_leaf']),
        'max_features': ['sqrt', 'log2', None][best_idx['max_features']]
    }

    mean_cv_auc, std_cv_auc = _cv_objective(
        lambda **p: RandomForestClassifier(**p, random_state=42, n_jobs=-1),
        best_params, X_train, y_train
    )

    print(f"Best parameters: {best_params}")
    print(f"CV ROC AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")

    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    return best_model, {'CV ROC AUC': round(mean_cv_auc, 4), 'CV ROC AUC std': round(std_cv_auc, 4)}


def train_xgboost(X_train, y_train, max_evals=5):
    """Trains XGBoost using Hyperopt with 4-fold CV objective.
    Returns model and CV metrics (no test set used)."""
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
        'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
    }

    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        mean_auc, _ = _cv_objective(
            lambda **p: XGBClassifier(**p, random_state=42, eval_metric='logloss', objective='binary:logistic'),
            params, X_train, y_train
        )
        return {'loss': -mean_auc, 'status': STATUS_OK}

    print("Optimizing XGBoost...")
    trials = Trials()
    best_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))

    best_params = {
        'n_estimators': int(best_idx['n_estimators']),
        'max_depth': [3, 5, 7, 9][best_idx['max_depth']],
        'learning_rate': best_idx['learning_rate'],
        'subsample': best_idx['subsample'],
        'colsample_bytree': best_idx['colsample_bytree']
    }

    mean_cv_auc, std_cv_auc = _cv_objective(
        lambda **p: XGBClassifier(**p, random_state=42, eval_metric='logloss', objective='binary:logistic'),
        best_params, X_train, y_train
    )

    print(f"Best parameters: {best_params}")
    print(f"CV ROC AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")

    best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    best_model.fit(X_train, y_train)

    return best_model, {'CV ROC AUC': round(mean_cv_auc, 4), 'CV ROC AUC std': round(std_cv_auc, 4)}
