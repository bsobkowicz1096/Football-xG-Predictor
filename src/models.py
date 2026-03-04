import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Import bezpośredni po dodaniu src do sys.path
from evaluation import evaluate_model

def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """Prepares train, validation and test split."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)

    print(f"Goal proportion in training set: {sum(y_train)/len(y_train):.2f}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, max_evals=20, save_plots=False):
    """Trains Logistic Regression using Hyperopt."""
    space = {
        'C': hp.loguniform('C', np.log(0.01), np.log(10)),
        'penalty': hp.choice('penalty', ['l2']),
        'solver': hp.choice('solver', ['lbfgs', 'liblinear']),
        'max_iter': hp.choice('max_iter', [100, 500])
    }

    def objective(params):
        model = LogisticRegression(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)
        loss = log_loss(y_val, preds)
        return {'loss': loss, 'status': STATUS_OK}

    print("Optimizing Logistic Regression...")
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))
    
    # Mapowanie indeksów hp.choice na wartości
    best_params['penalty'] = ['l2'][best_params['penalty']]
    best_params['solver'] = ['lbfgs', 'liblinear'][best_params['solver']]
    best_params['max_iter'] = [100, 500][best_params['max_iter']]

    print(f"Best parameters: {best_params}")
    best_model = LogisticRegression(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    metrics = evaluate_model(best_model, X_test, y_test, X_val, y_val, "logistic_regression" if save_plots else None)
    return best_model, metrics

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, max_evals=25, save_plots=False):
    """Trains Random Forest using Hyperopt."""
    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 500]),
        'max_depth': hp.choice('max_depth', [None, 5, 10, 15, 20, 30, 50]),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
    }

    def objective(params):
        # Konwersja float na int dla parametrów dyskretnych
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)
        loss = log_loss(y_val, preds)
        return {'loss': loss, 'status': STATUS_OK}

    print("Optimizing Random Forest...")
    trials = Trials()
    best_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))
    
    # Rekonstrukcja parametrów z indeksów
    best_params = {
        'n_estimators': [50, 100, 200, 300, 500][best_idx['n_estimators']],
        'max_depth': [None, 5, 10, 15, 20, 30, 50][best_idx['max_depth']],
        'min_samples_split': int(best_idx['min_samples_split']),
        'min_samples_leaf': int(best_idx['min_samples_leaf']),
        'max_features': ['sqrt', 'log2', None][best_idx['max_features']]
    }

    print(f"Best parameters: {best_params}")
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    metrics = evaluate_model(best_model, X_test, y_test, X_val, y_val, "random_forest" if save_plots else None)
    return best_model, metrics

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, max_evals=30, save_plots=False):
    """Trains XGBoost using Hyperopt."""
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
        'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
    }

    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        model = XGBClassifier(**params, random_state=42, eval_metric='logloss', objective='binary:logistic')
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)
        loss = log_loss(y_val, preds)
        return {'loss': loss, 'status': STATUS_OK}

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

    print(f"Best parameters: {best_params}")
    best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    best_model.fit(X_train, y_train)
    
    metrics = evaluate_model(best_model, X_test, y_test, X_val, y_val, "xgboost" if save_plots else None)
    return best_model, metrics