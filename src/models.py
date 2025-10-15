from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

from .evaluation import evaluate_model

def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """Prepares train, validation and test split."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)
    
    # Undersampling for training set
    rus = RandomUnderSampler(sampling_strategy=0.43, random_state=random_state)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    
    print(f"Goal proportion after undersampling in training set: {sum(y_train)/len(y_train):.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, save_plots=False):
    """
    Trains logistic regression model with hyperparameter optimization using GridSearchCV.
    """
    # Parameters to search
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],       # Regularization type
        'solver': ['liblinear'],       # Solver compatible with l1 and l2
        'class_weight': [None, 'balanced']  # Class weighting option
    }

    # Model initialization
    log_reg = LogisticRegression(random_state=42, max_iter=1000)

    # GridSearch with cross-validation
    grid_search = GridSearchCV(
        log_reg, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Training on undersampled data
    grid_search.fit(X_train, y_train)

    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    # Best model
    best_log_reg = grid_search.best_estimator_

    # Evaluation
    print("\nModel results:")
    metrics_log_reg = evaluate_model(best_log_reg, X_test, y_test, X_val, y_val, "logistic_regression" if save_plots else None)
    
    for metric, value in metrics_log_reg.items():
        print(f'{metric}: {value}')

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, save_plots=False):
    """
    Trains random forest model with hyperparameter optimization using GridSearchCV.
    """
    # Parameters to search
    param_grid = {
        'n_estimators': [100, 200, 500],        # Number of trees
        'max_depth': [5, 10, 50],            # Maximum depth
        'min_samples_split': [2, 5, 10],     # Min samples to split node
        'min_samples_leaf': [2, 5],      # Min samples in leaf
        'max_features': ['sqrt']     # Number of features to consider for split
    }

    # Model initialization
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # GridSearch with cross-validation
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Training on data
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    # Best model
    best_rf = grid_search.best_estimator_

    # Model evaluation
    print("\nModel results:")
    metrics_rf = evaluate_model(best_rf, X_test, y_test, X_val, y_val, "random_forest" if save_plots else None)
    
    for metric, value in metrics_rf.items():
        print(f'{metric}: {value}')

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, save_plots=False):
    """
    Trains XGBoost model with hyperparameter optimization using GridSearchCV.
    """
    # Parameters to search
    param_grid = {
        'n_estimators': [100, 200, 300],        # Number of boosting trees
        'max_depth': [3, 6, 9],                  # Maximum tree depth
        'learning_rate': [0.01, 0.1],        # Learning rate
        'subsample': [0.8, 1.0],                 # Percentage of samples used for each tree
        'colsample_bytree': [0.6, 0.8]     # Percentage of features used for each tree
    }

    # Model initialization
    xgb = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,  
        objective='binary:logistic'
    )

    # GridSearch with cross-validation
    grid_search = GridSearchCV(
        xgb, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Training on data
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    # Best model
    best_xgb = grid_search.best_estimator_

    # Model evaluation
    print("\nModel results:")
    metrics_xgb = evaluate_model(best_xgb, X_test, y_test, X_val, y_val, "xgboost" if save_plots else None)
    
    for metric, value in metrics_xgb.items():
        print(f'{metric}: {value}')