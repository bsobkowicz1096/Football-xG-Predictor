from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

from .evaluation import evaluate_model



def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """Przygotowuje podział na zbiory treningowy, walidacyjny i testowy."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)
    
    # Undersampling do zbioru treningowego
    rus = RandomUnderSampler(sampling_strategy=0.43, random_state=random_state)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    
    print(f"Proporcja goli po undersamplingu w zbiorze treningowym: {sum(y_train)/len(y_train):.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, save_plots=False):
    """
    Trenuje model regresji logistycznej z optymalizacją hiperparametrów za pomocą GridSearchCV.
    """
    # Parametry do przeszukania
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Siła regularyzacji
        'penalty': ['l1', 'l2'],       # Typ regularyzacji
        'solver': ['liblinear'],       # Solver kompatybilny z l1 i l2
        'class_weight': [None, 'balanced']  # Opcja ważenia klas
    }

    # Inicjalizacja modelu
    log_reg = LogisticRegression(random_state=42, max_iter=1000)

    # GridSearch z cross-walidacją
    grid_search = GridSearchCV(
        log_reg, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Trenowanie na danych oversamplingowych
    grid_search.fit(X_train, y_train)

    # Najlepsze parametry
    print(f"Najlepsze parametry: {grid_search.best_params_}")
    print(f"Najlepszy wynik: {grid_search.best_score_:.4f}")

    # Najlepszy model
    best_log_reg = grid_search.best_estimator_

    # Ewaluacja
    print("\nWyniki modelu:")
    metrics_log_reg = evaluate_model(best_log_reg, X_test, y_test, X_val, y_val, "logistic_regression" if save_plots else None)
    
    for metric, value in metrics_log_reg.items():
        print(f'{metric}: {value}')



def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, save_plots=False):
    """
    Trenuje model lasów losowych z optymalizacją hiperparametrów za pomocą GridSearchCV.
    """
    # Parametry do przeszukania
    param_grid = {
        'n_estimators': [100],        # Liczba drzew
        'max_depth': [10],            # Maksymalna głębokość
        'min_samples_split': [2],     # Min. liczba próbek do podziału węzła
        'min_samples_leaf': [2],      # Min. liczba próbek w liściu
        'max_features': ['sqrt']     # Liczba cech do rozważenia przy podziale
    }

    # Inicjalizacja modelu
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # GridSearch z cross-walidacją
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Trenowanie na danych
    grid_search.fit(X_train, y_train)

    # Najlepsze parametry i wynik
    print(f"Najlepsze parametry: {grid_search.best_params_}")
    print(f"Najlepszy wynik: {grid_search.best_score_:.4f}")

    # Najlepszy model
    best_rf = grid_search.best_estimator_

    # Ewaluacja modelu
    print("\nWyniki modelu:")
    metrics_rf = evaluate_model(best_rf, X_test, y_test, X_val, y_val, "random_forest" if save_plots else None)
    
    for metric, value in metrics_rf.items():
        print(f'{metric}: {value}')



def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, save_plots=False):
    """
    Trenuje model XGBoost z optymalizacją hiperparametrów za pomocą GridSearchCV.
    """
    # Parametry do przeszukania
    param_grid = {
        'n_estimators': [100, 200, 300],        # Liczba drzew boosting
        'max_depth': [3, 6, 9],                  # Maksymalna głębokość drzewa
        'learning_rate': [0.01, 0.1, 1],        # Szybkość uczenia
        'subsample': [0.8, 1.0],                 # Odsetek próbek używanych dla każdego drzewa
        'colsample_bytree': [0.6, 0.8, 1.0]     # Odsetek cech używanych dla każdego drzewa
    }

    # Inicjalizacja modelu
    xgb = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,  
        objective='binary:logistic'
    )

    # GridSearch z cross-walidacją
    grid_search = GridSearchCV(
        xgb, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Trenowanie na danych
    grid_search.fit(X_train, y_train)

    # Najlepsze parametry i wynik
    print(f"Najlepsze parametry: {grid_search.best_params_}")
    print(f"Najlepszy wynik: {grid_search.best_score_:.4f}")

    # Najlepszy model
    best_xgb = grid_search.best_estimator_

    # Ewaluacja modelu
    print("\nWyniki modelu:")
    metrics_xgb = evaluate_model(best_xgb, X_test, y_test, X_val, y_val, "xgboost" if save_plots else None)
    
    for metric, value in metrics_xgb.items():
        print(f'{metric}: {value}')

