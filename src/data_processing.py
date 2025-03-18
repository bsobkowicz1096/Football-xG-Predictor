import os
import pandas as pd

def load_data(file_path):
    """Wczytuje dane z pliku CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"Wczytano dane: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Nie znaleziono pliku: {file_path}")
        return None

def clean_data(df):
    """
    Usuwa kolumny z samymi wartościami NaN.

    """
    # Zapisz oryginalne kolumny
    original_columns = set(df.columns)
    
    # Usunięcie kolumn z samymi NaN
    df_cleaned = df.dropna(axis=1, how='all')
    
    # Sprawdź, które kolumny zostały zachowane
    remaining_columns = set(df_cleaned.columns)
    
    print(f"Usunięto {len(original_columns) - len(remaining_columns)} kolumn z samymi NaN")
    print(f"Nowy wymiar danych: {df_cleaned.shape}")
    
    print(f"Pozostało {len(remaining_columns)} kolumn:")
    for col in sorted(remaining_columns):
        print(f"- {col}")
    
    return df_cleaned

def select_features(df, features):
    """
    Wybiera określone cechy z dataframe.

    """
    df_selected = df[features]
    print(f"Wybrano {len(features)} cech")
    print(f"Nowy wymiar danych: {df_selected.shape}")
    return df_selected
