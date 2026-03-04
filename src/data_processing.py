import os
import pandas as pd

def load_data(file_path):
    """Loads data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def clean_data(df):
    """
    Removes columns with all NaN values.
    """
    # Save original columns
    original_columns = set(df.columns)
    
    # Remove columns with all NaN
    df_cleaned = df.dropna(axis=1, how='all')
    
    # Check which columns were kept
    remaining_columns = set(df_cleaned.columns)
    
    print(f"Removed {len(original_columns) - len(remaining_columns)} columns with all NaN")
    print(f"New data dimensions: {df_cleaned.shape}")
    
    print(f"Remaining {len(remaining_columns)} columns:")
    for col in sorted(remaining_columns):
        print(f"- {col}")
    
    return df_cleaned

def select_features(df, features):
    """
    Selects specified features from dataframe.
    """
    df_selected = df[features]
    print(f"Selected {len(features)} features")
    print(f"New data dimensions: {df_selected.shape}")
    return df_selected