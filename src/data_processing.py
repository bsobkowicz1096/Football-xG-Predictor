import io
import os
import contextlib
import numpy as np
import pandas as pd

from feature_engineering import (
    transform_to_binary, extract_xy, calculate_angles_distances,
    transform_body_part, analyze_freeze_frame, standardize_features,
    create_dummies, transform_play_pattern,
)


def load_data(file_path):
    """Loads data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def clean_data(df, verbose=True):
    """Removes columns with all NaN values."""
    original_columns = set(df.columns)
    df_cleaned = df.dropna(axis=1, how='all')
    remaining_columns = set(df_cleaned.columns)

    if verbose:
        print(f"Removed {len(original_columns) - len(remaining_columns)} columns with all NaN")
        print(f"New data dimensions: {df_cleaned.shape}")
        print(f"Remaining {len(remaining_columns)} columns:")
        for col in sorted(remaining_columns):
            print(f"- {col}")

    return df_cleaned


def select_features(df, features):
    """Selects specified features from dataframe."""
    df_selected = df[features]
    print(f"Selected {len(features)} features")
    print(f"New data dimensions: {df_selected.shape}")
    return df_selected


def prepare_test_set(path, continuous_vars, feature_columns):
    """
    Loads FIFA World Cup 2022 shot data and applies the full feature engineering
    pipeline, mirroring the transformations applied to the training data.

    Parameters
    ----------
    path : str or Path
        Path to the FIFA 2022 CSV file.
    continuous_vars : list of str
        Continuous variable names passed to standardize_features.
    feature_columns : list of str
        Ordered list of model feature column names (X_train.columns).

    Returns
    -------
    X_test : pd.DataFrame
    y_test : pd.Series
    df_spatial : pd.DataFrame
        x, y shot coordinates (saved before feature selection, used for xG scatter plot).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_data(path)
        df = clean_data(df, verbose=False)

        df = df[df['shot_type'] != 'Penalty']
        df = transform_to_binary(df, column='shot_outcome',   positive_value='Goal')
        df = extract_xy(df)
        df = calculate_angles_distances(df)
        df = transform_body_part(df)
        df = transform_to_binary(df, column='under_pressure',  positive_value=True)
        df = transform_to_binary(df, column='shot_first_time', positive_value=True)
        df['normal_shot']    = np.where(df['shot_technique'] == 'Normal', 1, 0)
        df['open_play_shot'] = np.where(df['shot_type'] == 'Open Play', 1, 0)
        df = analyze_freeze_frame(df)
        df['defenders_in_path']      = np.where(df['defenders_in_path'] >= 3, 3, df['defenders_in_path'])
        df['log_passes_in_sequence'] = np.log1p(df['n_passes_in_sequence'])
        df['log_angle']              = np.log1p(df['angle'])
        df = standardize_features(df, continuous_vars)
        df = create_dummies(df, 'refined_body_part', drop_category='head')
        df = transform_play_pattern(df)

    df_spatial = df[['x', 'y']].copy()
    df = df[['shot_outcome'] + list(feature_columns)]

    X_test = df[list(feature_columns)]
    y_test = df['shot_outcome']

    print(f"Test set (FIFA 2022):  {X_test.shape[0]} shots | Goal rate: {y_test.mean():.2f}")
    return X_test, y_test, df_spatial
