import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import ast
import re   

def transform_to_binary(df, column, positive_value, display_distribution=True):
    """
    Przekształca zmienną kategoryczną na zmienną binarną.
    
    """
    # Kopia dataframe
    df_result = df.copy()
    
    # Wyświetlenie oryginalnego rozkładu
    if display_distribution:
        print(f"Oryginalny rozkład {column}:")
        print(df_result[column].value_counts(normalize=True, dropna=False).multiply(100).round(2))
    
    # Przekształcenie na zmienną binarną
    df_result[column] = np.where(df_result[column]==positive_value, 1, 0)
    
    # Wyświetlenie nowego rozkładu
    if display_distribution:
        print(f"\nRozkład {column} po przekształceniu:")
        print(df_result[column].value_counts(normalize=True, dropna=False).multiply(100).round(2))
    
    return df_result

def analyze_categorical_variables(df, columns=None, separator='-'*50):
    """
    Analizuje rozkład wartości w zmiennych kategorycznych.

    """
    results = {}
    
    # Jeśli nie podano kolumn, użyj wszystkich
    if columns is None:
        columns = df.columns
        
    for column in columns:
        if column in df.columns:
            # Obliczanie rozkładu procentowego
            distribution = df[column].value_counts(normalize=True, dropna=False).multiply(100).round(2)
            results[column] = distribution
            
            # Wyświetlanie wyników
            print(f"Rozkład dla zmiennej '{column}':")
            print(distribution)
            print(separator)
    
    return results

def extract_xy(df):
    """Ekstrahuje współrzędne x, y ze zmiennej location."""
    def extract_coords(loc_str):
        numbers = re.findall(r"\d*\.\d+|\d+", loc_str)
        if len(numbers) == 2:
            return float(numbers[0]), float(numbers[1])
        else:
            return None, None
    
    df[['x', 'y']] = df['location'].apply(extract_coords).apply(pd.Series)
    df['distance_to_end_line'] = 120 - df['x']
    
    return df

def calculate_angles_distances(df):
    """Oblicza odległość i kąt strzału."""
    # Odległość od środka bramki
    df['distance'] = np.sqrt((df['distance_to_end_line'])**2 + (abs(40-df['y']))**2)
    
    # Współrzędne słupków bramki
    left_post = [120, 36]
    right_post = [120, 44]
    
    # Obliczanie kąta strzału
    def calculate_shot_angle(x, y):
        vector_to_left = [left_post[0] - x, left_post[1] - y]
        vector_to_right = [right_post[0] - x, right_post[1] - y]
        
        mag_left = np.sqrt(vector_to_left[0]**2 + vector_to_left[1]**2)
        mag_right = np.sqrt(vector_to_right[0]**2 + vector_to_right[1]**2)
        
        dot_product = vector_to_left[0]*vector_to_right[0] + vector_to_left[1]*vector_to_right[1]
        
        angle_rad = np.arccos(dot_product / (mag_left * mag_right))
        angle_deg = angle_rad * 180 / np.pi
        
        return angle_deg
    
    df['angle'] = df.apply(lambda row: calculate_shot_angle(row['x'], row['y']), axis=1)
    
    return df

def transform_body_part(df, body_part_col='shot_body_part', player_id_col='player_id'):
    """
    Przekształca informacje o części ciała użytej do strzału na bardziej użyteczną kategorię
    identyfikującą czy strzał był oddany lepszą/gorszą nogą lub głową.

    """
    # Kopia DataFrame
    df_result = df.copy()
    
    # Krok 1: Kodowanie nóg (1 dla prawej, 0 dla lewej)
    foot_encoded = df_result[df_result[body_part_col].isin(['Right Foot', 'Left Foot'])].copy()
    foot_encoded['is_right_foot'] = (foot_encoded[body_part_col] == 'Right Foot').astype(int)
    
    # Krok 2: Obliczenie dominującej nogi dla każdego zawodnika
    player_foot_dominance = foot_encoded.groupby(player_id_col)['is_right_foot'].mean().reset_index()
    
    # Krok 3: Określenie dominującej nogi
    player_foot_dominance['foot_type'] = 'dominant_right'
    player_foot_dominance.loc[player_foot_dominance['is_right_foot'] <= 0.5, 'foot_type'] = 'dominant_left'
    
    # Krok 4: Połączenie informacji o typie nogi z oryginalnym zbiorem danych
    df_result = df_result.merge(player_foot_dominance[[player_id_col, 'foot_type']], on=player_id_col, how='left')
    
    # Krok 5: Utworzenie nowej zmiennej refined_body_part
    df_result['refined_body_part'] = df_result[body_part_col]
    
    # Aktualizacja dla strzałów nogą
    mask_right_foot = df_result[body_part_col] == 'Right Foot'
    mask_left_foot = df_result[body_part_col] == 'Left Foot'
    
    # Strzały dominującą lub niedominującą nogą
    df_result.loc[mask_right_foot & (df_result['foot_type'] == 'dominant_right'), 'refined_body_part'] = 'better_foot'
    df_result.loc[mask_left_foot & (df_result['foot_type'] == 'dominant_left'), 'refined_body_part'] = 'better_foot'
    df_result.loc[mask_right_foot & (df_result['foot_type'] == 'dominant_left'), 'refined_body_part'] = 'worse_foot'
    df_result.loc[mask_left_foot & (df_result['foot_type'] == 'dominant_right'), 'refined_body_part'] = 'worse_foot'
    
    # Strzały głową
    df_result.loc[df_result[body_part_col] == 'Head', 'refined_body_part'] = 'head'
    
    # Filtrowanie DataFrame, zostawiając tylko wiersze z wybranymi wartościami
    df_result = df_result[df_result['refined_body_part'].isin(['better_foot', 'worse_foot', 'head'])]
    
    # Wyświetlenie statystyk
    print(f"Rozkład dominującej nogi wśród zawodników:")
    print(player_foot_dominance['foot_type'].value_counts(normalize=True).multiply(100).round(2))
    print(50*'-')
    print(f"Rozkład strzałów wg rodzaju:")
    print(df_result['refined_body_part'].value_counts(normalize=True).multiply(100).round(2))
    
    return df_result

def analyze_freeze_frame(df, freeze_frame_col='shot_freeze_frame', x_col='x', y_col='y'):
    """
    Analizuje ustawienie zawodników (shot_freeze_frame) w momencie oddania strzału 
    i określa liczbę obrońców oraz obecność bramkarza na linii strzału.

    """
    # Kopia DataFrame
    df_result = df.copy()
    
    # Parsowanie freeze_frame ze stringa na strukturę Pythona
    def parse_freeze_frame(freeze_frame_str):
        if isinstance(freeze_frame_str, str):
            try:
                return ast.literal_eval(freeze_frame_str)
            except (ValueError, SyntaxError):
                return None
        return freeze_frame_str
    
    df_result[freeze_frame_col] = df_result[freeze_frame_col].apply(parse_freeze_frame)
    
    # Funkcja analizująca ścieżkę strzału
    def analyze_shot_path(freeze_frame, shooter_x, shooter_y, goal_left=[120, 36], goal_right=[120, 44]):
        # Jeśli brak danych o freeze_frame, zwróć None
        if freeze_frame is None:
            return None, None
        
        shooter_loc = [shooter_x, shooter_y]
        
        # Funkcja pomocnicza do sprawdzenia czy punkt jest po lewej stronie linii
        def is_left(p0, p1, p2):
            return ((p1[0] - p0[0])*(p2[1] - p0[1]) - (p1[1] - p0[1])*(p2[0] - p0[0])) > 0
        
        # Liczniki
        defenders_count = 0
        goalkeeper_in_path = 0
        
        for player in freeze_frame:
            # Pomijanie graczy z drużyny strzelającej lub bez lokalizacji
            if player.get('teammate', True) == True or 'location' not in player:
                continue
            
            # Sprawdzenie lokalizacji względem trójkąta strzału
            is_inside = (is_left(shooter_loc, goal_left, player['location']) != 
                         is_left(shooter_loc, goal_right, player['location']))
            
            # Zliczanie obrońców i bramkarza na linii strzału
            if is_inside and player['location'][0] > shooter_loc[0]:
                if player.get('position', {}).get('name') == 'Goalkeeper':
                    goalkeeper_in_path = 1
                else:
                    defenders_count += 1
        
        return defenders_count, goalkeeper_in_path
    
    # Aplikowanie analizy do każdego wiersza DataFrame
    result = df_result.apply(
        lambda row: analyze_shot_path(
            row[freeze_frame_col], 
            row[x_col], 
            row[y_col]
        ), 
        axis=1
    )
    
    # Rozdzielenie wyników na dwie kolumny
    df_result['defenders_in_path'] = result.apply(lambda x: x[0] if x is not None else None)
    df_result['goalkeeper_in_path'] = result.apply(lambda x: x[1] if x is not None else None)
    
    # Podsumowanie wyników
    print("Częstości obrońców na linii strzału:")
    print(df_result['defenders_in_path'].value_counts(normalize=True).multiply(100).round(2))
    print("\nCzęstości obecności bramkarza na linii strzału:")
    print(df_result['goalkeeper_in_path'].value_counts(normalize=True).multiply(100).round(2))
    
    return df_result

def standardize_features(df, continuous_vars):
    """
    Standaryzuje wybrane zmienne ciągłe w dataframe.

    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[continuous_vars])
    df[[f'{var}_scaled' for var in continuous_vars]] = scaled_features
    return df

def create_dummies(df, column_name, drop_category=None):
    """
    Tworzy zmienne dummy dla wybranej kolumny w dataframe.
    """
    dummies = pd.get_dummies(df[column_name]).astype(int)
    if drop_category:
        dummies = dummies.drop(drop_category, axis=1)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column_name, axis=1)
    return df

def calculate_vif(df, features):
    """
    Oblicza VIF dla podanych zmiennych w dataframe.
    """
    X = df[features]
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values("VIF", ascending=False)