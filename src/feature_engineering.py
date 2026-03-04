import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ast
import re   

def transform_to_binary(df, column, positive_value, display_distribution=True):
    """
    Transforms categorical variable to binary variable.
    """
    # DataFrame copy
    df_result = df.copy()
    
    # Display original distribution
    if display_distribution:
        print(f"Original distribution of {column}:")
        print(df_result[column].value_counts(normalize=True, dropna=False).multiply(100).round(2))
    
    # Transform to binary variable
    df_result[column] = np.where(df_result[column]==positive_value, 1, 0)
    
    # Display new distribution
    if display_distribution:
        print(f"\nDistribution of {column} after transformation:")
        print(df_result[column].value_counts(normalize=True, dropna=False).multiply(100).round(2))
    
    return df_result

def analyze_categorical_variables(df, columns=None, separator='-'*50):
    """
    Analyzes value distribution in categorical variables.
    """
    results = {}
    
    # If no columns specified, use all
    if columns is None:
        columns = df.columns
        
    for column in columns:
        if column in df.columns:
            # Calculate percentage distribution
            distribution = df[column].value_counts(normalize=True, dropna=False).multiply(100).round(2)
            results[column] = distribution
            
            # Display results
            print(f"Distribution for variable '{column}':")
            print(distribution)
            print(separator)
    
    return results

def extract_xy(df):
    """Extracts x, y coordinates from location variable."""
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
    """Calculates shot distance and angle."""
    # Distance from goal center
    df['distance'] = np.sqrt((df['distance_to_end_line'])**2 + (abs(40-df['y']))**2)
    
    # Goal post coordinates
    left_post = [120, 36]
    right_post = [120, 44]
    
    # Calculate shot angle
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
    Transforms body part information used for shot into more useful category
    identifying whether shot was taken with stronger/weaker foot or head.
    """
    # DataFrame copy
    df_result = df.copy()
    
    # Step 1: Foot encoding (1 for right, 0 for left)
    foot_encoded = df_result[df_result[body_part_col].isin(['Right Foot', 'Left Foot'])].copy()
    foot_encoded['is_right_foot'] = (foot_encoded[body_part_col] == 'Right Foot').astype(int)
    
    # Step 2: Calculate dominant foot for each player
    player_foot_dominance = foot_encoded.groupby(player_id_col)['is_right_foot'].mean().reset_index()
    
    # Step 3: Determine dominant foot
    player_foot_dominance['foot_type'] = 'dominant_right'
    player_foot_dominance.loc[player_foot_dominance['is_right_foot'] <= 0.5, 'foot_type'] = 'dominant_left'
    
    # Step 4: Merge foot type information with original dataset
    df_result = df_result.merge(player_foot_dominance[[player_id_col, 'foot_type']], on=player_id_col, how='left')
    
    # Step 5: Create new refined_body_part variable
    df_result['refined_body_part'] = df_result[body_part_col]
    
    # Update for foot shots
    mask_right_foot = df_result[body_part_col] == 'Right Foot'
    mask_left_foot = df_result[body_part_col] == 'Left Foot'
    
    # Dominant or non-dominant foot shots
    df_result.loc[mask_right_foot & (df_result['foot_type'] == 'dominant_right'), 'refined_body_part'] = 'better_foot'
    df_result.loc[mask_left_foot & (df_result['foot_type'] == 'dominant_left'), 'refined_body_part'] = 'better_foot'
    df_result.loc[mask_right_foot & (df_result['foot_type'] == 'dominant_left'), 'refined_body_part'] = 'worse_foot'
    df_result.loc[mask_left_foot & (df_result['foot_type'] == 'dominant_right'), 'refined_body_part'] = 'worse_foot'
    
    # Head shots
    df_result.loc[df_result[body_part_col] == 'Head', 'refined_body_part'] = 'head'
    
    # Filter DataFrame, keeping only rows with selected values
    df_result = df_result[df_result['refined_body_part'].isin(['better_foot', 'worse_foot', 'head'])]
    
    # Display statistics
    print(f"Distribution of dominant foot among players:")
    print(player_foot_dominance['foot_type'].value_counts(normalize=True).multiply(100).round(2))
    print(50*'-')
    print(f"Distribution of shots by type:")
    print(df_result['refined_body_part'].value_counts(normalize=True).multiply(100).round(2))
    
    return df_result

def analyze_freeze_frame(df, freeze_frame_col='shot_freeze_frame', x_col='x', y_col='y'):
    """
    Analyzes player positioning (shot_freeze_frame) at moment of shot 
    and determines:
    - number of defenders on shot line
    - goalkeeper presence on shot line
    - goalkeeper distance ratio (from goal center toward shooter)
    """
    # DataFrame copy
    df_result = df.copy()
    
    # Constants
    GOAL_X = 120
    GOAL_CENTER_Y = 40
    GOAL_LEFT = [120, 36]
    GOAL_RIGHT = [120, 44]
    
    # Parse freeze_frame from string to Python structure
    def parse_freeze_frame(freeze_frame_str):
        if isinstance(freeze_frame_str, str):
            try:
                return ast.literal_eval(freeze_frame_str)
            except (ValueError, SyntaxError):
                return None
        return freeze_frame_str
    
    df_result[freeze_frame_col] = df_result[freeze_frame_col].apply(parse_freeze_frame)
    
    # Main analysis function
    def analyze_shot_context(freeze_frame, shooter_x, shooter_y):
        """
        Analyzes shot context from freeze_frame data.
        Returns: defenders_in_path, goalkeeper_in_path, goalkeeper_distance_ratio
        """
        # If no freeze_frame data, return None for all
        if freeze_frame is None or not isinstance(freeze_frame, list):
            return None, None, None
        
        shooter_loc = [shooter_x, shooter_y]
        
        # Distance from shooter to goal center
        shot_distance_from_goal = np.sqrt((GOAL_X - shooter_x)**2 + 
                                          (GOAL_CENTER_Y - shooter_y)**2)
        
        # Helper function to check if point is left of line
        def is_left(p0, p1, p2):
            return ((p1[0] - p0[0])*(p2[1] - p0[1]) - (p1[1] - p0[1])*(p2[0] - p0[0])) > 0
        
        # Initialize variables
        defenders_count = 0
        goalkeeper_in_path = 0
        goalkeeper_loc = None
        
        # Iterate through all players in freeze_frame
        for player in freeze_frame:
            # Skip shooting team players or players without location
            if player.get('teammate', True) or 'location' not in player:
                continue
            
            player_loc = player['location']
            position_name = player.get('position', {}).get('name', '')
            
            # Check if player is on shot line (in triangle)
            is_in_triangle = (is_left(shooter_loc, GOAL_LEFT, player_loc) != 
                             is_left(shooter_loc, GOAL_RIGHT, player_loc))
            
            # Process goalkeeper
            if position_name == 'Goalkeeper':
                goalkeeper_loc = player_loc
                if is_in_triangle and player_loc[0] > shooter_x:
                    goalkeeper_in_path = 1
            # Process defenders
            else:
                # Count defenders in shot path
                if is_in_triangle and player_loc[0] > shooter_x:
                    defenders_count += 1
        
        # Calculate goalkeeper_distance_ratio (from GOAL CENTER)
        goalkeeper_distance_ratio = None
        if goalkeeper_loc is not None and shot_distance_from_goal > 0:
            # Distance from goalkeeper to goal center
            gk_distance_from_goal_center = np.sqrt((GOAL_X - goalkeeper_loc[0])**2 + 
                                                   (GOAL_CENTER_Y - goalkeeper_loc[1])**2)
            
            # How far GK advanced toward shooter (as ratio)
            # 0 = GK at goal center, 1 = GK at shooter position
            goalkeeper_distance_ratio = gk_distance_from_goal_center / shot_distance_from_goal
            goalkeeper_distance_ratio = np.clip(goalkeeper_distance_ratio, 0, 1)

        return defenders_count, goalkeeper_in_path, goalkeeper_distance_ratio
    
    # Apply analysis to each DataFrame row
    results = df_result.apply(
        lambda row: analyze_shot_context(
            row[freeze_frame_col], 
            row[x_col], 
            row[y_col]
        ), 
        axis=1
    )
    
    # Split results into separate columns
    df_result['defenders_in_path'] = results.apply(lambda x: x[0] if x is not None else None)
    df_result['goalkeeper_in_path'] = results.apply(lambda x: x[1] if x is not None else None)
    df_result['goalkeeper_distance_ratio'] = results.apply(lambda x: x[2] if x is not None else None)

    df_result['goalkeeper_distance_ratio'] = df_result['goalkeeper_distance_ratio'].fillna(df_result['goalkeeper_distance_ratio'].median())
    
    # Results summary
    print("="*70)
    print("DEFENDERS ON SHOT LINE:")
    print("="*70)
    print(df_result['defenders_in_path'].value_counts(normalize=True).multiply(100).round(2).sort_index())
    
    print("\n" + "="*70)
    print("GOALKEEPER ON SHOT LINE:")
    print("="*70)
    print(df_result['goalkeeper_in_path'].value_counts(normalize=True).multiply(100).round(2).sort_index())
    
    print("\n" + "="*70)
    print("GOALKEEPER DISTANCE RATIO:")
    print("="*70)
    print(df_result['goalkeeper_distance_ratio'].describe())
    
    return df_result
    
def standardize_features(df, continuous_vars):
    """
    Standardizes selected continuous variables in dataframe.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[continuous_vars])
    df[[f'{var}_scaled' for var in continuous_vars]] = scaled_features
    return df

def transform_play_pattern(df):
    """
    Groups play_pattern into 4 categories and creates dummy variables.
    Reference category (dropped): open_play.

    Groups
    ------
    corner   : From Corner
    set_piece: From Free Kick
    counter  : From Counter
    open_play: everything else (Regular Play, Throw In, Goal Kick,
               From Keeper, From Kick Off, Other)
    """
    mapping = {
        'From Corner':   'corner',
        'From Free Kick': 'set_piece',
        'From Counter':  'counter',
    }
    df['play_pattern_group'] = df['play_pattern'].map(mapping).fillna('open_play')
    df = create_dummies(df, 'play_pattern_group', drop_category='open_play')
    df = df.drop('play_pattern', axis=1)
    return df


def create_dummies(df, column_name, drop_category=None):
    """
    Creates dummy variables for selected column in dataframe.
    """
    dummies = pd.get_dummies(df[column_name]).astype(int)
    if drop_category:
        dummies = dummies.drop(drop_category, axis=1)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column_name, axis=1)
    return df

