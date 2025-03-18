import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve
from mplsoccer import VerticalPitch
from matplotlib.lines import Line2D

def create_quantile_efficiency(df, quantile_col, target_col, num_quantiles=15):
    """
    Tworzy kwantyle dla wybranej zmiennej i oblicza średnią wartość zmiennej docelowej dla każdego kwantyla.
    """
    # Tworzenie nowej kolumny z kwantylami
    quantile_name = f"{quantile_col}_quantile"
    df_result = df.copy()
    df_result[quantile_name] = pd.qcut(df_result[quantile_col], q=num_quantiles, labels=False)
    
    # Obliczanie średniej wartości zmiennej docelowej dla każdego kwantyla
    df_efficiency = df_result.groupby(quantile_name)[target_col].mean().reset_index()
    df_efficiency['count'] = df_result.groupby(quantile_name)[target_col].count().values
    
    return df_efficiency

def plot_shot_accuracy_by_distance_and_y(eff_by_distance, eff_by_y, save_path=None):
    """
    Rysuje wykresy porównujące skuteczność strzału względem odległości od linii końcowej i pozycji Y.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(x=eff_by_distance['distance_to_end_line_quantile'].astype(str), 
                y=eff_by_distance['shot_outcome'], 
                ax=axes[0])
    axes[0].set_title('Skuteczność strzału a odległość od linii końcowej (kwantyle)', size=16)
    axes[0].set_xlabel('Kwantyl odległości od linii końcowej', size=13)
    axes[0].set_ylabel('Skuteczność strzału', size=13)

    sns.barplot(x=eff_by_y['y_quantile'].astype(str), 
                y=eff_by_y['shot_outcome'], 
                ax=axes[1])
    axes[1].set_title('Skuteczność strzału a pozycja Y (kwantyle)', size=16)
    axes[1].set_xlabel('Kwantyl pozycji Y', size=13)
    axes[1].set_ylabel('Skuteczność strzału', size=13)

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres do {save_path}")
        
    return fig, axes

def visualize_shot_situation(df, shot_index, save_path=None):
    """
    Wizualizuje sytuację konkretnego strzału z uwzględnieniem ustawienia zawodników,
    trójkąta strzału i informacji o obrońcach na linii strzału.
    """
    # Pobranie danych o ustawieniu zawodników
    freeze_frame = df.loc[shot_index, 'shot_freeze_frame']
    
    # Pobieranie lokalizacji strzelca i bramki
    striker_x = df.loc[shot_index, 'x']
    striker_y = df.loc[shot_index, 'y']
    goal_left = [120, 36]
    goal_right = [120, 44]
    
    # Pobranie wcześniej obliczonych wartości
    defenders_count = df.loc[shot_index, 'defenders_in_path']
    goalkeeper_in_path = df.loc[shot_index, 'goalkeeper_in_path']
    
    # Tworzenie wykresu
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='grass', line_color='white', 
                         goal_type='line', half=True)
    fig, ax = pitch.draw(figsize=(8, 10))
    
    # Rysowanie zawodników
    for player in freeze_frame:
        if 'location' not in player:
            continue
            
        x, y = player['location']
        
        # Bez zawodników z własnej połowy
        if x < 60:
            continue
        
        if player.get('teammate', False):
            jersey_color = 'blue'
        else:
            if player.get('position', {}).get('name') == 'Goalkeeper':
                jersey_color = 'yellow'
            else:
                jersey_color = 'red'
        
        ax.plot(y, x, 'o', markersize=10, color=jersey_color, zorder=2)
        
    # Rysowanie strzelca i linii strzału
    if striker_x >= 60:
        ax.plot(striker_y, striker_x, 'o', markersize=12, color='lime', zorder=4)
        ax.text(striker_y, striker_x - 3, f"STRZELEC", ha='center', fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', pad=1), zorder=5)
        
        # Linie strzału
        ax.plot([striker_y, goal_left[1]], [striker_x, goal_left[0]], 'k--', alpha=0.7, zorder=1)
        ax.plot([striker_y, goal_right[1]], [striker_x, goal_right[0]], 'k--', alpha=0.7, zorder=1)
        
        # Trójkąt strzału
        triangle = plt.Polygon(
            [[striker_y, striker_x], [goal_left[1], goal_left[0]], [goal_right[1], goal_right[0]]],
            alpha=0.4, color='yellow', zorder=0
        )
        ax.add_patch(triangle)
    
    # Tytuł i legenda
    shot_outcome = "Gol" if df.loc[shot_index, 'shot_outcome'] == 1 else "Brak gola"
    title = f"Strzał #{shot_index} | Wynik: {shot_outcome} | Obrońcy: {defenders_count} | Bramkarz: {'Tak' if goalkeeper_in_path == 1 else 'Nie'}"
    ax.set_title(title)
    
    # Legenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Drużyna strzelająca'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Drużyna broniąca'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Bramkarz'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Strzelec')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano wizualizację strzału do {save_path}")
        
    return fig, ax

def plot_binary_features_comparison(df, target_col='shot_outcome', features=None, figsize=(10, 8), save_path=None):
    """
    Tworzy wykresy słupkowe porównujące skuteczność strzałów dla różnych binarnych cech.
    """
    if features is None:
        features = ['under_pressure', 'shot_first_time', 'normal_shot', 'open_play_shot', 'goalkeeper_in_path']
    
    # Obliczenie liczby wierszy i kolumn dla subplotów
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  # Zaokrąglenie w górę
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs_flat = axs.flatten()
    
    for i, feature in enumerate(features):
        if i < len(axs_flat):
            # Obliczenie średnich wartości celu dla każdej kategorii
            goal_rate_0 = df[df[feature] == 0][target_col].mean() * 100
            goal_rate_1 = df[df[feature] == 1][target_col].mean() * 100
            
            # Wykres słupkowy
            axs_flat[i].bar(['Nie', 'Tak'], [goal_rate_0, goal_rate_1], color='blue', alpha=0.8)
            axs_flat[i].set_title(f'Procent goli: {feature}')
    
    # Ukrycie pustych wykresów
    for i in range(n_features, len(axs_flat)):
        axs_flat[i].set_visible(False)
    
    # Dodanie siatki i dostosowanie layoutu
    for ax in axs_flat[:n_features]:
        ax.grid(axis='y', linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano porównanie cech binarnych do {save_path}")
        
    return fig, axs

def plot_stacked_bar(df, group_col, title, xlabel, save_path=None):
    """
    Tworzy wykresy słupkowe porównujące skuteczność strzałów dla różnych zmiennych kategorialnych.
    """    
    # Grupowanie danych według wybranej kolumny i wyniku strzału
    counts = df.groupby([group_col, 'shot_outcome']).size().unstack(fill_value=0)
    
    # Zmiana nazw kolumn jeśli mamy tylko 0 i 1
    if set(counts.columns) == {0, 1}:
        counts.columns = ['Nie-gol', 'Gol']
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Skumulowany wykres słupkowy
    counts.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green'], alpha=0.8)

    # Etykiety i tytuł
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Liczba strzałów', fontsize=12)

    # Procenty goli nad każdym słupkiem
    for i, idx in enumerate(counts.index):
        total = counts.loc[idx].sum()
        if total > 0:  # bezpiecznik przed dzieleniem przez zero
            goal_rate = counts.loc[idx, 'Gol'] / total * 100
            ax.text(i, total + 5, f"{goal_rate:.1f}%", ha='center', fontweight='bold', size=12)

    ax.legend(title='Wynik strzału')

    plt.grid(axis='y', linestyle='--', alpha=0.8)
    plt.xticks(rotation=45 if group_col == 'refined_body_part' else 0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykres słupkowy dla {group_col} do {save_path}")
        
    return fig, ax

def plot_shot_success_heatmap(df, quantiles=5, show_corr=False, save_path=None):
    """
    Wizualizuje skuteczność strzałów w zależności od kwantyli dystansu i kąta.
    """
    # Podział na kwantyle
    df['distance_quantile'] = pd.qcut(df['distance'], q=quantiles, labels=False)
    df['angle_quantile'] = pd.qcut(df['angle'], q=quantiles, labels=False)

    # Średnia skuteczność w siatce (dystans vs kąt)
    df_heatmap = df.groupby(['distance_quantile', 'angle_quantile'])['shot_outcome'].mean().unstack()

    # Wizualizacja
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = sns.heatmap(df_heatmap, annot=True, cmap='RdYlGn', fmt=".2f", ax=ax)
    ax.set_title('Skuteczność strzałów w zależności od dystansu i kąta')
    ax.set_xlabel('Kwantyl kąta strzału')
    ax.set_ylabel('Kwantyl dystansu')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano heatmapę skuteczności strzałów do {save_path}")

    # Opcjonalnie: korelacja
    if show_corr:
        corr = df[['angle', 'distance']].corr()
        print(corr)
        
    return fig, ax

def plot_shot_effectiveness_by_quantiles(df, quantiles=15, save_path=None):
    """
    Wizualizuje skuteczność strzałów w podziale na kwantyle dystansu i kąta strzału.
    """
    # Kwantyle dla dystansu i kąta
    df['distance_quantile'] = pd.qcut(df['distance'], q=quantiles, labels=False)
    df['angle_quantile'] = pd.qcut(df['angle'], q=quantiles, labels=False)

    # Skuteczność strzałów dla dystansu
    df_eff_distance = df.groupby('distance_quantile')['shot_outcome'].mean().reset_index()
    df_eff_distance['count'] = df.groupby('distance_quantile')['shot_outcome'].count().values

    # Skuteczność strzałów dla kąta
    df_eff_angle = df.groupby('angle_quantile')['shot_outcome'].mean().reset_index()
    df_eff_angle['count'] = df.groupby('angle_quantile')['shot_outcome'].count().values

    # Wizualizacja
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x=df_eff_distance['distance_quantile'].astype(str), 
                y=df_eff_distance['shot_outcome'], ax=axes[0])
    axes[0].set_title('Skuteczność strzału a dystans (kwantyle)')
    axes[0].set_xlabel('Kwantyl dystansu')
    axes[0].set_ylabel('Skuteczność strzału')

    sns.barplot(x=df_eff_angle['angle_quantile'].astype(str), 
                y=df_eff_angle['shot_outcome'], ax=axes[1])
    axes[1].set_title('Skuteczność strzału a kąt strzału (kwantyle)')
    axes[1].set_xlabel('Kwantyl kąta strzału')
    axes[1].set_ylabel('Skuteczność strzału')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykresy skuteczności wg kwantyli do {save_path}")
        
    return fig, axes

def plot_body_part_by_distance_quantile(df, quantiles=10, save_path=None):
    """
    Wizualizuje procentowy rozkład części ciała użytych do strzału 
    w podziale na kwantyle dystansu.
    """
    # Podział dystansu na kwantyle
    df['distance_quantile'] = pd.qcut(df['distance'], q=quantiles, labels=False)

    # Analiza procentowa
    quantile_distribution = df.groupby(['distance_quantile', 'refined_body_part']).size().unstack(fill_value=0)
    quantile_percentage = quantile_distribution.div(quantile_distribution.sum(axis=1), axis=0) * 100

    # Wizualizacja
    fig, ax = plt.subplots(figsize=(10, 6))
    quantile_percentage.plot(kind='bar', stacked=True, alpha=0.8, ax=ax)
    ax.set_title('Procentowy udział części ciała w strzałach według kwantyli dystansu')
    ax.set_xlabel('Kwantyl dystansu')
    ax.set_ylabel('Procent strzałów (%)')
    ax.legend(title='Część ciała', loc='lower right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano wykresy rozkładu części ciała wg dystansu do {save_path}")
        
    return fig, ax

def plot_angle_distribution(df, save_path=None):
    """
    Wizualizuje rozkład kąta strzału przed i po transformacji logarytmicznej.
    """
    # Logarytmizacja
    df['log_angle'] = np.log1p(df['angle'])

    # Skośność
    angle_skew = stats.skew(df['angle'])
    log_angle_skew = stats.skew(df['log_angle'])

    # Histogramy
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df['angle'], kde=True, ax=axes[0])
    axes[0].set_title(f'Rozkład kąta strzału (skośność: {angle_skew:.3f})')

    sns.histplot(df['log_angle'], kde=True, ax=axes[1])
    axes[1].set_title(f'Rozkład zlogarytmizowanego kąta strzału (skośność: {log_angle_skew:.3f})')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano histogramy rozkładu kąta strzału do {save_path}")

    # Wydruk wartości skośności
    print("Skośność oryginalnych zmiennych:")
    print(f"Kąt strzału: {angle_skew:.3f}")
    print("\nSkośność zlogarytmizowanych zmiennych:")
    print(f"Log(kąt strzału): {log_angle_skew:.3f}")
    
    return fig, axes

def plot_roc_curve(y_test, y_pred_proba, roc_auc, ax=None, save_path=None):
    """
    Rysuje krzywą ROC.
    Jeśli podano `ax`, rysuje na danym wykresie, w przeciwnym razie tworzy nowy wykres.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC', size=18)
    ax.legend(loc="lower right")
    
    if save_path and ax.get_figure() is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano krzywą ROC do {save_path}")
    
    return fig, ax

def plot_expected_vs_actual_goals(total_xg, total_xg_beta, total_goals, xg_ratio, xg_beta_ratio, ax=None, save_path=None):
    """
    Rysuje wykres porównujący oczekiwane i rzeczywiste gole.
    Jeśli podano `ax`, rysuje na danym wykresie, w przeciwnym razie tworzy nowy wykres.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    
    bar_labels = ['Raw xG', 'Beta Cal.', 'Actual Goals']
    bar_values = [total_xg, total_xg_beta, total_goals]
    bar_colors = ['skyblue', 'orange', 'navy']
    
    ax.bar(bar_labels, bar_values, color=bar_colors)
    ax.axhline(y=total_goals, color='red', linestyle='--')
    ax.set_ylabel('Goals', size=14)
    ax.set_title('Expected vs Actual Goals', size=18)
    
    for i, v in enumerate(bar_values):
        ax.text(i, v + 5, f"{v:.1f}", ha='center', fontsize=14)
    
    y_pos = max(bar_values) * 0.1
    ax.text(0, total_xg - y_pos, f"Ratio: {xg_ratio:.2f}", ha='center', fontsize=14)
    ax.text(1, total_xg_beta - y_pos, f"Ratio: {xg_beta_ratio:.2f}", ha='center', fontsize=14)
    
    if save_path and ax.get_figure() is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano porównanie goli do {save_path}")
    
    return fig, ax

def plot_reliability_diagram(y_test, y_pred_proba, y_pred_proba_beta, ax=None, save_path=None):
    """
    Rysuje diagram wiarygodności.
    Jeśli podano `ax`, rysuje na danym wykresie, w przeciwnym razie tworzy nowy wykres.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    
    bins = 10
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def calc_reliability_data(y_true, y_pred):
        bin_indices = np.digitize(y_pred, bin_edges[1:-1])
        bin_sums = np.bincount(bin_indices, minlength=bins)
        bin_true = np.bincount(bin_indices, weights=y_true, minlength=bins)
        bin_probs = np.zeros(bins)
        for i in range(bins):
            if bin_sums[i] > 0:
                bin_probs[i] = bin_true[i] / bin_sums[i]
        return bin_probs, bin_sums
    
    bin_probs_raw, bin_sums_raw = calc_reliability_data(y_test, y_pred_proba)
    bin_probs_beta, bin_sums_beta = calc_reliability_data(y_test, y_pred_proba_beta)
    
    ax.plot(bin_centers, bin_probs_raw, 'o-', color='skyblue', label='Raw predictions')
    ax.plot(bin_centers, bin_probs_beta, 's-', color='orange', label='Beta calibration')
    ax.plot([0, 1], [0, 1], 'k--', color='navy', label='Perfect calibration')
    
    ax.set_xlabel('Predicted probability', size=14)
    ax.set_ylabel('Observed frequency', size=14)
    ax.set_title('Reliability Diagram', size=18)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if save_path and ax.get_figure() is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Zapisano diagram wiarygodności do {save_path}")
    
    return fig, ax