import os
import warnings
import pandas as pd
import statsbombpy.sb as sb

warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

DEFAULT_SEASON = '2015/2016'
TOP5_LEAGUES = ['Italy', 'England', 'Spain', 'Germany', 'France']
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _count_passes_in_sequence(events):
    """
    For each shot in `events`, counts the number of passes in the same
    possession sequence (same match_id + possession number).

    Returns a Series indexed like the shots DataFrame.
    """
    shots_mask = events['type'] == 'Shot'
    passes_mask = events['type'] == 'Pass'

    pass_counts = (
        events[passes_mask]
        .groupby(['match_id', 'possession'])
        .size()
        .rename('n_passes_in_sequence')
    )

    shots = events[shots_mask].copy()
    shots = shots.join(pass_counts, on=['match_id', 'possession'])
    shots['n_passes_in_sequence'] = shots['n_passes_in_sequence'].fillna(0).astype(int)
    return shots


def collect_shots_data(
    league=TOP5_LEAGUES,
    save_path=DATA_DIR,
    season_name=DEFAULT_SEASON,
    save_files=True,
    include_passes_in_sequence=True,
):
    """
    Retrieves shot data from StatsBomb open data for the given league(s) and season.

    Parameters
    ----------
    league : str or list
        League name(s) to collect. Default: all TOP5 leagues.
    save_path : str
        Directory to save CSV files.
    season_name : str
        Season string, e.g. '2015/2016'.
    save_files : bool
        Whether to write CSV files to disk.
    include_passes_in_sequence : bool
        If True, adds `n_passes_in_sequence` column (passes in the same
        possession leading up to each shot). Requires fetching all events
        per match (slower). Default: True.

    Returns
    -------
    pd.DataFrame or None
    """
    if save_files:
        os.makedirs(save_path, exist_ok=True)

    leagues_to_process = [league] if isinstance(league, str) else list(league)
    all_shots_data = []

    free_comps = sb.competitions()

    for current_league in leagues_to_process:
        print(f"\n--- {current_league} ---")

        league_data = free_comps[
            (free_comps['season_name'] == season_name) &
            (free_comps['country_name'] == current_league)
        ]

        if league_data.empty:
            print(f"No data found for {current_league} / {season_name}. Skipping.")
            continue

        competitions = list(league_data['competition_id'])
        season_id = league_data['season_id'].iloc[0]

        all_matches = pd.concat([
            sb.matches(competition_id=comp_id, season_id=season_id)
            for comp_id in competitions
        ])
        match_ids = list(all_matches['match_id'])
        print(f"Matches to process: {len(match_ids)}")

        shot_data = []

        for idx, match_id in enumerate(match_ids):
            print(f"  {idx + 1}/{len(match_ids)}", end='\r')
            try:
                events = sb.events(match_id=match_id)
                events['match_id'] = match_id

                if include_passes_in_sequence:
                    shots = _count_passes_in_sequence(events)
                else:
                    shots = events[events['type'] == 'Shot'].copy()
                    shots['match_id'] = match_id

                if not shots.empty:
                    shot_data.append(shots)

            except Exception as e:
                print(f"\n  Error on match {match_id}: {e}")

        if not shot_data:
            print(f"No shots found for {current_league}.")
            continue

        league_shots = pd.concat(shot_data, ignore_index=True)
        print(f"\nShots collected: {len(league_shots)}")

        if save_files:
            season_str = season_name.replace("/", "_")
            path = os.path.join(save_path, f'shots_{current_league}_{season_str}.csv')
            league_shots.to_csv(path, index=False)
            print(f"Saved: {path}")

        all_shots_data.append(league_shots)

    if not all_shots_data:
        print("No data retrieved.")
        return None

    all_shots = pd.concat(all_shots_data, ignore_index=True)

    if save_files and len(leagues_to_process) > 1:
        season_str = season_name.replace("/", "_")
        combined_path = os.path.join(save_path, f'shots_combined_{season_str}.csv')
        all_shots.to_csv(combined_path, index=False)
        print(f"\nCombined file saved: {combined_path}")

    return all_shots


FIFA_WORLD_CUP_2022 = {'competition_id': 43, 'season_id': 106}


def collect_fifa_world_cup_2022(
    save_path=DATA_DIR,
    include_passes_in_sequence=True,
):
    """
    Retrieves shot data from StatsBomb open data for FIFA World Cup 2022.

    Returns
    -------
    pd.DataFrame or None
    """
    os.makedirs(save_path, exist_ok=True)

    print("\n--- FIFA World Cup 2022 ---")

    all_matches = sb.matches(
        competition_id=FIFA_WORLD_CUP_2022['competition_id'],
        season_id=FIFA_WORLD_CUP_2022['season_id'],
    )
    match_ids = list(all_matches['match_id'])
    print(f"Matches to process: {len(match_ids)}")

    shot_data = []

    for idx, match_id in enumerate(match_ids):
        print(f"  {idx + 1}/{len(match_ids)}", end='\r')
        try:
            events = sb.events(match_id=match_id)
            events['match_id'] = match_id

            if include_passes_in_sequence:
                shots = _count_passes_in_sequence(events)
            else:
                shots = events[events['type'] == 'Shot'].copy()
                shots['match_id'] = match_id

            if not shots.empty:
                shot_data.append(shots)

        except Exception as e:
            print(f"\n  Error on match {match_id}: {e}")

    if not shot_data:
        print("No shots found.")
        return None

    all_shots = pd.concat(shot_data, ignore_index=True)
    print(f"\nShots collected: {len(all_shots)}")

    path = os.path.join(save_path, 'shots_fifa_world_cup_2022.csv')
    all_shots.to_csv(path, index=False)
    print(f"Saved: {path}")

    return all_shots


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect StatsBomb shot data.')
    parser.add_argument(
        '--leagues', nargs='+', default=TOP5_LEAGUES,
        help='Leagues to collect (default: all TOP5)'
    )
    parser.add_argument(
        '--season', default=DEFAULT_SEASON,
        help='Season name (default: 2015/2016)'
    )
    parser.add_argument(
        '--no-passes-in-sequence', action='store_true',
        help='Skip n_passes_in_sequence calculation (faster)'
    )
    parser.add_argument(
        '--output-dir', default=DATA_DIR,
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--fifa-2022', action='store_true',
        help='Also collect FIFA World Cup 2022 data (saved as shots_fifa_world_cup_2022.csv)'
    )
    parser.add_argument(
        '--skip-club', action='store_true',
        help='Skip club league collection (useful when only --fifa-2022 is needed)'
    )
    args = parser.parse_args()

    if not args.skip_club:
        collect_shots_data(
            league=args.leagues,
            season_name=args.season,
            save_path=args.output_dir,
            include_passes_in_sequence=not args.no_passes_in_sequence,
        )

    if args.fifa_2022:
        collect_fifa_world_cup_2022(
            save_path=args.output_dir,
            include_passes_in_sequence=not args.no_passes_in_sequence,
        )
