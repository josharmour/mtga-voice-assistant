import pandas as pd

def load_replay_data(filepath: str) -> pd.DataFrame:
    """
    Loads 17Lands replay data from a gzipped CSV file.

    Args:
        filepath: The path to the gzipped CSV file.

    Returns:
        A pandas DataFrame containing the replay data.
    """
    return pd.read_csv(filepath, compression='gzip', low_memory=False)

def parse_decklist(decklist: str) -> list[str]:
    """
    Parses a pipe-separated decklist string into a list of card identifiers.

    Args:
        decklist: A string containing card identifiers separated by '|'.

    Returns:
        A list of card identifiers.
    """
    return decklist.split('|')

def calculate_archetype_win_rate(df: pd.DataFrame, archetype: str) -> float:
    """
    Calculates the win rate for a specific deck archetype.

    Args:
        df: The DataFrame containing the replay data.
        archetype: The deck archetype to analyze (e.g., "WU", "BR").

    Returns:
        The win rate of the archetype as a float between 0 and 1.
    """
    archetype_games = df[df['user_deck_archetype'] == archetype]
    if len(archetype_games) == 0:
        return 0.0

    wins = len(archetype_games[archetype_games['game_result'] == 'win'])
    return wins / len(archetype_games)

def average_turns_for_archetype_wins(df: pd.DataFrame, archetype: str) -> float:
    """
    Calculates the average number of turns for games won by a specific archetype.

    Args:
        df: The DataFrame containing the replay data.
        archetype: The deck archetype to analyze.

    Returns:
        The average number of turns for games won by the archetype.
    """
    archetype_wins = df[(df['user_deck_archetype'] == archetype) & (df['game_result'] == 'win')]
    if len(archetype_wins) == 0:
        return 0.0

    return archetype_wins['game_n_turns'].mean()

if __name__ == '__main__':
    # This is an example of how to use the functions.
    # You will need to download the data file first.
    # wget https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.EOE.PremierDraft.csv.gz

    try:
        df = load_replay_data('replay_data_public.EOE.PremierDraft.csv.gz')

        # Example 1: Calculate the win rate for the "WU" (Blue/White) archetype
        wu_win_rate = calculate_archetype_win_rate(df, 'WU')
        print(f"Win rate for WU archetype: {wu_win_rate:.2%}")

        # Example 2: Calculate the average number of turns for games won by the "BR" (Black/Red) archetype
        br_avg_turns = average_turns_for_archetype_wins(df, 'BR')
        print(f"Average turns for BR archetype wins: {br_avg_turns:.2f}")

    except FileNotFoundError:
        print("The replay data file was not found.")
        print("Please download it by running the following command:")
        print("wget https://17lands-public.s3.amazonaws.com/analysis_data/replay_data/replay_data_public.EOE.PremierDraft.csv.gz")
