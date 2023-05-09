"""
Constants for analysis
"""

from pathlib import Path

HOME: Path = Path(__file__).resolve().parents[1]

DATA_DIR = HOME / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"

RAW_DATA_FILE = "all_matches.csv"
INTERMEDIATE_DATA_MATCHES = "all_matches_chunk_preprocessed.csv"
INTERMEDIATE_DATA_RATES = "rates.csv"

START_YEAR = 2016
NOT_USED_COLUMNS = [
    "start_date",
    "end_date",
    "location",
    "prize_money",
    "currency",
    "player_id",
    "opponent_id",
    "serve_rating",
    "service_games_won",
    "return_rating",
    "return_games_played",
    "duration",
    "seed",
    "nation",
]
CHUNK_SIZE = 100
ANALYZE_TARGET_COLS = [
    "first_serve_rate",
    "first_serve_point_rate",
    "second_serve_rate",
    "second_serve_point_rate",
    "first_serve_return_point_rate",
    "second_serve_return_point_rate",
]
