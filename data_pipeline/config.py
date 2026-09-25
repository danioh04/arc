from arc.core.config import ROOT

FIRST_DRAFT_YEAR = 2008
FIRST_COLLEGE_SEASON = 2008
LATEST_COLLEGE_SEASON = 2026
FIRST_NBA_SEASON_END = 2009
LATEST_NBA_SEASON_END = 2026
LATEST_DRAFT_YEAR = 2026

COLLEGE_SEASON_YEARS = range(FIRST_COLLEGE_SEASON, LATEST_COLLEGE_SEASON + 1)
DRAFT_YEARS = range(FIRST_DRAFT_YEAR, LATEST_DRAFT_YEAR + 1)
NBA_SEASON_YEARS = range(FIRST_NBA_SEASON_END, LATEST_NBA_SEASON_END + 1)

MAX_COLLEGE_GAP = 5

TRAIN_MAX_YEAR = 2019
HOLDOUT_MAX_YEAR = 2022

MODEL_VERSION = "v1.0-xgboost-regressor"

GENERATED_DIR = ROOT / "data" / "generated"
TORVIK_RAW_DIR = ROOT / "data" / "raw" / "torvik"

LABELLED_PATH = GENERATED_DIR / "labelled.csv"
SERVING_PLAYERS_PATH = GENERATED_DIR / "serving_players.csv"
SERVING_SEASONS_PATH = GENERATED_DIR / "serving_seasons.csv"
IDENTITY_EXCLUSIONS_PATH = GENERATED_DIR / "identity_exclusions.csv"
LABEL_SPEC_PATH = GENERATED_DIR / "label_spec.json"

SERVING_SEASON_COLUMNS = [
    "season",
    "college_team",
    "conference",
    "class_year",
    "draft_age",
    "min_pct",
    "usg_pct",
    "ts_pct",
    "orb_pct",
    "drb_pct",
    "ast_pct",
    "to_pct",
    "blk_pct",
    "stl_pct",
    "ftr",
    "fg3_pct",
    "fg3a_per_fga",
    "ft_pct",
    "bpm",
    "obpm",
    "dbpm",
]


def assign_split(draft_year: int) -> str:
    if draft_year <= TRAIN_MAX_YEAR:
        return "train"
    if draft_year <= HOLDOUT_MAX_YEAR:
        return "holdout"
    return "live"
