import logging
from pathlib import Path
from typing import Any

import pandas as pd

from arc.core.constants import CLASS_YEAR_DRAFT_AGE, DEFAULT_DRAFT_AGE, MAX_DRAFT_AGE, MIN_DRAFT_AGE

logger = logging.getLogger(__name__)

RATE_STAT_BOUNDS = {
    "min_pct": (0.0, 100.0),
    "usg_pct": (0.0, 100.0),
    "ts_pct": (0.0, 3.0),
    "fg3_pct": (0.0, 1.0),
    "ft_pct": (0.0, 1.0),
    "fg3a_per_fga": (0.0, 1.0),
    "orb_pct": (0.0, 200.0),
    "drb_pct": (0.0, 200.0),
    "ast_pct": (0.0, 200.0),
    "to_pct": (0.0, 100.0),
    "blk_pct": (0.0, 200.0),
    "stl_pct": (0.0, 200.0),
    "ftr": (0.0, 400.0),
    "bpm": (-60.0, 60.0),
    "obpm": (-60.0, 60.0),
    "dbpm": (-60.0, 60.0),
}

TORVIK_ROLE_TO_POSITION = {
    "Pure PG": "G",
    "Scoring PG": "G",
    "Combo G": "G",
    "Wing G": "G",
    "Wing F": "W",
    "Stretch 4": "F",
    "PF/C": "F",
    "C": "C",
}


def clamp_rate_stats(df: pd.DataFrame, season: int) -> pd.DataFrame:
    clipped = df.copy()
    counts: dict[str, int] = {}
    for column, (low, high) in RATE_STAT_BOUNDS.items():
        if column not in clipped.columns:
            continue
        series = pd.to_numeric(clipped[column], errors="coerce")
        n_outside = int(((series < low) | (series > high)).sum())
        if n_outside:
            counts[column] = n_outside
        clipped[column] = series.clip(lower=low, upper=high)
    if counts:
        detail = ", ".join(f"{column}={n}" for column, n in sorted(counts.items()))
        logger.info(f"Season {season}: clamped {sum(counts.values())} out-of-range rate stats ({detail}).")
    return clipped


def _parse_height_inches(value: Any) -> float:
    try:
        parts = str(value).replace("'", "-").split("-")
        inches = int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        return float("nan")
    return float(inches) if 60 <= inches <= 90 else float("nan")


def parse_torvik_season(year: int, file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None, low_memory=False)
    parsed = pd.DataFrame()
    parsed["player_name"] = df[0].astype(str)
    parsed["college_team"] = df[1].astype(str)
    parsed["conference"] = df[2].astype(str)
    parsed["min_pct"] = df[4]
    parsed["usg_pct"] = df[6]
    parsed["ts_pct"] = df[8] / 100.0
    parsed["orb_pct"] = df[9]
    parsed["drb_pct"] = df[10]
    parsed["ast_pct"] = df[11]
    parsed["to_pct"] = df[12]
    parsed["ft_pct"] = df[15]
    parsed["fg3_pct"] = df[21]
    parsed["blk_pct"] = df[22]
    parsed["stl_pct"] = df[23]
    parsed["ftr"] = df[24]
    class_values = set(CLASS_YEAR_DRAFT_AGE)
    raw_class = df[25].astype("string").str.strip()
    raw_height = df[26].astype("string").str.strip()
    class_from_25 = raw_class.isin(class_values)
    class_from_26 = raw_height.isin(class_values)
    parsed["class_year"] = raw_class.where(class_from_25, None)
    parsed.loc[~class_from_25 & class_from_26, "class_year"] = raw_height[~class_from_25 & class_from_26]
    height_source = df[26].where(~class_from_26, df[25])
    parsed["height_inches"] = height_source.apply(_parse_height_inches)
    parsed["bpm"] = df[50]
    parsed["obpm"] = df[51]
    parsed["dbpm"] = df[52]
    fga = df[17] + df[20]
    parsed["fg3a_per_fga"] = (df[20] / fga).round(3)
    parsed["season"] = year
    draft_date = pd.to_datetime(f"{year}-06-25")
    dob = pd.to_datetime(df[66], errors="coerce")
    dob = dob.mask((dob.dt.month == 10) & (dob.dt.day == 15))
    parsed["birth_date"] = dob
    exact_age = ((draft_date - dob).dt.days / 365.25).round(1)
    invalid = exact_age.isna() | (exact_age < MIN_DRAFT_AGE) | (exact_age > MAX_DRAFT_AGE)
    parsed["draft_age"] = exact_age.where(
        ~invalid, parsed["class_year"].map(CLASS_YEAR_DRAFT_AGE).fillna(DEFAULT_DRAFT_AGE)
    )
    parsed["primary_position"] = df[64].astype(str).map(TORVIK_ROLE_TO_POSITION)
    heights = parsed["height_inches"]
    position_from_height = pd.Series(
        pd.cut(heights, bins=[-float("inf"), 76, 79, 82, float("inf")], labels=["G", "W", "F", "C"]).astype("object"),
        index=parsed.index,
    ).where(heights.notna(), None)
    parsed["primary_position"] = parsed["primary_position"].fillna(position_from_height)
    guard_mask = (parsed["height_inches"] <= 76) & (parsed["primary_position"].isin(["W", "F"]))
    parsed.loc[guard_mask, "primary_position"] = "G"
    parsed = clamp_rate_stats(parsed, year)
    parsed = parsed.dropna(subset=["bpm"])
    logger.info(f"Parsed {len(parsed)} player records for season {year}")
    return parsed
