from typing import TYPE_CHECKING, Any

import pandas as pd

from arc.core.constants import (
    CLASS_YEAR_DRAFT_AGE,
    DEFAULT_DRAFT_AGE,
    DEFAULT_POSITION,
    HEIGHT_BASELINES,
    MAX_DRAFT_AGE,
    MIN_DRAFT_AGE,
    RATE_STAT_DEFAULTS,
    normalize_position,
)

if TYPE_CHECKING:
    from arc.db import CollegeStats, Player

RATE_STAT_COLUMNS: list[str] = list(RATE_STAT_DEFAULTS)
FEATURE_COLUMNS: list[str] = ["draft_age", "positional_size_delta", *RATE_STAT_COLUMNS]
SIMILARITY_COLUMNS: list[str] = ["positional_size_delta", *RATE_STAT_COLUMNS]
RECENCY_DECAY = 0.5


def resolve_draft_age(draft_year: int, birth_date: object) -> float | None:
    dob = pd.to_datetime(birth_date, errors="coerce")
    if pd.isna(dob):
        return None
    draft_date = pd.to_datetime(f"{draft_year}-06-25")
    age = float((draft_date - dob).days / 365.25)
    if not MIN_DRAFT_AGE <= age <= MAX_DRAFT_AGE:
        return None
    return round(age, 1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    class_year_age = data["class_year"].astype(str).str.strip().map(CLASS_YEAR_DRAFT_AGE)
    data["draft_age"] = (
        pd.to_numeric(data["draft_age"], errors="coerce").fillna(class_year_age).fillna(DEFAULT_DRAFT_AGE)
    )
    baseline_heights = data["primary_position"].map(normalize_position).fillna(DEFAULT_POSITION).map(HEIGHT_BASELINES)
    data["height_inches"] = pd.to_numeric(data["height_inches"], errors="coerce").fillna(baseline_heights)
    data["positional_size_delta"] = data["height_inches"] - baseline_heights
    for column, default in RATE_STAT_DEFAULTS.items():
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(default)
    return data


def blend_college_seasons(seasons: list[dict[str, Any]]) -> dict[str, Any]:
    def _val(item: dict[str, Any], key: str) -> Any:
        value = item.get(key)
        return None if pd.isna(value) else value

    def _minutes(item: dict[str, Any], default: float) -> float:
        value = _val(item, "min_pct")
        return default if value is None else float(value)

    sorted_seasons = sorted(seasons, key=lambda s: (s["season"], _minutes(s, 0.0)), reverse=True)
    latest = sorted_seasons[0]
    weights = [(RECENCY_DECAY**i) * max(_minutes(s, 50.0), 10.0) / 100.0 for i, s in enumerate(sorted_seasons)]
    blended: dict[str, Any] = {}
    for col in RATE_STAT_COLUMNS:
        pairs = [
            (float(value), weight)
            for season, weight in zip(sorted_seasons, weights, strict=True)
            if (value := _val(season, col)) is not None
        ]
        total_w = sum(w for _, w in pairs)
        blended[col] = round(sum(v * w for v, w in pairs) / total_w, 4) if pairs else None
    blended["class_year"] = _val(latest, "class_year")
    blended["draft_age"] = _val(latest, "draft_age")
    blended["last_college_season"] = _val(latest, "season")
    return blended


def _season_row(season: "CollegeStats") -> dict[str, Any]:
    return {
        "season": season.season,
        "class_year": season.class_year,
        "draft_age": season.draft_age,
        **{column: getattr(season, column) for column in RATE_STAT_COLUMNS},
    }


def player_feature_row(player: "Player") -> dict[str, Any]:
    blended = blend_college_seasons([_season_row(season) for season in player.college_seasons])
    blended["player_id"] = str(player.id)
    blended["primary_position"] = player.primary_position
    blended["height_inches"] = player.height_inches
    reference_year = player.draft_pick.draft_year if player.draft_pick else blended["last_college_season"]
    draft_age = resolve_draft_age(reference_year, player.birth_date)
    if draft_age is not None:
        blended["draft_age"] = draft_age
    return blended
