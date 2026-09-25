import logging
import re

import pandas as pd

from arc.core.names import clean_name
from arc.ml.feature_engineering import RATE_STAT_COLUMNS, blend_college_seasons, resolve_draft_age
from arc.ml.tiers import TIER_NAMES
from data_pipeline.config import (
    COLLEGE_SEASON_YEARS,
    DRAFT_YEARS,
    HOLDOUT_MAX_YEAR,
    IDENTITY_EXCLUSIONS_PATH,
    LABELLED_PATH,
    LATEST_NBA_SEASON_END,
    NBA_SEASON_YEARS,
    SERVING_PLAYERS_PATH,
    SERVING_SEASON_COLUMNS,
    SERVING_SEASONS_PATH,
    TORVIK_RAW_DIR,
    assign_split,
)
from data_pipeline.identity import (
    bio_identity_mismatch,
    careers_by_player_page,
    identity_seasons,
    resolve_college_identity,
    resolve_player_identity,
    stable_player_id,
)
from data_pipeline.labels import compute_peak_stats, empty_peak_stats, label_outcomes
from data_pipeline.sources.basketball_reference import fetch_bio, fetch_drafts, fetch_seasons
from data_pipeline.sources.torvik import parse_torvik_season

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_labelled")


MIN_TORVIK_SEASON_ROWS = 1_000

_SUFFIX_COMMA = re.compile(r",\s*(Jr\.|Sr\.|II|III|IV)\s*$")

SERVING_PLAYER_COLUMNS = [
    "player_id",
    "player_name",
    "clean_name",
    "birth_date",
    "height_inches",
    "primary_position",
    "draft_year",
    "overall_pick",
    "drafting_team",
]


LABELLED_COLUMNS = [
    "player_id",
    "player_name",
    "college_team",
    "class_year",
    "draft_age",
    "height_inches",
    "primary_position",
    *RATE_STAT_COLUMNS,
    "split",
    "overall_pick",
    "cohort_year",
    "composite_z",
    "career_tier",
    "tier_label",
    "peak_bpm",
    "peak_vorp",
    "peak_mpg",
    "career_win_shares",
    "seasons_played",
    "games_played",
    "is_active",
    "peak_window_seasons",
]

EXCLUSION_COLUMNS = [
    "player_name",
    "reason",
    "draft_year",
    "candidate_birth_dates",
    "candidate_teams",
]


def tidy_name(name: str) -> str:
    return _SUFFIX_COMMA.sub(r" \1", str(name).strip())


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_college_index() -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for year in COLLEGE_SEASON_YEARS:
        path = TORVIK_RAW_DIR / f"torvik_{year}.csv"
        df = parse_torvik_season(year, path)
        if len(df) < MIN_TORVIK_SEASON_ROWS:
            raise RuntimeError(
                f"Torvik season {year} produced only {len(df)} rows from {path}; "
                "refusing to label against a partial season."
            )
        for record in df.to_dict("records"):
            key = clean_name(str(record["player_name"]))
            if not key:
                continue
            index.setdefault(key, []).append(
                {
                    "season": int(record["season"]),
                    "row": record,
                    "college_team": _text(record.get("college_team")),
                    "birth_date": _as_timestamp(record.get("birth_date")),
                }
            )
    return index


def load_draft_entries() -> dict[tuple[str, int], dict]:
    drafts = fetch_drafts(min(DRAFT_YEARS), max(DRAFT_YEARS))
    entries: dict[tuple[str, int], dict] = {}
    for row in drafts.to_dict("records"):
        name_key = clean_name(str(row["player_name"]))
        if not name_key:
            continue
        draft_year = int(row["draft_year"])
        entries[(name_key, draft_year)] = {
            "player_name": str(row["player_name"]).strip(),
            "draft_year": draft_year,
            "overall_pick": row.get("overall_pick"),
            "team": row.get("nba_team"),
            "college": row.get("college_name"),
        }
    return entries


def resolve_cohort_year(draft_entry: dict | None, final_college_season: int) -> int:
    if draft_entry is not None:
        return int(draft_entry["draft_year"])
    return int(final_college_season)


def _append_labelled_row(
    labelled_rows: list[dict],
    *,
    candidate: dict,
    candidate_window: list[dict],
    draft_entry: dict | None,
    peaks: dict,
    split: str,
    cohort_year: int,
    draft_age: float | None,
    player_id: str,
) -> None:
    row = candidate["row"]
    blended = blend_college_seasons([c["row"] for c in candidate_window])
    labelled_rows.append(
        {
            "player_id": player_id,
            "player_name": tidy_name(str(row["player_name"])),
            "college_team": _text(row.get("college_team")),
            "class_year": _text(row.get("class_year")),
            "draft_age": draft_age if draft_age is not None else row["draft_age"],
            "height_inches": None if pd.isna(row.get("height_inches")) else row["height_inches"],
            "primary_position": None if pd.isna(row.get("primary_position")) else row["primary_position"],
            **{column: blended.get(column) for column in RATE_STAT_COLUMNS},
            "split": split,
            "overall_pick": draft_entry["overall_pick"] if draft_entry else None,
            "cohort_year": cohort_year,
            "composite_z": None,
            "career_tier": 5,
            "tier_label": TIER_NAMES[5],
            "peak_bpm": peaks["peak_bpm"],
            "peak_vorp": peaks["peak_vorp"],
            "peak_mpg": peaks["peak_mpg"],
            "career_win_shares": peaks["career_win_shares"],
            "seasons_played": peaks["seasons_played"],
            "games_played": peaks["games_played"],
            "is_active": peaks["last_season_end"] == LATEST_NBA_SEASON_END,
            "peak_window_seasons": peaks["peak_window_seasons"],
        }
    )


def _record_serving_player(
    serving_players: dict[str, dict],
    serving_seasons: dict[tuple[str, int], dict],
    *,
    player_id: str,
    name_key: str,
    candidate: dict,
    candidate_window: list[dict],
    draft_entry: dict | None,
) -> None:
    if player_id in serving_players:
        logger.debug(f"Serving identity for {name_key!r} already recorded.")
        return
    row = candidate["row"]
    birth_date = candidate["birth_date"]
    serving_players[player_id] = {
        "player_id": player_id,
        "player_name": tidy_name(str(row["player_name"])),
        "clean_name": name_key,
        "birth_date": None if pd.isna(birth_date) else pd.Timestamp(birth_date).date().isoformat(),
        "height_inches": None if pd.isna(row.get("height_inches")) else int(row["height_inches"]),
        "primary_position": None if pd.isna(row.get("primary_position")) else str(row["primary_position"]),
        "draft_year": int(draft_entry["draft_year"]) if draft_entry else None,
        "overall_pick": int(draft_entry["overall_pick"]) if draft_entry else None,
        "drafting_team": str(draft_entry["team"]) if draft_entry else None,
    }
    for season_candidate in candidate_window:
        season_row = season_candidate["row"]
        key = (player_id, int(season_candidate["season"]))
        if key in serving_seasons:
            continue
        record = {
            column: (None if pd.isna(value := season_row.get(column)) else value) for column in SERVING_SEASON_COLUMNS
        }
        record["player_id"] = player_id
        record["season"] = int(season_candidate["season"])
        serving_seasons[key] = record


def _exclusion(*, player_name: str, reason: str, draft_year: int, candidates: list[dict]) -> dict:
    return {
        "player_name": player_name,
        "reason": reason,
        "draft_year": draft_year,
        "candidate_birth_dates": "|".join(sorted({str(c["birth_date"]) for c in candidates})),
        "candidate_teams": "|".join(sorted({c["college_team"] for c in candidates if c["college_team"]})),
    }


def _record_identity(
    *,
    name_key: str,
    display_name: str,
    candidate: dict,
    candidate_window: list[dict],
    draft_entry: dict | None,
    cohort_year: int,
    peaks: dict,
    serving_players: dict[str, dict],
    serving_seasons: dict[tuple[str, int], dict],
    labelled_rows: list[dict],
    labelled_ids: set[str],
) -> None:
    split = assign_split(cohort_year)
    player_id = stable_player_id(name_key, candidate["birth_date"], cohort_year, candidate["college_team"])
    _record_serving_player(
        serving_players,
        serving_seasons,
        player_id=player_id,
        name_key=name_key,
        candidate=candidate,
        candidate_window=candidate_window,
        draft_entry=draft_entry,
    )
    if split == "live":
        return

    if player_id in labelled_ids:
        logger.warning(f"Duplicate resolved identity for {display_name!r}; keeping the first row.")
        return
    _append_labelled_row(
        labelled_rows,
        candidate=candidate,
        candidate_window=candidate_window,
        draft_entry=draft_entry,
        peaks=peaks,
        split=split,
        cohort_year=cohort_year,
        draft_age=resolve_draft_age(cohort_year, candidate["birth_date"]),
        player_id=player_id,
    )
    labelled_ids.add(player_id)


def main() -> None:
    LABELLED_PATH.parent.mkdir(parents=True, exist_ok=True)
    br_seasons = fetch_seasons(min(NBA_SEASON_YEARS), max(NBA_SEASON_YEARS))
    draft_entries = load_draft_entries()
    draft_entries_by_name: dict[str, list[dict]] = {}
    for (name_key, _draft_year), entry in draft_entries.items():
        draft_entries_by_name.setdefault(name_key, []).append(entry)
    college_index = load_college_index()
    nba_seasons_by_name: dict[str, list[dict]] = {}
    for season in br_seasons.to_dict("records"):
        name_key = clean_name(str(season["player"]))
        nba_seasons_by_name.setdefault(name_key, []).append(
            {
                "player_name": str(season["player"]),
                "season_end": int(season["season_end"]),
                "age": season.get("age"),
                "g": season["g"],
                "mp": season["mp"],
                "ws": season["ws"],
                "bpm": season["bpm"],
                "vorp": season["vorp"],
                "href": season.get("player_href"),
            }
        )
    logger.info(f"NBA name groups: {len(nba_seasons_by_name)} | college names indexed: {len(college_index)}")
    labelled_rows: list[dict] = []
    exclusions: list[dict] = []
    serving_players: dict[str, dict] = {}
    serving_seasons: dict[tuple[str, int], dict] = {}
    unmatched = 0
    attempted_draft_keys: set[tuple[str, int]] = set()
    labelled_ids: set[str] = set()
    birth_dates_enriched = 0
    for name_key, seasons in nba_seasons_by_name.items():
        display_name = seasons[0]["player_name"]
        for career in careers_by_player_page(seasons):
            draft_entry, candidate, reason = resolve_player_identity(
                name_key, career, draft_entries_by_name, college_index
            )
            if draft_entry is not None:
                attempted_draft_keys.add((name_key, int(draft_entry["draft_year"])))
            bio = None
            if candidate is not None and (pd.isna(candidate["birth_date"]) or draft_entry is None):
                bio = fetch_bio(career[0]["href"])
            if (
                candidate is not None
                and draft_entry is None
                and bio is not None
                and bio_identity_mismatch(candidate["college_team"], bio.colleges)
            ):
                candidate = None
                reason = "college_mismatch"
            if candidate is None:
                assert reason is not None
                if draft_entry is not None:
                    exclusions.append(
                        _exclusion(
                            player_name=display_name,
                            reason=reason,
                            draft_year=draft_entry["draft_year"],
                            candidates=college_index.get(name_key, []),
                        )
                    )
                else:
                    unmatched += 1
                continue
            if bio is not None and bio.birth_date is not None and pd.isna(candidate["birth_date"]):
                candidate["birth_date"] = pd.Timestamp(bio.birth_date)
                birth_dates_enriched += 1
            cohort_year = resolve_cohort_year(draft_entry, int(candidate["season"]))
            _record_identity(
                name_key=name_key,
                display_name=display_name,
                candidate=candidate,
                candidate_window=identity_seasons(college_index, name_key, cohort_year, candidate),
                draft_entry=draft_entry,
                cohort_year=cohort_year,
                peaks=compute_peak_stats(career),
                serving_players=serving_players,
                serving_seasons=serving_seasons,
                labelled_rows=labelled_rows,
                labelled_ids=labelled_ids,
            )
    for (name_key, draft_year), draft_entry in draft_entries.items():
        if (name_key, draft_year) in attempted_draft_keys:
            continue
        candidate, reason = resolve_college_identity(
            college_index.get(name_key, []),
            anchor_year=draft_year,
            college_hint=draft_entry.get("college"),
        )
        if candidate is None:
            assert reason is not None
            if draft_year > HOLDOUT_MAX_YEAR:
                logger.info(f"Draft {draft_year} pick {draft_entry['player_name']!r} has no college match; not served.")
                continue
            exclusions.append(
                _exclusion(
                    player_name=draft_entry["player_name"],
                    reason=reason,
                    draft_year=draft_year,
                    candidates=college_index.get(name_key, []),
                )
            )
            continue
        _record_identity(
            name_key=name_key,
            display_name=draft_entry["player_name"],
            candidate=candidate,
            candidate_window=identity_seasons(college_index, name_key, draft_year, candidate),
            draft_entry=draft_entry,
            cohort_year=draft_year,
            peaks=empty_peak_stats(games_played=0),
            serving_players=serving_players,
            serving_seasons=serving_seasons,
            labelled_rows=labelled_rows,
            labelled_ids=labelled_ids,
        )
    label_outcomes(labelled_rows)
    labelled_df = pd.DataFrame(labelled_rows, columns=LABELLED_COLUMNS)
    exclusions_df = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS)
    serving_players_df = pd.DataFrame(serving_players.values(), columns=SERVING_PLAYER_COLUMNS).sort_values(
        "player_id", ignore_index=True
    )
    serving_seasons_df = pd.DataFrame(
        serving_seasons.values(), columns=["player_id", *SERVING_SEASON_COLUMNS]
    ).sort_values(["player_id", "season"], ignore_index=True)
    labelled_df.to_csv(LABELLED_PATH, index=False)
    exclusions_df.to_csv(IDENTITY_EXCLUSIONS_PATH, index=False)
    serving_players_df.to_csv(SERVING_PLAYERS_PATH, index=False)
    serving_seasons_df.to_csv(SERVING_SEASONS_PATH, index=False)
    logger.info(f"Labelled players: {len(labelled_df)}")
    logger.info(f"Split sizes: {labelled_df.groupby('split').size().to_dict() if not labelled_df.empty else {}}")
    logger.info(f"Identity exclusions: {len(exclusions_df)}")
    logger.info(f"Birth dates enriched from Basketball Reference: {birth_dates_enriched}")
    logger.info(f"Serving population: {len(serving_players_df)} players, {len(serving_seasons_df)} player-seasons.")
    logger.info(f"Unmatched NBA players (no college match): {unmatched}")


if __name__ == "__main__":
    main()
