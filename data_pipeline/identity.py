import re
import uuid

import pandas as pd

from data_pipeline.config import MAX_COLLEGE_GAP

_SCHOOL_ALIASES = {
    "unc": "north carolina",
    "uconn": "connecticut",
    "pitt": "pittsburgh",
    "nc state": "north carolina state",
    "n c state": "north carolina state",
    "cal": "california",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "byu": "brigham young",
    "vcu": "virginia commonwealth",
    "ucf": "central florida",
    "unlv": "nevada las vegas",
    "ole miss": "mississippi",
    "app state": "appalachian state",
    "ga tech": "georgia tech",
    "va tech": "virginia tech",
}

AGE_TOLERANCE = 1.5


def _candidate_age_at(candidate: dict, on_date: str) -> float | None:
    birth_date = candidate.get("birth_date")
    if pd.isna(birth_date):
        return None
    return round(float((pd.Timestamp(on_date) - pd.Timestamp(birth_date)).days / 365.25), 1)


def _latest_candidate(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda c: c["season"])


def _same_identity(first: dict, second: dict) -> bool:
    first_dob = first.get("birth_date")
    second_dob = second.get("birth_date")
    if pd.notna(first_dob) and pd.notna(second_dob):
        return first_dob == second_dob
    return first["college_team"].lower() == second["college_team"].lower()


def _normalize_school(value: object) -> str:
    text = ("" if value is None or pd.isna(value) else str(value)).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\.", "", text)
    text = re.sub(r"\bst\b", "state", text)
    text = re.sub(r"\buniv\b", "university", text)
    text = re.sub(r"^university of\s+", "", text)
    text = re.sub(r"\s+university$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _SCHOOL_ALIASES.get(text, text)


def identity_seasons(college_index: dict[str, list[dict]], name_key: str, cohort: int, chosen: dict) -> list[dict]:
    window = [
        c
        for c in college_index.get(name_key, [])
        if cohort - MAX_COLLEGE_GAP <= c["season"] <= cohort and _same_identity(c, chosen)
    ]
    return window or [chosen]


def _same_season_conflict(candidates: list[dict]) -> bool:
    by_season: dict[int, set[str]] = {}
    for candidate in candidates:
        by_season.setdefault(candidate["season"], set()).add(candidate["college_team"].lower())
    return any(len(teams) > 1 for teams in by_season.values())


def _age_matches(candidate: dict, anchor_year: int, target_age: float) -> bool:
    age = _candidate_age_at(candidate, f"{anchor_year}-06-25")
    return age is not None and abs(age - target_age) <= AGE_TOLERANCE


def _school_prefix_match(first: str, second: str) -> bool:
    if not first or not second:
        return False
    return first.startswith(second) or second.startswith(first)


def resolve_college_identity(
    candidates: list[dict],
    anchor_year: int,
    college_hint: str | None = None,
    target_age: float | None = None,
) -> tuple[dict | None, str | None]:
    window = [c for c in candidates if anchor_year - MAX_COLLEGE_GAP <= c["season"] <= anchor_year]
    schools = {c["college_team"].lower() for c in candidates if c["college_team"]}
    if not window and target_age is not None and len(schools) == 1:
        window = [c for c in candidates if c["season"] <= anchor_year]
    if not window:
        return None, "no_college_match"
    hint = _normalize_school(college_hint)
    if hint:
        exact = [c for c in window if _normalize_school(c["college_team"]) == hint]
        if exact:
            window = exact
    groups: dict[pd.Timestamp | None, list[dict]] = {}
    for candidate in window:
        dob = candidate["birth_date"]
        key = dob if pd.notna(dob) else None
        groups.setdefault(key, []).append(candidate)
    if target_age is not None:
        matching_age: dict[pd.Timestamp | None, list[dict]] = {}
        for key, rows in groups.items():
            if key is None:
                continue
            if _age_matches(rows[0], anchor_year, target_age):
                matching_age[key] = rows
        if matching_age:
            groups = matching_age
    if len(groups) != 1:
        return None, "ambiguous_college_candidates"
    chosen_rows = next(iter(groups.values()))
    if _same_season_conflict(chosen_rows):
        return None, "ambiguous_college_candidates"
    return _latest_candidate(chosen_rows), None


def careers_by_player_page(seasons: list[dict]) -> list[list[dict]]:
    careers: dict[str, list[dict]] = {}
    for season in sorted(seasons, key=lambda s: s["season_end"]):
        careers.setdefault(season["href"], []).append(season)
    return list(careers.values())


def _draft_entry_for_career(
    name_key: str, career: list[dict], draft_entries_by_name: dict[str, list[dict]]
) -> dict | None:
    first_season = career[0]["season_end"]
    same_name = draft_entries_by_name.get(name_key, [])
    if not same_name:
        return None
    preferred = [draft_entry for draft_entry in same_name if draft_entry["draft_year"] == first_season - 1]
    if len(preferred) == 1:
        return preferred[0]
    plausible = [draft_entry for draft_entry in same_name if 0 <= first_season - draft_entry["draft_year"] <= 5]
    if len(plausible) == 1:
        return plausible[0]
    return None


def _corroborated_draft_entry(
    name_key: str,
    candidate: dict,
    draft_entries_by_name: dict[str, list[dict]],
    first_nba_season: int,
) -> dict | None:
    college = _normalize_school(candidate["college_team"])
    matches = [
        draft_entry
        for draft_entry in draft_entries_by_name.get(name_key, [])
        if int(candidate["season"]) <= int(draft_entry["draft_year"]) <= first_nba_season
        and _normalize_school(draft_entry.get("college")) == college
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_player_identity(
    name_key: str,
    career: list[dict],
    draft_entries_by_name: dict[str, list[dict]],
    college_index: dict[str, list[dict]],
) -> tuple[dict | None, dict | None, str | None]:
    first_season = career[0]["season_end"]
    draft_entry = _draft_entry_for_career(name_key, career, draft_entries_by_name)
    target_age: float | None = None
    if draft_entry is not None:
        anchor_year = int(draft_entry["draft_year"])
        college_hint = draft_entry.get("college")
    else:
        anchor_year = first_season - 1
        college_hint = None
        first_age = career[0].get("age")
        if first_age is not None and pd.notna(first_age):
            target_age = float(first_age) - 0.5
    candidate, reason = resolve_college_identity(
        college_index.get(name_key, []),
        anchor_year=anchor_year,
        college_hint=college_hint,
        target_age=target_age,
    )
    if candidate is not None and draft_entry is None:
        adopted = _corroborated_draft_entry(name_key, candidate, draft_entries_by_name, first_season)
        if adopted is not None:
            draft_entry = adopted
    return draft_entry, candidate, reason


def stable_player_id(name_key: str, birth_date: object, draft_year: int, team: str) -> str:
    dob = "" if pd.isna(birth_date) else pd.Timestamp(birth_date).strftime("%Y-%m-%d")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name_key}|{dob}|{draft_year}|{_normalize_school(team)}"))


def bio_identity_mismatch(candidate_team: str, colleges: list[str]) -> bool:
    if not colleges:
        return True
    candidate = _normalize_school(candidate_team)
    return not any(_school_prefix_match(candidate, _normalize_school(college)) for college in colleges)
