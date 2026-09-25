import sqlite3
import tomllib
import uuid

import numpy as np
import pandas as pd
import pytest

from arc.core.config import ARTIFACTS_DIR, DB_PATH, ROOT
from arc.ml.bundle import read_bundle
from arc.ml.feature_engineering import FEATURE_COLUMNS, engineer_features
from arc.ml.tiers import MIN_CAREER_GAMES, calculate_outcome_tier, calculate_tier_probabilities
from data_pipeline.config import LABELLED_PATH, SERVING_PLAYERS_PATH, SERVING_SEASONS_PATH
from data_pipeline.identity import careers_by_player_page, resolve_college_identity
from data_pipeline.labels import ComponentAnchors, calculate_composite_z
from data_pipeline.training.evaluation import spearman, top_k_star_precision
from data_pipeline.training.model_card import TRACKED_LIBRARIES, dataset_fingerprint

MIN_SPEARMAN = 0.20
MIN_STAR_PRECISION_AT_10 = 0.20
MIN_STAR_LIFT = 2.5

ANCHORS = ComponentAnchors(bpm_mean=0.0, bpm_std=2.0, vorp_mean=1.0, vorp_std=3.0, mpg_mean=20.0, mpg_std=7.0)


@pytest.fixture(scope="module")
def bundle() -> dict:
    return read_bundle(ARTIFACTS_DIR)


@pytest.fixture(scope="module")
def db():
    connection = sqlite3.connect(DB_PATH)
    yield connection
    connection.close()


def _candidate(season: int, team: str, birth_date: str) -> dict:
    return {
        "season": season,
        "row": pd.Series({"season": season, "college_team": team, "player_name": "Test Player"}),
        "college_team": team,
        "birth_date": pd.Timestamp(birth_date),
    }


def test_the_bundle_was_trained_on_the_committed_dataset(bundle):
    assert bundle["model_card"]["dataset_fingerprint"] == dataset_fingerprint(LABELLED_PATH), (
        "the model bundle was trained on a different dataset; re-run the training stage"
    )


def test_the_pinned_libraries_are_the_ones_the_bundle_was_trained_with(bundle):
    requirements = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    pins = {
        name.strip().lower(): version.strip()
        for name, _, version in (r.partition("==") for r in requirements)
        if version
    }

    assert {library: pins.get(library) for library in TRACKED_LIBRARIES} == bundle["model_card"]["library_versions"]


def test_the_holdout_ranking_clears_its_floors(bundle):
    label_spec = bundle["model_card"]["label_spec"]
    labelled = pd.read_csv(LABELLED_PATH)
    holdout = labelled[labelled.split == "holdout"]
    actual_z = holdout["composite_z"].fillna(label_spec["sentinel_composite_z"]).to_numpy(dtype=float)
    projected_z = bundle["regressor"].predict(engineer_features(holdout)[FEATURE_COLUMNS].to_numpy(dtype=float))
    probabilities = np.array(
        [
            calculate_tier_probabilities(float(z), bundle["residuals"], label_spec["tier_boundaries"])
            for z in projected_z
        ]
    )
    actual_class = (holdout["career_tier"] - 1).to_numpy()
    correlation = spearman(actual_z, projected_z)
    precision_at_10 = top_k_star_precision(actual_class, probabilities[:, 0] + probabilities[:, 1], k=10)
    lift = precision_at_10 / np.isin(actual_class, (0, 1)).mean()
    print(
        f"\n[holdout n={len(holdout)}] Spearman {correlation:+.4f} | precision@10 {precision_at_10:.2f} | lift {lift:.1f}x"
    )
    assert len(holdout) >= 120
    assert correlation >= MIN_SPEARMAN
    assert precision_at_10 >= MIN_STAR_PRECISION_AT_10
    assert lift >= MIN_STAR_LIFT


def test_the_database_is_exactly_the_serving_files(db):
    players = pd.read_csv(SERVING_PLAYERS_PATH)
    seasons = pd.read_csv(SERVING_SEASONS_PATH)
    labelled_ids = {uuid.UUID(str(player_id)).hex for player_id in pd.read_csv(LABELLED_PATH)["player_id"]}
    db_players = set(
        db.execute("SELECT id, name, clean_name, birth_date, height_inches, primary_position FROM players")
    )
    csv_players = {
        (
            uuid.UUID(str(row.player_id)).hex,
            row.player_name,
            row.clean_name,
            None if pd.isna(row.birth_date) else row.birth_date,
            None if pd.isna(row.height_inches) else int(row.height_inches),
            None if pd.isna(row.primary_position) else row.primary_position,
        )
        for row in players.itertuples()
    }
    db_seasons = set(db.execute("SELECT player_id, season, college_team FROM college_stats"))
    csv_seasons = {
        (uuid.UUID(str(row.player_id)).hex, int(row.season), row.college_team) for row in seasons.itertuples()
    }
    assert db_players == csv_players
    assert db_seasons == csv_seasons
    assert labelled_ids <= {player[0] for player in db_players}


def test_every_served_identity_is_unique_and_has_a_college_season(db):
    shared = db.execute(
        "SELECT clean_name, birth_date FROM players WHERE birth_date IS NOT NULL GROUP BY 1, 2 HAVING count(*) > 1"
    ).fetchall()
    seasonless = db.execute("SELECT count(*) FROM players WHERE id NOT IN (SELECT player_id FROM college_stats)")
    assert shared == []
    assert seasonless.fetchone()[0] == 0


def test_the_composite_label_weighs_its_components_and_needs_a_real_career():
    value = calculate_composite_z(5.0, 30.0, 1.0, 600, ANCHORS)
    missing_vorp = calculate_composite_z(5.0, 30.0, None, 600, ANCHORS)
    assert value == pytest.approx(round(0.40 * (5.0 / 2.0) + 0.30 * 0.0 + 0.30 * (10.0 / 7.0), 4))
    assert missing_vorp == value
    assert calculate_composite_z(5.0, 30.0, 1.0, MIN_CAREER_GAMES - 1, ANCHORS) is None


def test_a_cutoff_belongs_to_the_higher_tier():
    expected = {2.5: 1, 1.90: 1, 1.8999: 2, 1.30: 2, 0.45: 3, -0.20: 4, -0.2001: 5, None: 5}
    assert {z: calculate_outcome_tier(z, [-0.20, 0.45, 1.30, 1.90])[0] for z in expected} == expected


def test_tier_probabilities_form_a_distribution_and_a_projection_on_a_cutoff_has_not_cleared_it():
    probabilities = calculate_tier_probabilities(0.0, [-1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 2.0, 3.0])
    assert probabilities[4] == pytest.approx(0.75)
    assert probabilities[3] == pytest.approx(0.25)
    assert sum(probabilities) == pytest.approx(1.0)


def test_same_name_college_players_are_told_apart_by_age_or_refused():
    older, younger = _candidate(2010, "Old School", "1980-01-01"), _candidate(2010, "Young School", "1990-01-01")
    chosen, reason = resolve_college_identity([older, younger], anchor_year=2010, target_age=20.5)
    assert chosen is not None
    assert (chosen["college_team"], reason) == ("Young School", None)
    assert resolve_college_identity([older, younger], anchor_year=2010) == (None, "ambiguous_college_candidates")


def test_same_name_nba_players_in_the_same_seasons_get_one_career_each():
    seasons = [
        {"season_end": 2009, "href": "/players/w/willima03.html"},
        {"season_end": 2009, "href": "/players/w/willima04.html"},
        {"season_end": 2010, "href": "/players/w/willima03.html"},
    ]
    careers = careers_by_player_page(seasons)
    assert [[season["season_end"] for season in career] for career in careers] == [[2009, 2010], [2009]]
