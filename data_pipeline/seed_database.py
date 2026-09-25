import logging
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import insert
from sqlalchemy.orm import Session

from arc.db import Base, CollegeStats, DraftPick, NBAOutcome, Player, SessionLocal, engine
from data_pipeline.config import LABELLED_PATH, SERVING_PLAYERS_PATH, SERVING_SEASON_COLUMNS, SERVING_SEASONS_PATH

logger = logging.getLogger(__name__)


def _or_none(value):
    return None if pd.isna(value) else value


def _int_or_none(value):
    return None if pd.isna(value) else int(value)


def _date_or_none(value):
    return None if pd.isna(value) else date.fromisoformat(value)


def load_serving_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(SERVING_PLAYERS_PATH)
    seasons = pd.read_csv(SERVING_SEASONS_PATH)
    labelled = pd.read_csv(LABELLED_PATH)
    known_ids = set(players["player_id"])
    for name, frame in (("serving_seasons.csv", seasons), ("labelled.csv", labelled)):
        orphans = sorted(set(frame["player_id"]) - known_ids)
        if orphans:
            raise ValueError(f"{name} references players absent from {SERVING_PLAYERS_PATH}: {orphans[:10]}")
    return players, seasons, labelled


def insert_serving_population(
    session: Session,
    players: pd.DataFrame,
    seasons: pd.DataFrame,
    labelled: pd.DataFrame,
) -> None:
    player_rows = [
        {
            "id": uuid.UUID(str(row.player_id)),
            "name": str(row.player_name),
            "clean_name": str(row.clean_name),
            "birth_date": _date_or_none(row.birth_date),
            "height_inches": _int_or_none(row.height_inches),
            "primary_position": _or_none(row.primary_position),
        }
        for row in players.itertuples()
    ]
    session.execute(insert(Player), player_rows)
    pick_rows = [
        {
            "player_id": uuid.UUID(str(row.player_id)),
            "draft_year": int(row.draft_year),
            "overall_pick": int(row.overall_pick),
            "drafting_team": _or_none(row.drafting_team),
        }
        for row in players.itertuples()
        if pd.notna(row.draft_year) and pd.notna(row.overall_pick)
    ]
    if pick_rows:
        session.execute(insert(DraftPick), pick_rows)
    season_rows = [
        {
            "player_id": uuid.UUID(str(row.player_id)),
            **{column: _or_none(getattr(row, column)) for column in SERVING_SEASON_COLUMNS if column != "season"},
            "season": int(row.season),
        }
        for row in seasons.itertuples()
    ]
    session.execute(insert(CollegeStats), season_rows)
    outcome_rows = [
        {
            "player_id": uuid.UUID(str(row.player_id)),
            "career_tier": int(row.career_tier),
            "tier_label": str(row.tier_label),
            "peak_bpm": _or_none(row.peak_bpm),
            "peak_vorp": _or_none(row.peak_vorp),
            "peak_mpg": _or_none(row.peak_mpg),
            "career_win_shares": _or_none(row.career_win_shares),
            "seasons_played": int(row.seasons_played),
            "games_played": int(row.games_played),
            "is_active": bool(row.is_active),
        }
        for row in labelled.itertuples()
    ]
    session.execute(insert(NBAOutcome), outcome_rows)


def seed_database() -> None:
    players, seasons, labelled = load_serving_frames()
    logger.info(
        f"Rebuilding database at {engine.url.render_as_string(hide_password=True)} from the serving files "
        f"({len(players)} players, {len(seasons)} player-seasons, {len(labelled)} labelled)..."
    )
    database_path = engine.url.database
    engine.dispose()
    if database_path and database_path != ":memory:":
        for suffix in ("", "-wal", "-shm"):
            Path(f"{database_path}{suffix}").unlink(missing_ok=True)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        insert_serving_population(session, players, seasons, labelled)
        session.commit()
    with engine.connect() as connection:
        connection.exec_driver_sql("VACUUM")
    logger.info("VACUUMmed SQLite database to remove freelist pages.")
