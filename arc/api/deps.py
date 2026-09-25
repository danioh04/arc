from typing import Annotated

from fastapi import Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session

from arc.core.constants import require_position
from arc.db import Player, get_db
from arc.services.players import find_player


def position_filter(
    position: Annotated[
        str | None,
        Query(description="Filter by position: G, W, F or C"),
    ] = None,
) -> str | None:
    if position is None or not position.strip():
        return None
    try:
        return require_position(position)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def resolved_prospect(
    identifier: Annotated[str, Path(min_length=1, description="Prospect UUID or player name")],
    db: Annotated[Session, Depends(get_db)],
    college: Annotated[str | None, Query(description="Optional college team name to disambiguate namesakes")] = None,
) -> Player:
    player = find_player(db, identifier, college=college)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Prospect {identifier!r} not found in database.")
    return player
