from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from arc.db import get_db
from arc.schemas.prospect import BigBoardResponse, DraftClassList
from arc.services.players import get_available_draft_classes
from arc.services.ranking import get_or_build_big_board

router = APIRouter()


@router.get(
    "/draft-classes",
    response_model=DraftClassList,
    summary="List all available draft classes",
)
def list_draft_classes(
    db: Annotated[Session, Depends(get_db)],
):
    return DraftClassList(classes=get_available_draft_classes(db))


@router.get(
    "/draft-classes/{draft_year}/big-board",
    response_model=BigBoardResponse,
    summary="Get model big board for a draft class",
)
def get_big_board(
    draft_year: int,
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=500, description="Top N prospects to return (default: 60)")] = 60,
):
    board = get_or_build_big_board(db, draft_year, limit=limit)
    if board is None:
        raise HTTPException(status_code=404, detail=f"No prospect records found for draft class year {draft_year}.")
    return board
