from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from arc.api.deps import position_filter, resolved_prospect
from arc.db import Player, get_db
from arc.schemas.common import PaginatedResponse, PaginationMeta
from arc.schemas.prediction import PredictionResponse, SimilarityResponse
from arc.schemas.prospect import CareerOutcome, DraftInfo, ProspectDetail, ProspectSummary
from arc.services.players import latest_season, query_prospects
from arc.services.prediction import prediction_service
from arc.services.similarity import similarity_service

router = APIRouter()


@router.get("", response_model=PaginatedResponse[ProspectSummary], summary="List and filter draft prospects")
def list_prospects(
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    search: Annotated[str | None, Query(description="Search prospect by player name")] = None,
    position: Annotated[str | None, Depends(position_filter)] = None,
    draft_year: Annotated[int | None, Query(description="Filter by NBA draft year (recorded draft picks)")] = None,
    college_team: Annotated[str | None, Query(description="Filter by college team")] = None,
    min_bpm: Annotated[
        float | None, Query(description="Filter by minimum Torvik BPM in the latest college season")
    ] = None,
):
    players, total_records = query_prospects(
        db=db,
        page=page,
        page_size=page_size,
        search=search,
        position=position,
        draft_year=draft_year,
        college_team=college_team,
        min_bpm=min_bpm,
    )
    items = []
    for player in players:
        season = latest_season(player)
        items.append(
            ProspectSummary(
                prospect_id=player.id,
                player_name=player.name,
                primary_position=player.primary_position,
                height_inches=player.height_inches,
                latest_team=season.college_team,
                latest_bpm=season.bpm,
                draft_pick=DraftInfo.model_validate(player.draft_pick) if player.draft_pick else None,
                nba_outcome=CareerOutcome.model_validate(player.nba_outcome) if player.nba_outcome else None,
            )
        )
    return PaginatedResponse(
        items=items,
        pagination=PaginationMeta(total_records=total_records, page=page, page_size=page_size),
    )


@router.get(
    "/{identifier}",
    response_model=ProspectDetail,
    summary="Get full prospect scouting profile",
)
def get_prospect_detail(player: Annotated[Player, Depends(resolved_prospect)]):
    return ProspectDetail.model_validate(player)


@router.get(
    "/{identifier}/predict",
    response_model=PredictionResponse,
    summary="Predict NBA career outcome tier for a prospect",
)
def predict_prospect(player: Annotated[Player, Depends(resolved_prospect)]):
    return prediction_service.get_or_predict_for_prospect(player)


@router.get(
    "/{identifier}/comparisons",
    response_model=SimilarityResponse,
    summary="Get historical college player comps",
)
def get_player_comparisons(
    player: Annotated[Player, Depends(resolved_prospect)],
    k: Annotated[int, Query(ge=1, le=10, description="Number of historical comps to return")] = 5,
):
    return similarity_service.get_or_find_comps_for_prospect(player, k=k)
