from sqlalchemy.orm import Session

from arc.db import CollegeStats, Player
from arc.ml.feature_engineering import resolve_draft_age
from arc.schemas.prospect import BigBoardItem, BigBoardResponse
from arc.services.cache import read_cached, write_cached
from arc.services.players import drafted_players, latest_season
from arc.services.prediction import prediction_service


def build_big_board(db: Session, draft_year: int) -> list[BigBoardItem]:
    candidates: list[tuple[Player, CollegeStats, int]] = []
    for player in drafted_players(db, draft_year):
        assert player.draft_pick is not None
        candidates.append((player, latest_season(player), player.draft_pick.overall_pick))
    predictions = prediction_service.predict_batch([player for player, _, _ in candidates])
    ranked = sorted(
        zip(candidates, predictions, strict=True),
        key=lambda item: item[1].projected_z,
        reverse=True,
    )
    pick_order = {pick: order for order, pick in enumerate(sorted(pick for _, _, pick in candidates), start=1)}
    board: list[BigBoardItem] = []
    for rank, ((player, season, pick), projection) in enumerate(ranked, start=1):
        draft_age = resolve_draft_age(draft_year, player.birth_date)
        board.append(
            BigBoardItem(
                model_rank=rank,
                prospect_id=player.id,
                player_name=player.name,
                primary_position=player.primary_position,
                height_inches=player.height_inches,
                latest_team=season.college_team,
                conference=season.conference,
                draft_age=draft_age if draft_age is not None else season.draft_age,
                latest_bpm=season.bpm,
                ts_pct=season.ts_pct,
                overall_pick=pick,
                value_delta=pick_order[pick] - rank,
                projected_tier=projection.projected_tier,
                projected_tier_label=projection.projected_tier_label,
                star_probability=projection.star_probability,
                bust_risk=projection.bust_risk,
                projected_z=round(projection.projected_z, 3),
            )
        )
    return board


def get_or_build_big_board(db: Session, draft_year: int, limit: int) -> BigBoardResponse | None:
    key = f"nba:big_board:{draft_year}"
    cached = read_cached(key, BigBoardResponse)
    if cached is None:
        board = build_big_board(db, draft_year)
        if not board:
            return None
        cached = BigBoardResponse(draft_year=draft_year, total_prospects=len(board), board=board)
        write_cached(key, cached)
    return BigBoardResponse(draft_year=draft_year, total_prospects=cached.total_prospects, board=cached.board[:limit])
