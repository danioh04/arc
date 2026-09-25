from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, joinedload, selectinload

from arc.core.names import clean_name
from arc.db import CollegeStats, DraftPick, Player
from arc.schemas.prospect import DraftClassSummary

_EAGER_OPTIONS = (
    selectinload(Player.college_seasons),
    joinedload(Player.draft_pick),
    joinedload(Player.nba_outcome),
)

_BOARD_OPTIONS = (
    selectinload(Player.college_seasons),
    joinedload(Player.draft_pick),
)


def _matches_college(player: Player, college: str) -> bool:
    return any(college in season.college_team.lower() for season in player.college_seasons)


def _most_relevant(players: list[Player]) -> Player | None:
    if not players:
        return None
    return max(
        players,
        key=lambda player: (
            2 if player.nba_outcome else (1 if player.draft_pick else 0),
            player.nba_outcome.peak_vorp
            if (player.nba_outcome and player.nba_outcome.peak_vorp is not None)
            else -99.0,
            max((season.season for season in player.college_seasons), default=0),
        ),
    )


def find_player(db: Session, identifier: str, college: str | None = None) -> Player | None:
    raw_id = str(identifier).strip()
    wanted_college = (college or "").strip().lower() or None
    try:
        player = db.scalars(select(Player).where(Player.id == UUID(raw_id)).options(*_EAGER_OPTIONS)).unique().first()
    except ValueError:
        player = None
    if player is not None:
        return player if wanted_college is None or _matches_college(player, wanted_college) else None
    compact = clean_name(raw_id).replace(" ", "")
    if not compact:
        return None
    statement = (
        select(Player).where(func.replace(Player.clean_name, " ", "") == compact).options(*_EAGER_OPTIONS).limit(20)
    )
    candidates = list(db.scalars(statement).unique().all())
    if wanted_college is not None:
        candidates = [player for player in candidates if _matches_college(player, wanted_college)]
    return _most_relevant(candidates)


def latest_season(player: Player) -> CollegeStats:
    return max(player.college_seasons, key=lambda season: season.season)


def _like_pattern(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def query_prospects(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    position: str | None = None,
    draft_year: int | None = None,
    college_team: str | None = None,
    min_bpm: float | None = None,
) -> tuple[list[Player], int]:
    base_query = select(Player)
    if search and search.strip():
        raw_search = search.strip().lower()
        patterns = [func.lower(Player.name).like(_like_pattern(raw_search), escape="\\")]
        clean_search = clean_name(search)
        if clean_search:
            patterns.append(Player.clean_name.like(_like_pattern(clean_search), escape="\\"))
        base_query = base_query.where(or_(*patterns))
    if position:
        base_query = base_query.where(Player.primary_position == position)
    if draft_year:
        draft_pick_subquery = select(DraftPick.player_id).where(DraftPick.draft_year == draft_year)
        base_query = base_query.where(Player.id.in_(draft_pick_subquery))
    if college_team and college_team.strip():
        team_filter = func.lower(CollegeStats.college_team).like(
            _like_pattern(college_team.strip().lower()), escape="\\"
        )
        base_query = base_query.where(Player.id.in_(select(CollegeStats.player_id).where(team_filter)))
    if min_bpm is not None:
        latest = (
            select(CollegeStats.player_id, func.max(CollegeStats.season).label("latest_season"))
            .group_by(CollegeStats.player_id)
            .subquery()
        )
        base_query = base_query.where(
            Player.id.in_(
                select(CollegeStats.player_id)
                .join(
                    latest,
                    and_(
                        CollegeStats.player_id == latest.c.player_id,
                        CollegeStats.season == latest.c.latest_season,
                    ),
                )
                .where(CollegeStats.bpm >= min_bpm)
            )
        )
    total_records = db.scalar(select(func.count()).select_from(base_query.subquery())) or 0
    offset = (page - 1) * page_size
    fetch_query = base_query.options(*_EAGER_OPTIONS).order_by(Player.name.asc()).offset(offset).limit(page_size)
    return list(db.scalars(fetch_query).unique().all()), total_records


def drafted_players(db: Session, draft_year: int) -> list[Player]:
    statement = (
        select(Player).join(Player.draft_pick).where(DraftPick.draft_year == draft_year).options(*_BOARD_OPTIONS)
    )
    return list(db.scalars(statement).unique().all())


def get_available_draft_classes(db: Session) -> list[DraftClassSummary]:
    rows = db.execute(
        select(DraftPick.draft_year, func.count()).group_by(DraftPick.draft_year).order_by(DraftPick.draft_year.desc())
    ).all()
    return [DraftClassSummary(draft_year=year, drafted_count=count) for year, count in rows]
