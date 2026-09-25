from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CollegeSeason(BaseModel):
    season: int = Field(..., description="NCAA season year (e.g. 2024 for 2023-24 season)")
    college_team: str = Field(..., description="College team name")
    conference: str | None = Field(None, description="College conference (e.g. SEC, ACC, B10)")
    class_year: str | None = Field(None, description="Class year: Fr, So, Jr, Sr, Gr")
    draft_age: float | None = Field(
        None, description="Age in June of this season (a class-year estimate when the birth date is unknown)"
    )
    min_pct: float | None = Field(None, description="Percentage of team minutes played")
    usg_pct: float | None = Field(None, description="Usage percentage")
    ts_pct: float | None = Field(None, description="True shooting percentage (0.0 to 1.0)")
    orb_pct: float | None = Field(None, description="Offensive rebound percentage")
    drb_pct: float | None = Field(None, description="Defensive rebound percentage")
    ast_pct: float | None = Field(None, description="Assist percentage")
    to_pct: float | None = Field(None, description="Turnover percentage")
    blk_pct: float | None = Field(None, description="Block percentage")
    stl_pct: float | None = Field(None, description="Steal percentage")
    ftr: float | None = Field(None, description="Free throw rate (FTA / FGA * 100)")
    ft_pct: float | None = Field(None, description="Free throw percentage (0.0 to 1.0)")
    fg3_pct: float | None = Field(None, description="3-Point field goal percentage")
    fg3a_per_fga: float | None = Field(None, description="3-point attempt rate (3PA / FGA)")
    bpm: float | None = Field(None, description="BartTorvik Box Plus-Minus")
    obpm: float | None = Field(None, description="Offensive Box Plus-Minus")
    dbpm: float | None = Field(None, description="Defensive Box Plus-Minus")
    model_config = ConfigDict(from_attributes=True)


class DraftInfo(BaseModel):
    draft_year: int = Field(..., description="Year drafted into NBA")
    overall_pick: int = Field(..., description="Overall draft pick number (1 to 60)")
    drafting_team: str | None = Field(None, description="Team that drafted the player")
    model_config = ConfigDict(from_attributes=True)


class CareerOutcome(BaseModel):
    career_tier: int = Field(..., ge=1, le=5, description="Objective outcome tier (1 to 5)")
    tier_label: str = Field(..., description="Outcome tier label (e.g. Franchise Star, Quality Starter, End of Bench)")
    peak_bpm: float | None = Field(None, description="Peak average NBA BPM over the best window")
    peak_vorp: float | None = Field(None, description="Peak 3-year window cumulative NBA VORP")
    peak_mpg: float | None = Field(None, description="Peak NBA minutes per game")
    career_win_shares: float | None = Field(None, description="Total career NBA Win Shares")
    seasons_played: int = Field(..., description="Total NBA seasons played")
    games_played: int = Field(..., description="Career NBA games played")
    is_active: bool = Field(..., description="Played in the most recent completed NBA season")
    model_config = ConfigDict(from_attributes=True)


class ProspectSummary(BaseModel):
    prospect_id: UUID = Field(..., description="Unique prospect UUID")
    player_name: str = Field(..., description="Player full name")
    primary_position: str | None = Field(None, description="Primary position (G, W, F, C)")
    height_inches: int | None = Field(None, description="Height in inches")
    latest_team: str = Field(..., description="Latest college team")
    latest_bpm: float | None = Field(None, description="Latest Torvik BPM")
    draft_pick: DraftInfo | None = None
    nba_outcome: CareerOutcome | None = None


class ProspectDetail(BaseModel):
    prospect_id: UUID = Field(..., validation_alias="id", description="Unique prospect UUID")
    player_name: str = Field(..., validation_alias="name", description="Player full name")
    primary_position: str | None = Field(None, description="Primary position (G, W, F, C)")
    height_inches: int | None = Field(None, description="Height in inches")
    draft_pick: DraftInfo | None = None
    nba_outcome: CareerOutcome | None = None
    college_seasons: list[CollegeSeason] = Field(default_factory=list, description="All college seasons")
    model_config = ConfigDict(from_attributes=True)


class BigBoardItem(BaseModel):
    model_rank: int = Field(..., description="Prospect rank on model Big Board")
    prospect_id: UUID = Field(..., description="Prospect UUID")
    player_name: str = Field(..., description="Player name")
    primary_position: str | None = Field(None, description="Position (G, W, F, C)")
    height_inches: int | None = Field(None, description="Height in inches")
    latest_team: str = Field(..., description="Latest college team")
    conference: str | None = Field(None, description="College conference (e.g. SEC, ACC, B10)")
    draft_age: float | None = Field(None, description="Draft age")
    latest_bpm: float | None = Field(None, description="Latest Torvik BPM")
    ts_pct: float | None = Field(None, description="True shooting percentage")
    overall_pick: int = Field(..., description="Actual NBA draft slot")
    value_delta: int = Field(
        ...,
        description=(
            "Draft value surplus: the pick's order among board players minus model_rank "
            "(positive = steal, negative = reach)"
        ),
    )
    projected_tier: int = Field(..., ge=1, le=5, description="Projected tier (1 to 5)")
    projected_tier_label: str = Field(..., description="Projected tier label")
    star_probability: float = Field(..., description="Probability of a Tier 1-2 Star outcome (0.0 to 1.0)")
    bust_risk: float = Field(..., description="Probability of a Tier 5 End of Bench outcome (0.0 to 1.0)")
    projected_z: float = Field(..., description="Projected composite Z-Score; the board is sorted by it")


class BigBoardResponse(BaseModel):
    draft_year: int = Field(..., description="Draft class year")
    total_prospects: int = Field(..., description="Players on the class board, before the limit was applied")
    board: list[BigBoardItem] = Field(..., description="Ranked prospect list")


class DraftClassSummary(BaseModel):
    draft_year: int = Field(..., description="Draft class year")
    drafted_count: int = Field(..., description="Number of drafted players with a recorded pick")


class DraftClassList(BaseModel):
    classes: list[DraftClassSummary] = Field(..., description="List of draft classes")
