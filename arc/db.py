from collections.abc import Generator
from datetime import date
from uuid import UUID

from sqlalchemy import ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from arc.core import config

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    clean_name: Mapped[str] = mapped_column(String(100))
    birth_date: Mapped[date | None]
    height_inches: Mapped[int | None]
    primary_position: Mapped[str | None] = mapped_column(String(10))
    college_seasons: Mapped[list["CollegeStats"]] = relationship(
        "CollegeStats",
        back_populates="player",
        order_by="CollegeStats.season",
    )
    draft_pick: Mapped["DraftPick | None"] = relationship(
        "DraftPick",
        back_populates="player",
        uselist=False,
    )
    nba_outcome: Mapped["NBAOutcome | None"] = relationship(
        "NBAOutcome",
        back_populates="player",
        uselist=False,
    )


class CollegeStats(Base):
    __tablename__ = "college_stats"
    player_id: Mapped[UUID] = mapped_column(ForeignKey("players.id"), primary_key=True)
    season: Mapped[int] = mapped_column(primary_key=True)
    college_team: Mapped[str] = mapped_column(String(50))
    conference: Mapped[str | None] = mapped_column(String(20))
    class_year: Mapped[str | None] = mapped_column(String(10))
    draft_age: Mapped[float | None]
    min_pct: Mapped[float | None]
    usg_pct: Mapped[float | None]
    ts_pct: Mapped[float | None]
    orb_pct: Mapped[float | None]
    drb_pct: Mapped[float | None]
    ast_pct: Mapped[float | None]
    to_pct: Mapped[float | None]
    blk_pct: Mapped[float | None]
    stl_pct: Mapped[float | None]
    ftr: Mapped[float | None]
    fg3_pct: Mapped[float | None]
    fg3a_per_fga: Mapped[float | None]
    ft_pct: Mapped[float | None]
    bpm: Mapped[float | None]
    obpm: Mapped[float | None]
    dbpm: Mapped[float | None]
    player: Mapped["Player"] = relationship("Player", back_populates="college_seasons")


class DraftPick(Base):
    __tablename__ = "draft_picks"
    player_id: Mapped[UUID] = mapped_column(ForeignKey("players.id"), primary_key=True)
    draft_year: Mapped[int] = mapped_column(index=True)
    overall_pick: Mapped[int]
    drafting_team: Mapped[str | None] = mapped_column(String(50))
    player: Mapped["Player"] = relationship("Player", back_populates="draft_pick")


class NBAOutcome(Base):
    __tablename__ = "nba_outcomes"
    player_id: Mapped[UUID] = mapped_column(ForeignKey("players.id"), primary_key=True)
    peak_bpm: Mapped[float | None]
    peak_vorp: Mapped[float | None]
    peak_mpg: Mapped[float | None]
    career_win_shares: Mapped[float | None]
    seasons_played: Mapped[int]
    games_played: Mapped[int]
    career_tier: Mapped[int]
    tier_label: Mapped[str] = mapped_column(String(50))
    is_active: Mapped[bool]
    player: Mapped["Player"] = relationship("Player", back_populates="nba_outcome")
