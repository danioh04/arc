import logging
import re
import time
from datetime import date
from io import StringIO
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import requests

from arc.core.config import ROOT

logger = logging.getLogger(__name__)

BR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2
REQUEST_DELAY_SECONDS = 1.5

BR_DRAFT_URL = "https://www.basketball-reference.com/draft/NBA_{year}.html"
BR_SEASON_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
BR_PLAYER_URL = "https://www.basketball-reference.com{path}"

DRAFT_CACHE_DIR = ROOT / "data" / "cache" / "br" / "drafts"
SEASON_CACHE_DIR = ROOT / "data" / "cache" / "br" / "seasons"
PLAYER_CACHE_DIR = ROOT / "data" / "cache" / "br" / "players"

_DRAFT_CACHE_MIN_BYTES = 10_000
_SEASON_CACHE_MIN_BYTES = 10_000
_PLAYER_CACHE_MIN_BYTES = 1_000

SEASON_COLUMNS = ["player", "age", "g", "mp", "ws", "bpm", "vorp"]

_DRAFT_COLUMN_MAP = {
    "pk": "overall_pick",
    "tm": "nba_team",
    "player": "player_name",
    "college": "college_name",
}

_BIRTH_DATE = re.compile(r'id="necro-birth"[^>]*data-birth="(\d{4}-\d{2}-\d{2})"')
_COLLEGE_BLOCK = re.compile(r"Colleges?:\s*</strong>(.*?)</p>", re.DOTALL)
_ANCHOR_TEXT = re.compile(r">([^<>]+)</a>")


class PlayerBio(NamedTuple):
    birth_date: date | None
    colleges: list[str]


def download_text(url: str) -> str | None:
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            response = requests.get(url, headers=BR_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.encoding = "utf-8"
            if response.status_code == 200:
                return response.text
            logger.warning(f"{url} returned {response.status_code} (attempt {attempt + 1})")
        except requests.RequestException as e:
            logger.warning(f"{url} failed ({e}); attempt {attempt + 1}")
        if attempt < REQUEST_ATTEMPTS - 1:
            time.sleep(BACKOFF_BASE_SECONDS * (attempt + 1))
    return None


def fetch_cached_text(cache_path: Path, url: str, *, min_bytes: int) -> str | None:
    if cache_path.exists() and cache_path.stat().st_size > min_bytes:
        return cache_path.read_text(encoding="utf-8")
    html = download_text(url)
    if html is None:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)
    return html


def _flatten_table(table: pd.DataFrame) -> pd.DataFrame:
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [column[-1] if isinstance(column, tuple) else column for column in table.columns]
    table.columns = [str(column).strip().lower() for column in table.columns]
    return table.loc[:, ~pd.Index(table.columns).duplicated()]


def _read_table(html: str, table_id: str, cache_path: Path, label: str, *, extract_links: bool) -> pd.DataFrame | None:
    uncommented = re.sub(r"<!--|-->", "", html)
    try:
        tables = pd.read_html(
            StringIO(uncommented), attrs={"id": table_id}, extract_links="body" if extract_links else None
        )
    except ValueError:
        logger.warning(f"No {table_id} table found in {label}")
        cache_path.unlink(missing_ok=True)
        return None
    return _flatten_table(tables[0])


def drop_non_pick_rows(df: pd.DataFrame) -> pd.DataFrame:
    names = df["player_name"].astype("string").str.strip()
    picks = pd.to_numeric(df["overall_pick"], errors="coerce")
    valid_name = names.notna() & (names != "") & (names.str.lower() != "player")
    cleaned = df[valid_name & picks.notna()].copy()
    cleaned["overall_pick"] = pd.to_numeric(cleaned["overall_pick"], errors="coerce")
    return cleaned


def _fetch_draft_class(year: int) -> pd.DataFrame | None:
    cache_path = DRAFT_CACHE_DIR / f"NBA_{year}.html"
    html = fetch_cached_text(cache_path, BR_DRAFT_URL.format(year=year), min_bytes=_DRAFT_CACHE_MIN_BYTES)
    if html is None:
        return None
    table = _read_table(html, "stats", cache_path, f"draft table for {year}", extract_links=False)
    if table is None:
        return None
    df = table.rename(
        columns={column: _DRAFT_COLUMN_MAP[column] for column in table.columns if column in _DRAFT_COLUMN_MAP}
    )
    if not {"player_name", "overall_pick"}.issubset(df.columns):
        logger.warning(f"Draft table for {year} missing expected columns; got {list(df.columns)}")
        return None
    keep = [column for column in _DRAFT_COLUMN_MAP.values() if column in df.columns]
    df = drop_non_pick_rows(df[keep])
    df["draft_year"] = year
    logger.info(f"Parsed {len(df)} drafted players for {year}")
    return df


def fetch_drafts(start_year: int, end_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    for year in range(start_year, end_year + 1):
        df = _fetch_draft_class(year)
        if df is None or df.empty:
            logger.warning(f"Draft class {year} produced no usable rows.")
            missing.append(year)
            continue
        frames.append(df)
    if missing:
        raise RuntimeError(
            f"No draft class could be fetched for {missing}; check the network connection or the on-disk HTML cache."
        )
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Fetched {len(combined)} drafted players across {start_year}-{end_year}")
    return combined


def normalize_season_rows(table: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_table(table)
    missing = [column for column in SEASON_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Advanced table missing columns {missing}; found {list(df.columns)}")
    hrefs = df["player"].map(lambda value: value[1] if isinstance(value, tuple) else None)
    df = df.map(lambda value: value[0] if isinstance(value, tuple) else value)
    df["player_href"] = hrefs
    df = df[[*SEASON_COLUMNS, "player_href"]]
    players = df["player"].astype("string").str.strip()
    df = df[players.notna() & (players != "") & (players.str.lower() != "league average")]
    for column in SEASON_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values("mp", ascending=False).drop_duplicates(subset=["player", "player_href"], keep="first")


def _fetch_season(year: int) -> pd.DataFrame | None:
    cache_path = SEASON_CACHE_DIR / f"NBA_{year}.html"
    html = fetch_cached_text(cache_path, BR_SEASON_URL.format(year=year), min_bytes=_SEASON_CACHE_MIN_BYTES)
    if html is None:
        return None
    table = _read_table(html, "advanced", cache_path, f"advanced table for {year}", extract_links=True)
    if table is None:
        return None
    df = normalize_season_rows(table)
    df.insert(0, "season_end", year)
    logger.info(f"Parsed {len(df)} player rows for NBA {year - 1}-{str(year)[2:]}")
    return df


def fetch_seasons(start_year: int, end_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    for year in range(start_year, end_year + 1):
        df = _fetch_season(year)
        if df is None or df.empty:
            logger.warning(f"NBA season {year} produced no usable rows.")
            missing.append(year)
            continue
        frames.append(df)
    if missing:
        raise RuntimeError(
            f"No NBA season data could be fetched for {missing}; "
            "check the network connection or the on-disk HTML cache."
        )
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Fetched {combined['player'].nunique()} unique players across {start_year}-{end_year}")
    return combined


def fetch_bio(player_path: str | None) -> PlayerBio | None:
    if not player_path or not isinstance(player_path, str):
        return None
    cache_path = PLAYER_CACHE_DIR / Path(player_path).name
    html = fetch_cached_text(cache_path, BR_PLAYER_URL.format(path=player_path), min_bytes=_PLAYER_CACHE_MIN_BYTES)
    if html is None:
        return None
    birth = _BIRTH_DATE.search(html)
    college_block = _COLLEGE_BLOCK.search(html)
    colleges = [text.strip() for text in _ANCHOR_TEXT.findall(college_block.group(1))] if college_block else []
    return PlayerBio(
        birth_date=date.fromisoformat(birth.group(1)) if birth else None,
        colleges=[college for college in colleges if college],
    )
