import json
import logging
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from arc.ml.tiers import MIN_CAREER_GAMES, calculate_outcome_tier
from data_pipeline.config import LABEL_SPEC_PATH

logger = logging.getLogger(__name__)

TIER_PERCENTILES = (50.0, 75.0, 90.0, 96.0)

SENTINEL_MARGIN = 0.25

MIN_PEAK_SEASON_MP = 250.0
PEAK_WINDOW = 3


class ComponentAnchors(NamedTuple):
    bpm_mean: float
    bpm_std: float
    vorp_mean: float
    vorp_std: float
    mpg_mean: float
    mpg_std: float


def empty_peak_stats(games_played: int, seasons_played: int = 0, career_ws: float | None = None) -> dict[str, Any]:
    return {
        "peak_bpm": None,
        "peak_mpg": None,
        "peak_vorp": None,
        "career_win_shares": round(career_ws, 2) if career_ws is not None else None,
        "seasons_played": seasons_played,
        "games_played": games_played,
        "last_season_end": None,
        "peak_window_seasons": 0,
    }


def compute_peak_stats(seasons: list[dict[str, Any]]) -> dict[str, Any]:
    played = [s for s in seasons if pd.notna(s.get("g")) and s["g"] > 0]
    if not played:
        return empty_peak_stats(games_played=0)
    seasons_played = len(played)
    games_played = int(sum(int(s["g"]) for s in played))
    career_ws = float(sum(float(s["ws"]) for s in played if pd.notna(s.get("ws"))))
    qualified = [s for s in played if pd.notna(s.get("mp")) and float(s["mp"]) >= MIN_PEAK_SEASON_MP]
    usable = sorted(
        (s for s in qualified if pd.notna(s.get("bpm"))),
        key=lambda s: s["season_end"],
    )
    if not usable:
        return empty_peak_stats(games_played=games_played, seasons_played=seasons_played, career_ws=career_ws)
    best: list[dict[str, Any]] = []
    best_window_size = 0
    for size in (PEAK_WINDOW, 2, 1):
        windows = [
            usable[i : i + size]
            for i in range(len(usable) - size + 1)
            if (usable[i + size - 1]["season_end"] - usable[i]["season_end"]) == size - 1
        ]
        if windows:
            best = max(windows, key=lambda w: sum(s["bpm"] for s in w) / len(w))
            best_window_size = size
            break
    total_mp = sum(float(s["mp"]) for s in best)
    total_g = sum(int(s["g"]) for s in best)
    vorp_present = [float(s["vorp"]) for s in best if pd.notna(s.get("vorp"))]
    peak_vorp = round(float(sum(vorp_present)) * PEAK_WINDOW / len(best), 2) if len(vorp_present) == len(best) else None
    return {
        "peak_bpm": round(float(sum(s["bpm"] for s in best) / len(best)), 4),
        "peak_mpg": round(total_mp / total_g, 2),
        "peak_vorp": peak_vorp,
        "career_win_shares": round(career_ws, 2),
        "seasons_played": seasons_played,
        "games_played": games_played,
        "last_season_end": max(s["season_end"] for s in played),
        "peak_window_seasons": best_window_size,
    }


def population_anchors(rows: Sequence[Mapping[str, Any]]) -> ComponentAnchors:
    bpm_values: list[float] = []
    mpg_values: list[float] = []
    vorp_values: list[float] = []
    for row in rows:
        bpm, mpg, games = row.get("peak_bpm"), row.get("peak_mpg"), row.get("games_played")
        if bpm is None or mpg is None or games is None or games < MIN_CAREER_GAMES:
            continue
        bpm_values.append(float(bpm))
        mpg_values.append(float(mpg))
        vorp = row.get("peak_vorp")
        if vorp is not None:
            vorp_values.append(float(vorp))
    if not bpm_values or not vorp_values:
        raise ValueError("no qualified rows to compute population anchors")
    bpm_mean, bpm_std = statistics.fmean(bpm_values), statistics.pstdev(bpm_values)
    vorp_mean, vorp_std = statistics.fmean(vorp_values), statistics.pstdev(vorp_values)
    mpg_mean, mpg_std = statistics.fmean(mpg_values), statistics.pstdev(mpg_values)
    if min(bpm_std, vorp_std, mpg_std) <= 0:
        raise ValueError("component standard deviation is zero; cannot standardize labels")
    return ComponentAnchors(
        round(bpm_mean, 4),
        round(bpm_std, 4),
        round(vorp_mean, 4),
        round(vorp_std, 4),
        round(mpg_mean, 4),
        round(mpg_std, 4),
    )


def calculate_composite_z(
    peak_bpm: float | None,
    peak_mpg: float | None,
    peak_vorp: float | None,
    games_played: int | float | None,
    anchors: ComponentAnchors,
) -> float | None:
    if peak_bpm is None or peak_mpg is None or games_played is None or games_played < MIN_CAREER_GAMES:
        return None
    z_bpm = (peak_bpm - anchors.bpm_mean) / anchors.bpm_std
    z_mpg = (peak_mpg - anchors.mpg_mean) / anchors.mpg_std
    vorp_val = anchors.vorp_mean if peak_vorp is None else peak_vorp
    z_vorp = (vorp_val - anchors.vorp_mean) / anchors.vorp_std
    return round((0.40 * z_bpm) + (0.30 * z_vorp) + (0.30 * z_mpg), 4)


def derive_sentinel(qualified_composites: Sequence[float]) -> float:
    return round(float(min(qualified_composites)) - SENTINEL_MARGIN, 4)


def label_outcomes(outcome_rows: list[dict]) -> None:
    anchors = population_anchors(outcome_rows)
    for row in outcome_rows:
        row["composite_z"] = calculate_composite_z(
            peak_bpm=row["peak_bpm"],
            peak_mpg=row["peak_mpg"],
            peak_vorp=row["peak_vorp"],
            games_played=row["games_played"],
            anchors=anchors,
        )
    composites = [row["composite_z"] for row in outcome_rows if row["composite_z"] is not None]
    boundaries = [round(float(value), 4) for value in np.percentile(composites, TIER_PERCENTILES)]
    sentinel = derive_sentinel(composites)
    for row in outcome_rows:
        tier, tier_label = calculate_outcome_tier(row["composite_z"], boundaries)
        row["career_tier"] = tier
        row["tier_label"] = tier_label
    spec = {
        "component_anchors": anchors._asdict(),
        "tier_boundaries": boundaries,
        "tier_percentiles": list(TIER_PERCENTILES),
        "vorp_window_years": PEAK_WINDOW,
        "sentinel_composite_z": sentinel,
        "sentinel_margin": SENTINEL_MARGIN,
    }
    LABEL_SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Label spec written to {LABEL_SPEC_PATH}: boundaries={boundaries} sentinel={sentinel}")
