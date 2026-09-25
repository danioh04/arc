import logging
from typing import Any, NamedTuple

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from arc.ml.feature_engineering import SIMILARITY_COLUMNS, engineer_features

logger = logging.getLogger(__name__)


class SimilarityIndex(NamedTuple):
    neighbors: NearestNeighbors
    scaler: StandardScaler
    records: list[dict[str, Any]]
    player_ids: list[str]
    info: dict[str, Any]


def build_similarity_index(historical_df: pd.DataFrame) -> SimilarityIndex:
    logger.info(f"Building player similarity index with {len(historical_df)} historical players...")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(engineer_features(historical_df)[SIMILARITY_COLUMNS])
    neighbors = NearestNeighbors(metric="cosine")
    neighbors.fit(x_scaled)
    rows = historical_df.to_dict("records")
    records = [
        {
            "player_name": row["player_name"],
            "draft_year": int(row["draft_year"]) if pd.notnull(row["draft_year"]) else None,
            "overall_pick": int(row["overall_pick"]) if pd.notnull(row["overall_pick"]) else None,
            "college_team": row["college_team"],
            "actual_tier": int(row["career_tier"]),
            "actual_tier_label": row["tier_label"],
            "actual_peak_bpm": float(row["peak_bpm"]) if pd.notnull(row["peak_bpm"]) else None,
            "actual_peak_mpg": float(row["peak_mpg"]) if pd.notnull(row["peak_mpg"]) else None,
        }
        for row in rows
    ]
    player_ids = [str(row["player_id"]) for row in rows]
    logger.info(f"Built player similarity index ({len(records)} players)")
    return SimilarityIndex(
        neighbors=neighbors,
        scaler=scaler,
        records=records,
        player_ids=player_ids,
        info={
            "index_size": len(records),
            "draft_year_range": [
                int(historical_df["draft_year"].min()),
                int(historical_df["draft_year"].max()),
            ],
            "metric": "cosine",
        },
    )
