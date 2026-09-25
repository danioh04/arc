import logging
from typing import Any

import numpy as np
import pandas as pd

from arc.db import Player
from arc.ml.feature_engineering import SIMILARITY_COLUMNS, engineer_features, player_feature_row
from arc.schemas.prediction import PlayerComp, SimilarityResponse, SimulatorInput
from arc.services.cache import read_cached, write_cached
from arc.services.prediction import ModelUnavailableError

logger = logging.getLogger(__name__)

COMPS_CACHE_K = 10


class SimilarityService:
    def __init__(self) -> None:
        self.neighbors: Any = None
        self.scaler: Any = None
        self.records: list[dict] = []
        self.player_ids: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.neighbors is not None

    def load(self, bundle: dict) -> None:
        try:
            neighbors = bundle["similarity_nn"]
            scaler = bundle["similarity_scaler"]
            records = list(bundle["similarity_records"])
            player_ids = list(bundle["similarity_ids"])
        except (KeyError, TypeError):
            logger.exception("Model bundle is missing the similarity contract; comps are unavailable.")
            return
        self.scaler = scaler
        self.records = records
        self.player_ids = player_ids
        self.neighbors = neighbors
        logger.info(f"SimilarityService: k-NN index loaded from the bundle ({len(records)} players).")

    def get_or_find_comps_for_prospect(self, player: Player, k: int) -> SimilarityResponse:
        key = f"nba:comps:{player.id}"
        cached = read_cached(key, SimilarityResponse)
        if cached is not None:
            return cached.model_copy(update={"comparisons": cached.comparisons[:k]})
        comps = self._find_comps(player_feature_row(player), k=COMPS_CACHE_K)
        response = SimilarityResponse(prospect_id=player.id, player_name=player.name, comparisons=comps)
        write_cached(key, response)
        return response.model_copy(update={"comparisons": comps[:k]})

    def find_comps_for_stat_line(self, stat_line: SimulatorInput, k: int) -> list[PlayerComp]:
        return self._find_comps(stat_line.model_dump(), k=k)

    def _find_comps(self, feature_row: dict, k: int) -> list[PlayerComp]:
        if not self.is_loaded:
            raise ModelUnavailableError("Similarity k-NN Model")
        processed = engineer_features(pd.DataFrame([feature_row]))
        x_scaled = self.scaler.transform(processed[SIMILARITY_COLUMNS])
        distances, indices = self.neighbors.kneighbors(x_scaled, n_neighbors=min(len(self.records), k + 1))
        comps: list[PlayerComp] = []
        query_id = str(feature_row.get("player_id") or "")
        for distance, index in zip(distances[0], indices[0], strict=False):
            row = int(index)
            if query_id and self.player_ids[row] == query_id:
                continue
            similarity = float(np.clip(1.0 - distance, 0.0, 1.0))
            comps.append(PlayerComp(**self.records[row], similarity_score=round(similarity, 3)))
            if len(comps) >= k:
                break
        return comps


similarity_service = SimilarityService()
