import logging
import math
from collections.abc import Sequence
from typing import Any, NamedTuple
from uuid import UUID

import numpy as np
import pandas as pd
import xgboost as xgb

from arc.db import Player
from arc.ml.feature_engineering import FEATURE_COLUMNS, engineer_features, player_feature_row
from arc.ml.tiers import TIER_NAMES, calculate_outcome_tier, calculate_tier_probabilities
from arc.schemas.prediction import (
    ModelCard,
    PredictionResponse,
    ShapFactor,
    SimulatorInput,
    TierProbability,
)
from arc.services.cache import read_cached, write_cached

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Machine learning model '{model_name}' is not loaded or unavailable.")


class Projection(NamedTuple):
    projected_z: float
    tier_probabilities: np.ndarray
    projected_tier: int
    projected_tier_label: str
    star_probability: float
    bust_risk: float


class PredictionService:
    def __init__(self) -> None:
        self.regressor: Any = None
        self.residuals: np.ndarray = np.array([])
        self.calibration_sigma: float = 0.0
        self.tier_boundaries: list[float] = []
        self.model_card: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self.regressor is not None

    @property
    def model_version(self) -> str:
        return str(self.model_card.get("model_version") or "unknown")

    @property
    def cache_namespace(self) -> str:
        fingerprint = str(self.model_card.get("dataset_fingerprint") or "")[:16]
        return f"{fingerprint}-{self.model_version}".strip("-")

    def load(self, bundle: dict) -> None:
        try:
            regressor = bundle["regressor"]
            residuals = np.asarray(bundle["residuals"], dtype=float)
            boundaries = [float(value) for value in bundle["model_card"]["label_spec"]["tier_boundaries"]]
        except (KeyError, TypeError, ValueError):
            logger.exception("Model bundle is missing the prediction contract; predictions are unavailable.")
            return
        sigma = float(np.std(residuals)) if residuals.size >= 2 else 0.0
        if not 0 < sigma < math.inf:
            logger.warning("Bundle residuals cannot calibrate the tiers; predictions are unavailable.")
            return
        self.residuals = residuals
        self.calibration_sigma = sigma
        self.tier_boundaries = boundaries
        self.model_card = bundle["model_card"]
        self.regressor = regressor
        logger.info("PredictionService: model bundle loaded (regressor, calibration, model card).")

    def get_or_predict_for_prospect(self, player: Player) -> PredictionResponse:
        key = f"nba:pred:{player.id}"
        cached = read_cached(key, PredictionResponse)
        if cached is not None:
            return cached
        prediction = self._predict_row(player_feature_row(player), prospect_id=player.id, player_name=player.name)
        write_cached(key, prediction)
        return prediction

    def predict_stat_line(self, stat_line: SimulatorInput) -> PredictionResponse:
        return self._predict_row(stat_line.model_dump(), prospect_id=None, player_name=stat_line.player_name)

    def predict_batch(self, players: Sequence[Player]) -> list[Projection]:
        if not players:
            return []
        x, _ = self._features(pd.DataFrame([player_feature_row(p) for p in players]))
        return [self._project(float(projected_z)) for projected_z in self.regressor.predict(x)]

    def get_model_card(self) -> ModelCard:
        self._ensure_loaded()
        return ModelCard.model_validate(self.model_card)

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelUnavailableError("XGBoost Prospect Regressor")

    def _features(self, raw_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        self._ensure_loaded()
        processed = engineer_features(raw_df)
        return processed[FEATURE_COLUMNS].to_numpy(dtype=float), processed

    def _project(self, projected_z: float) -> Projection:
        probs: np.ndarray = np.asarray(
            calculate_tier_probabilities(projected_z, self.residuals, boundaries=self.tier_boundaries)
        )
        projected_tier, projected_tier_label = calculate_outcome_tier(projected_z, self.tier_boundaries)
        return Projection(
            projected_z=projected_z,
            tier_probabilities=probs,
            projected_tier=projected_tier,
            projected_tier_label=projected_tier_label,
            star_probability=round(float(probs[0] + probs[1]), 4),
            bust_risk=round(float(probs[4]), 4),
        )

    def _predict_row(
        self,
        feature_row: dict,
        prospect_id: UUID | None,
        player_name: str,
    ) -> PredictionResponse:
        x, processed_df = self._features(pd.DataFrame([feature_row]))
        projected_z = float(self.regressor.predict(x)[0])
        projection = self._project(projected_z)
        tier_probabilities = [
            TierProbability(
                tier=index + 1,
                tier_label=TIER_NAMES[index + 1],
                probability=round(float(probability), 4),
            )
            for index, probability in enumerate(projection.tier_probabilities)
        ]
        pos_factors, neg_factors = self._compute_shap_factors(x[0], processed_df.iloc[0])
        return PredictionResponse(
            prospect_id=prospect_id,
            player_name=player_name,
            projected_tier=projection.projected_tier,
            projected_tier_label=projection.projected_tier_label,
            projected_z=round(projected_z, 3),
            tier_probabilities=tier_probabilities,
            top_positive_factors=pos_factors,
            top_negative_factors=neg_factors,
            model_version=self.model_version,
        )

    def _compute_shap_factors(
        self,
        sample: np.ndarray,
        raw_sample: pd.Series,
    ) -> tuple[list[ShapFactor], list[ShapFactor]]:
        dmatrix = xgb.DMatrix(sample.reshape(1, -1), feature_names=FEATURE_COLUMNS)
        contributions = self.regressor.get_booster().predict(dmatrix, pred_contribs=True)[0, :-1]
        factors = [
            (name, float(raw_sample[name]), float(contributions[index])) for index, name in enumerate(FEATURE_COLUMNS)
        ]
        strongest_positive = sorted((f for f in factors if f[2] > 0), key=lambda f: f[2], reverse=True)[:3]
        strongest_negative = sorted((f for f in factors if f[2] < 0), key=lambda f: f[2])[:3]

        def make_factor(factor: tuple[str, float, float]) -> ShapFactor:
            name, raw_value, shap_value = factor
            return ShapFactor(
                feature_name=name,
                feature_value=round(raw_value, 3),
                shap_value=round(shap_value, 4),
                contribution_sigma=f"{shap_value / self.calibration_sigma:+.2f}σ",
            )

        return ([make_factor(f) for f in strongest_positive], [make_factor(f) for f in strongest_negative])


prediction_service = PredictionService()
