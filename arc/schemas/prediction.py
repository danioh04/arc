from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from arc.core.constants import RATE_STAT_DEFAULTS, require_position


class ShapFactor(BaseModel):
    feature_name: str = Field(..., description="Name of the model feature")
    feature_value: float = Field(..., description="Prospect's actual value for this feature")
    shap_value: float = Field(..., description="TreeSHAP contribution to the projected score")
    contribution_sigma: str = Field(
        ..., description="shap_value in units of the model's residual standard deviation (e.g. +0.14σ)"
    )


class TierProbability(BaseModel):
    tier: int = Field(..., ge=1, le=5, description="Outcome Tier (1 to 5)")
    tier_label: str = Field(..., description="Tier name (e.g. Franchise Star)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Model-predicted probability")


class PredictionResponse(BaseModel):
    prospect_id: UUID | None = Field(None, description="Prospect UUID (if existing database player)")
    player_name: str = Field(..., description="Player name")
    projected_tier: int = Field(..., ge=1, le=5, description="Projected tier from the point estimate (1 to 5)")
    projected_tier_label: str = Field(..., description="Label of the projected tier")
    projected_z: float = Field(..., description="Raw continuous standardized Composite Z-Score")
    tier_probabilities: list[TierProbability] = Field(..., description="Probabilities across all 5 tiers")
    top_positive_factors: list[ShapFactor] = Field(default_factory=list, description="Top positive drivers (TreeSHAP)")
    top_negative_factors: list[ShapFactor] = Field(default_factory=list, description="Top risk factors (TreeSHAP)")
    model_version: str = Field(..., description="Active model version identifier")


class PlayerComp(BaseModel):
    player_name: str = Field(..., description="Historical comparison player name")
    draft_year: int | None = Field(None, description="Year player was drafted")
    overall_pick: int | None = Field(None, description="Pick number drafted at")
    college_team: str = Field(..., description="College alma mater")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    actual_tier: int = Field(..., ge=1, le=5, description="Actual mature career outcome tier (1 to 5)")
    actual_tier_label: str = Field(..., description="Actual career outcome label")
    actual_peak_bpm: float | None = Field(None, description="Actual peak NBA BPM")
    actual_peak_mpg: float | None = Field(None, description="Actual peak NBA minutes per game")


class SimilarityResponse(BaseModel):
    prospect_id: UUID = Field(..., description="Prospect UUID")
    player_name: str = Field(..., description="Player name")
    comparisons: list[PlayerComp] = Field(..., description="Top closest historical college statistical comps")


class SimulatorInput(BaseModel):
    player_name: str = Field("Custom Prospect", description="Optional prospect name")
    primary_position: str = Field("G", description="Primary position: G, W, F, C")
    height_inches: int = Field(77, ge=65, le=92, description="Height in inches")
    draft_age: float | None = Field(None, ge=17.0, le=26.0, description="Age on draft night")
    class_year: Literal["Fr", "So", "Jr", "Sr", "Gr"] = Field("Fr", description="Class year")
    min_pct: float = Field(RATE_STAT_DEFAULTS["min_pct"], ge=10.0, le=100.0, description="Minutes percentage")
    usg_pct: float = Field(RATE_STAT_DEFAULTS["usg_pct"], ge=5.0, le=60.0, description="Usage percentage")
    ts_pct: float = Field(
        RATE_STAT_DEFAULTS["ts_pct"], ge=0.30, le=0.85, description="True shooting percentage (e.g. 0.55)"
    )
    fg3a_per_fga: float = Field(
        RATE_STAT_DEFAULTS["fg3a_per_fga"], ge=0.0, le=1.0, description="3-point attempt rate (3PA / FGA)"
    )
    orb_pct: float = Field(RATE_STAT_DEFAULTS["orb_pct"], ge=0.0, le=30.0, description="Offensive rebound percentage")
    drb_pct: float = Field(RATE_STAT_DEFAULTS["drb_pct"], ge=0.0, le=45.0, description="Defensive rebound percentage")
    ast_pct: float = Field(RATE_STAT_DEFAULTS["ast_pct"], ge=0.0, le=60.0, description="Assist percentage")
    to_pct: float = Field(RATE_STAT_DEFAULTS["to_pct"], ge=1.0, le=40.0, description="Turnover percentage")
    blk_pct: float = Field(RATE_STAT_DEFAULTS["blk_pct"], ge=0.0, le=25.0, description="Block percentage")
    stl_pct: float = Field(RATE_STAT_DEFAULTS["stl_pct"], ge=0.0, le=10.0, description="Steal percentage")
    ftr: float = Field(RATE_STAT_DEFAULTS["ftr"], ge=5.0, le=150.0, description="Free throw rate (FTA/FGA * 100)")
    ft_pct: float = Field(
        RATE_STAT_DEFAULTS["ft_pct"], ge=0.30, le=1.0, description="Free throw percentage (e.g. 0.70)"
    )
    obpm: float = Field(RATE_STAT_DEFAULTS["obpm"], ge=-10.0, le=30.0, description="Offensive BPM")
    dbpm: float = Field(RATE_STAT_DEFAULTS["dbpm"], ge=-10.0, le=15.0, description="Defensive BPM")

    @field_validator("primary_position")
    @classmethod
    def _normalize_primary_position(cls, value: str) -> str:
        return require_position(value)


class SimulatorResponse(BaseModel):
    prediction: PredictionResponse = Field(..., description="Projected outcome and factor cards")
    comparisons: list[PlayerComp] = Field(..., description="Top closest historical college comps")


class ModelCard(BaseModel):
    model_name: str = Field(..., description="Model architecture name")
    model_version: str = Field(..., description="Model version tag")
    trained_at: str = Field(..., description="UTC timestamp of the training run")
    dataset_fingerprint: str = Field(..., description="SHA-256 of the dataset the model was trained on")
    library_versions: dict[str, str] = Field(..., description="Versions of the libraries that score the model")
    training_rows: int = Field(..., description="Rows used to fit the regressor")
    holdout_rows: int = Field(..., description="Rows in the holdout evaluation")
    feature_names: list[str] = Field(..., description="Feature column names in model order")
    calibration_sigma: float = Field(
        ..., description="Spread of the out-of-fold residuals; tier probabilities use the full residual distribution"
    )
    label_spec: dict = Field(
        ...,
        description="Label normalization anchors, quantile boundaries and sentinel from the labelling run",
    )
    tier_definitions: dict[str, str] = Field(..., description="Definitions of the 5 career tiers")
    evaluation: dict = Field(..., description="Holdout metrics, ranking metrics and baselines")
    similarity: dict = Field(..., description="Similarity index metadata (size, covered draft years)")
