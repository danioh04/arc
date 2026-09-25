import hashlib
import importlib.metadata as package_metadata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc.ml.feature_engineering import FEATURE_COLUMNS
from arc.ml.tiers import tier_definitions
from data_pipeline.config import MODEL_VERSION

TRACKED_LIBRARIES = ("scikit-learn", "xgboost", "numpy", "pandas", "joblib", "scipy")


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def dataset_fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.name.encode())
    digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    return {name: package_metadata.version(name) for name in TRACKED_LIBRARIES}


def build_model_card(
    *,
    metrics: dict[str, Any],
    sigma: float,
    training_rows: int,
    holdout_rows: int,
    provenance: dict[str, Any],
    label_spec: dict[str, Any],
    similarity_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": "XGBoost Monotone Regressor",
        "model_version": MODEL_VERSION,
        "trained_at": provenance["trained_at"],
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "library_versions": provenance["library_versions"],
        "training_rows": training_rows,
        "holdout_rows": holdout_rows,
        "feature_names": list(FEATURE_COLUMNS),
        "calibration_sigma": round(float(sigma), 6),
        "tier_definitions": tier_definitions(label_spec["tier_boundaries"]),
        "label_spec": label_spec,
        "evaluation": metrics,
        "similarity": similarity_info,
    }
