import logging
import os
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

BUNDLE_NAME = "model_bundle.joblib"


def bundle_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / BUNDLE_NAME


def read_bundle(artifacts_dir: Path) -> dict[str, Any]:
    path = bundle_path(artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `python -m data_pipeline.run_pipeline` to build it."
        )
    return joblib.load(path)


def write_bundle(artifacts_dir: Path, sections: dict[str, Any]) -> None:
    path = bundle_path(artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(sections, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info(f"Model bundle written with {sorted(sections)} -> {path}")
