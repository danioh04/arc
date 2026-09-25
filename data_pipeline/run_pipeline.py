import argparse
import logging

import pandas as pd

from arc.core.config import ARTIFACTS_DIR
from arc.ml.bundle import write_bundle
from data_pipeline import build_labelled_dataset
from data_pipeline.config import LABELLED_PATH
from data_pipeline.seed_database import seed_database
from data_pipeline.training.model_card import (
    build_model_card,
    dataset_fingerprint,
    library_versions,
    utc_timestamp,
)
from data_pipeline.training.train_outcome_model import load_label_spec, train_outcome_model
from data_pipeline.training.train_similarity import build_similarity_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def load_labelled_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(LABELLED_PATH)
    logger.info(f"Labelled dataset loaded: {len(dataset)} rows | {dataset.groupby('split').size().to_dict()}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arc data pipeline: labels -> database -> model artifacts.")
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Reuse data/generated/*.csv instead of re-scraping Basketball Reference.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Do not rebuild data/nba_prospects.db (that stage drops every table).",
    )
    parser.add_argument("--skip-train", action="store_true", help="Do not retrain or export model artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== Pipeline Execution ===")
    logger.info(
        f"Stages: labels={'skip' if args.skip_labels else 'run'} "
        f"database={'skip' if args.skip_db else 'run'} train={'skip' if args.skip_train else 'run'}"
    )
    if not args.skip_labels:
        build_labelled_dataset.main()
    if not args.skip_db:
        seed_database()
    if args.skip_train:
        logger.info("=== Pipeline Complete (training skipped). ===")
        return
    logger.info(f"Loading labelled dataset ({LABELLED_PATH})...")
    dataset = load_labelled_dataset()
    train_df = dataset[dataset.split == "train"]
    holdout_df = dataset[dataset.split == "holdout"]
    logger.info(f"Splits: train={len(train_df)} | holdout={len(holdout_df)}")
    label_spec = load_label_spec()
    outcome = train_outcome_model(train_df, holdout_df, label_spec)
    logger.info(f"Benchmark Results: {outcome.metrics}")
    historical_all = dataset.copy()
    historical_all["draft_year"] = historical_all["cohort_year"].where(historical_all["overall_pick"].notna())
    similarity = build_similarity_index(historical_all)
    provenance = {
        "trained_at": utc_timestamp(),
        "dataset_fingerprint": dataset_fingerprint(LABELLED_PATH),
        "library_versions": library_versions(),
    }
    write_bundle(
        ARTIFACTS_DIR,
        {
            "regressor": outcome.regressor,
            "residuals": outcome.residuals,
            "similarity_nn": similarity.neighbors,
            "similarity_scaler": similarity.scaler,
            "similarity_records": similarity.records,
            "similarity_ids": similarity.player_ids,
            "model_card": build_model_card(
                metrics=outcome.metrics,
                sigma=outcome.sigma,
                training_rows=len(train_df),
                holdout_rows=len(holdout_df),
                provenance=provenance,
                label_spec=label_spec,
                similarity_info=similarity.info,
            ),
        },
    )
    logger.info("=== Pipeline Complete. All artifacts exported. ===")


if __name__ == "__main__":
    main()
