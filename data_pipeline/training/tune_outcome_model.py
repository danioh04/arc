import argparse
import logging
import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from arc.ml.feature_engineering import FEATURE_COLUMNS, engineer_features
from data_pipeline.config import LABELLED_PATH
from data_pipeline.training.evaluation import (
    PER_DRAFT_CUTOFFS,
    per_draft_comparison,
    spearman,
    top_k_star_precision,
)
from data_pipeline.training.train_outcome_model import (
    FIXED_XGB_PARAMS,
    TUNED_XGB_PARAMS,
    build_estimator,
    load_label_spec,
    target_z,
)

SEARCH_SEED = 0
FOLD_SEED = 100
SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [2, 3, 4],
    "n_estimators": [50, 100, 150, 200, 300, 400, 600],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.05, 0.5, 1],
    "reg_lambda": [0.5, 1.2, 3, 10],
    "gamma": [0, 0.1, 0.5, 1],
}
LEARNING_RATE_LOG10_RANGE = (-2.0, -0.7)


def sample_candidates(count: int) -> list[dict[str, Any]]:
    rng = random.Random(SEARCH_SEED)
    candidates = []
    for _ in range(count):
        candidate = {"max_depth": rng.choice(SEARCH_SPACE["max_depth"])}
        candidate["learning_rate"] = round(10 ** rng.uniform(*LEARNING_RATE_LOG10_RANGE), 4)
        for name in list(SEARCH_SPACE)[1:]:
            candidate[name] = rng.choice(SEARCH_SPACE[name])
        candidates.append(candidate)
    return candidates


def cross_validated_spearman(params: dict[str, Any], x: np.ndarray, y: np.ndarray, repeats: int) -> float:
    scores = []
    for repeat in range(repeats):
        out_of_fold = np.zeros_like(y)
        for train_idx, valid_idx in KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED + repeat).split(x):
            model = build_estimator({**params, **FIXED_XGB_PARAMS})
            out_of_fold[valid_idx] = model.fit(x[train_idx], y[train_idx]).predict(x[valid_idx])
        scores.append(spearman(y, out_of_fold))
    return float(np.mean(scores))


def holdout_scores(params: dict[str, Any], seed: int, data: dict[str, Any]) -> dict[str, Any]:
    model = build_estimator({**params, **FIXED_XGB_PARAMS, "random_state": seed})
    projected_z = model.fit(data["x_train"], data["y_train"]).predict(data["x_holdout"])
    holdout = data["holdout"]
    return {
        "spearman": spearman(data["y_holdout"], projected_z),
        "star_top10_precision": top_k_star_precision(data["classes"], projected_z, k=10),
        **per_draft_comparison(data["classes"], projected_z, holdout["overall_pick"], holdout["cohort_year"]),
    }


def describe(values: list[Any]) -> str:
    return ", ".join(f"{value} ({count}x)" for value, count in sorted(Counter(values).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=int, default=200, help="Random settings to try")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated 5-fold splits per candidate")
    parser.add_argument("--stability", type=int, default=0, help="Seeds for the read-only holdout report")
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    label_spec = load_label_spec()
    sentinel = float(label_spec["sentinel_composite_z"])
    dataset = pd.read_csv(LABELLED_PATH)
    train_df, holdout_df = dataset[dataset.split == "train"], dataset[dataset.split == "holdout"]
    x_train = engineer_features(train_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = target_z(train_df, sentinel)
    candidates = sample_candidates(args.candidates)
    print(f"Scoring {len(candidates)} candidates on {len(train_df)} training rows (holdout untouched)...")
    scores = Parallel(n_jobs=-1)(
        delayed(cross_validated_spearman)(params, x_train, y_train, args.repeats) for params in candidates
    )
    ranked = sorted(zip(scores, candidates, strict=True), key=lambda item: -item[0])
    current = cross_validated_spearman(TUNED_XGB_PARAMS, x_train, y_train, args.repeats)
    print(f"\nCurrent TUNED_XGB_PARAMS: CV Spearman {current:.4f}")
    print("Best candidates:")
    for score, params in ranked[:5]:
        print(f"  {score:.4f}  {params}")
    best = ranked[0][1]
    print(f"\nPaste into TUNED_XGB_PARAMS:\n{best}")
    if args.stability:
        data = {
            "holdout": holdout_df,
            "x_train": x_train,
            "y_train": y_train,
            "x_holdout": engineer_features(holdout_df)[FEATURE_COLUMNS].to_numpy(dtype=float),
            "y_holdout": target_z(holdout_df, sentinel),
            "classes": (holdout_df["career_tier"] - 1).astype(int).to_numpy(),
        }
        runs = Parallel(n_jobs=-1)(delayed(holdout_scores)(best, seed, data) for seed in range(args.stability))
        print(f"\nHoldout across {args.stability} seeds (report only, {runs[0]['stars']} stars):")
        print(f"  Spearman median {np.median([run['spearman'] for run in runs]):.4f}")
        print(f"  Pooled top 10 star precision: {describe([run['star_top10_precision'] for run in runs])}")
        for name in PER_DRAFT_CUTOFFS:
            draft_order = runs[0][name]["draft_order"]
            print(f"  {name:<12} model {describe([run[name]['model'] for run in runs]):<24} draft order {draft_order}")


if __name__ == "__main__":
    main()
