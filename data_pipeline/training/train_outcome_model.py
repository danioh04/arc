import json
import logging
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from arc.ml.feature_engineering import FEATURE_COLUMNS, engineer_features
from arc.ml.tiers import calculate_tier_probabilities
from data_pipeline.config import IDENTITY_EXCLUSIONS_PATH, LABEL_SPEC_PATH
from data_pipeline.training.evaluation import draft_order_baseline, spearman, top_k_star_precision

logger = logging.getLogger(__name__)

MONOTONE_DIRECTIONS = {
    "draft_age": -1,
    "positional_size_delta": 1,
    "to_pct": -1,
    "ts_pct": 1,
    "ft_pct": 1,
    "stl_pct": 1,
    "blk_pct": 1,
    "obpm": 1,
    "dbpm": 1,
}


XGB_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.13,
    "n_estimators": 90,
    "min_child_weight": 5,
    "subsample": 1.0,
    "colsample_bytree": 0.60,
    "reg_alpha": 0.05,
    "reg_lambda": 1.20,
    "gamma": 0.10,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
}


class OutcomeModel(NamedTuple):
    regressor: XGBRegressor
    residuals: list[float]
    sigma: float
    metrics: dict[str, Any]


def load_label_spec() -> dict:
    if not LABEL_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Label spec not found at {LABEL_SPEC_PATH}. Re-run `python -m data_pipeline.build_labelled_dataset`."
        )
    return json.loads(LABEL_SPEC_PATH.read_text(encoding="utf-8"))


def _build_estimator() -> XGBRegressor:
    return XGBRegressor(
        **XGB_PARAMS,
        monotone_constraints=tuple(MONOTONE_DIRECTIONS.get(column, 0) for column in FEATURE_COLUMNS),
    )


def _target_z(df: pd.DataFrame, sentinel: float) -> np.ndarray:
    return df["composite_z"].fillna(sentinel).to_numpy(dtype=float)


def _out_of_fold_predictions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    out_of_fold = np.zeros_like(y, dtype=float)
    for train_idx, valid_idx in folds.split(x):
        fold_model = _build_estimator()
        fold_model.fit(x[train_idx], y[train_idx])
        out_of_fold[valid_idx] = fold_model.predict(x[valid_idx])
    return out_of_fold


def _exclusion_summary() -> dict:
    exclusions = pd.read_csv(IDENTITY_EXCLUSIONS_PATH)
    by_reason = exclusions["reason"].value_counts()
    return {"count": len(exclusions), "by_reason": {str(key): int(value) for key, value in by_reason.items()}}


def train_outcome_model(train_df: pd.DataFrame, holdout_df: pd.DataFrame, label_spec: dict) -> OutcomeModel:
    boundaries = [float(value) for value in label_spec["tier_boundaries"]]
    sentinel = float(label_spec["sentinel_composite_z"])
    x_train = engineer_features(train_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = _target_z(train_df, sentinel)
    out_of_fold = _out_of_fold_predictions(x_train, y_train)
    cv_spearman = spearman(y_train, out_of_fold)
    residuals = y_train - out_of_fold
    sigma = float(np.std(residuals))
    residuals_sorted = [float(value) for value in np.sort(residuals)]
    logger.info(f"Cross-validated Spearman {cv_spearman:.4f}; residual sigma {sigma:.4f}")
    model = _build_estimator()
    model.fit(x_train, y_train)
    x_holdout = engineer_features(holdout_df)[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_holdout_z = _target_z(holdout_df, sentinel)
    y_holdout_class = (holdout_df["career_tier"] - 1).astype(int).to_numpy()
    projected_z = model.predict(x_holdout)
    error = y_holdout_z - projected_z
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt((error**2).mean()))
    r2 = float(1.0 - (error**2).sum() / ((y_holdout_z - y_holdout_z.mean()) ** 2).sum())
    spearman_corr = spearman(y_holdout_z, projected_z)
    prob_matrix = np.array([calculate_tier_probabilities(float(z), residuals_sorted, boundaries) for z in projected_z])
    star_probs = prob_matrix[:, 0] + prob_matrix[:, 1]
    peak_windows = train_df["peak_window_seasons"].value_counts().sort_index()
    metrics = {
        "hyperparameters": XGB_PARAMS,
        "cv_spearman": round(cv_spearman, 4),
        "peak_window_distribution": {str(int(k)): int(v) for k, v in peak_windows.items()},
        "identity_exclusions": _exclusion_summary(),
        "holdout_r2": r2,
        "holdout_mae": mae,
        "holdout_rmse": rmse,
        "spearman_rank_correlation": spearman_corr,
        "star_base_rate": round(float(np.isin(y_holdout_class, (0, 1)).mean()), 4),
        "ranking": {
            "star_top10_precision": top_k_star_precision(y_holdout_class, star_probs, k=10),
            "star_top25_precision": top_k_star_precision(y_holdout_class, star_probs, k=25),
        },
        "baselines": {
            "draft_market_order": draft_order_baseline(holdout_df["overall_pick"], y_holdout_class, y_holdout_z),
        },
    }
    logger.info(f"Backtest Evaluation - Continuous R2: {r2:.4f}, Spearman Corr: {spearman_corr:.4f}")
    return OutcomeModel(regressor=model, residuals=residuals_sorted, sigma=sigma, metrics=metrics)
