import numpy as np
import pandas as pd


def spearman(first, second) -> float:
    return float(pd.Series(np.asarray(first, dtype=float)).corr(pd.Series(np.asarray(second, dtype=float)), "spearman"))


def top_k_star_precision(y_true, star_probs, k: int) -> float | None:
    is_star = np.isin(np.asarray(y_true), (0, 1))
    if is_star.sum() == 0:
        return None
    order = np.argsort(-np.asarray(star_probs))[:k]
    return round(float(is_star[order].mean()), 4)


def draft_order_baseline(picks, y_true, y_true_z) -> dict[str, float | None]:
    values = np.asarray(picks, dtype=float)
    scores = np.where(np.isnan(values), -np.inf, -values)
    return {
        "spearman_rank_correlation": spearman(scores, y_true_z),
        **{f"star_top{k}_precision": top_k_star_precision(y_true, scores, k=k) for k in (10, 25)},
    }
