from typing import Any

import numpy as np
import pandas as pd

PER_DRAFT_CUTOFFS = {"top_5": 5, "top_10": 10, "lottery": 14, "first_round": 30}


def spearman(first, second) -> float:
    return float(pd.Series(np.asarray(first, dtype=float)).corr(pd.Series(np.asarray(second, dtype=float)), "spearman"))


def top_k_star_precision(y_true, star_probs, k: int) -> float | None:
    is_star = np.isin(np.asarray(y_true), (0, 1))
    if is_star.sum() == 0:
        return None
    order = np.argsort(-np.asarray(star_probs))[:k]
    return round(float(is_star[order].mean()), 4)


def _pick_order_scores(picks) -> np.ndarray:
    values = np.asarray(picks, dtype=float)
    return np.where(np.isnan(values), -np.inf, -values)


def draft_order_baseline(picks, y_true, y_true_z) -> dict[str, float | None]:
    scores = _pick_order_scores(picks)
    return {
        "spearman_rank_correlation": spearman(scores, y_true_z),
        **{f"star_top{k}_precision": top_k_star_precision(y_true, scores, k=k) for k in (10, 25)},
    }


def per_draft_star_hits(y_true, scores, draft_years, k: int) -> int:
    is_star = np.isin(np.asarray(y_true), (0, 1))
    values, years = np.asarray(scores, dtype=float), np.asarray(draft_years)
    hits = 0
    for year in np.unique(years):
        members = np.flatnonzero(years == year)
        hits += int(is_star[members[np.argsort(-values[members], kind="stable")[:k]]].sum())
    return hits


def per_draft_comparison(y_true, projected_z, picks, draft_years) -> dict[str, Any]:
    draft_scores = _pick_order_scores(picks)
    return {
        "stars": int(np.isin(np.asarray(y_true), (0, 1)).sum()),
        **{
            name: {
                "model": per_draft_star_hits(y_true, projected_z, draft_years, k),
                "draft_order": per_draft_star_hits(y_true, draft_scores, draft_years, k),
            }
            for name, k in PER_DRAFT_CUTOFFS.items()
        },
    }
