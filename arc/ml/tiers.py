from bisect import bisect_right
from collections.abc import Sequence

import numpy as np

MIN_CAREER_GAMES = 50

TIER_NAMES = {
    1: "Franchise Star",
    2: "All-Star",
    3: "Quality Starter",
    4: "Role Player",
    5: "End of Bench",
}


def tier_definitions(boundaries: Sequence[float]) -> dict[str, str]:
    t4_t5, t3_t4, t2_t3, t1_t2 = boundaries
    return {
        "Tier 1": f"Franchise Star (Composite Z >= {t1_t2:+.2f}; top ~4% of qualified careers)",
        "Tier 2": f"All-Star ({t2_t3:+.2f} <= Composite Z < {t1_t2:+.2f}; next ~6%)",
        "Tier 3": f"Quality Starter ({t3_t4:+.2f} <= Composite Z < {t2_t3:+.2f}; next ~15%)",
        "Tier 4": f"Role Player ({t4_t5:+.2f} <= Composite Z < {t3_t4:+.2f}; next ~25%)",
        "Tier 5": f"End of Bench (Composite Z < {t4_t5:+.2f} or <{MIN_CAREER_GAMES} GP; bottom ~50%)",
    }


def calculate_tier_probabilities(
    projected_z: float,
    residuals: Sequence[float] | np.ndarray,
    boundaries: Sequence[float],
) -> list[float]:
    residual_values = np.asarray(residuals)
    fractions = (
        np.searchsorted(residual_values, np.asarray(boundaries) - projected_z, side="right") / residual_values.size
    )
    t4_t5, t3_t4, t2_t3, t1_t2 = fractions
    return [1.0 - t1_t2, t1_t2 - t2_t3, t2_t3 - t3_t4, t3_t4 - t4_t5, float(t4_t5)]


def calculate_outcome_tier(composite_z: float | None, boundaries: Sequence[float]) -> tuple[int, str]:
    if composite_z is None:
        return 5, TIER_NAMES[5]
    tier = 5 - bisect_right(boundaries, composite_z)
    return tier, TIER_NAMES[tier]
