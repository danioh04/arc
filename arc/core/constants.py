RATE_STAT_DEFAULTS: dict[str, float] = {
    "min_pct": 50.0,
    "usg_pct": 20.0,
    "ts_pct": 0.52,
    "fg3a_per_fga": 0.30,
    "orb_pct": 4.0,
    "drb_pct": 12.0,
    "ast_pct": 12.0,
    "to_pct": 15.0,
    "blk_pct": 1.5,
    "stl_pct": 1.5,
    "ftr": 30.0,
    "ft_pct": 0.70,
    "obpm": 0.0,
    "dbpm": 0.0,
}

CLASS_YEAR_DRAFT_AGE = {"Fr": 19.5, "So": 20.5, "Jr": 21.5, "Sr": 22.5, "Gr": 23.5}
DEFAULT_DRAFT_AGE = 20.0
MIN_DRAFT_AGE = 15.0
MAX_DRAFT_AGE = 30.0

HEIGHT_BASELINES = {
    "G": 74.0,
    "W": 78.0,
    "F": 80.0,
    "C": 82.0,
}
DEFAULT_POSITION = "W"

POSITION_CODES = {
    "G": "G",
    "PG": "G",
    "SG": "G",
    "GUARD": "G",
    "W": "W",
    "SF": "W",
    "WING": "W",
    "F": "F",
    "PF": "F",
    "FORWARD": "F",
    "C": "C",
    "CENTER": "C",
    "BIG": "C",
}


def normalize_position(value: object) -> str | None:
    return POSITION_CODES.get(str(value).strip().upper())


def require_position(value: str) -> str:
    normalized = normalize_position(value)
    if normalized is None:
        aliases = ", ".join(sorted(alias for alias, code in POSITION_CODES.items() if alias != code))
        raise ValueError(f"Unknown position {value!r}. Use G, W, F or C (aliases: {aliases}).")
    return normalized
