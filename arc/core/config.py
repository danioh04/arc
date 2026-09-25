import os
from pathlib import Path

IS_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("LAMBDA_TASK_ROOT"))
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
DB_PATH = ROOT / "data" / "nba_prospects.db"
DATABASE_URL = os.environ.get("DATABASE_URL") or (
    f"sqlite:///file:{DB_PATH.as_posix()}?uri=true&mode=ro&immutable=1"
    if IS_LAMBDA
    else f"sqlite:///{DB_PATH.as_posix()}"
)
REDIS_URL = os.environ.get("REDIS_URL", "")
