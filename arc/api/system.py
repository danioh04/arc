from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from arc.db import get_db
from arc.schemas.common import HealthCheck
from arc.schemas.prediction import ModelCard
from arc.services.cache import cache_is_healthy
from arc.services.prediction import prediction_service
from arc.services.similarity import similarity_service

router = APIRouter()


@router.get("/health", response_model=HealthCheck, summary="System health check probe")
def health_check(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    try:
        result = db.execute(text("SELECT 1 FROM players LIMIT 1"))
        db_connected = result.scalar() == 1
    except Exception:  # noqa: BLE001
        db_connected = False
    models_ready = prediction_service.is_loaded and similarity_service.is_loaded
    status = "healthy" if (db_connected and models_ready) else "degraded"
    response.status_code = 200 if status == "healthy" else 503
    return HealthCheck(
        status=status,
        database_connected=db_connected,
        cache_connected=cache_is_healthy(),
        models_loaded=models_ready,
        active_model_version=prediction_service.model_version,
    )


@router.get("/models/metadata", response_model=ModelCard, summary="Get active ML model metadata & schemas")
def get_model_metadata():
    return prediction_service.get_model_card()
