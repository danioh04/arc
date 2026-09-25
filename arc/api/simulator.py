from typing import Annotated

from fastapi import APIRouter, Query

from arc.schemas.prediction import SimulatorInput, SimulatorResponse
from arc.services.prediction import prediction_service
from arc.services.similarity import similarity_service

router = APIRouter()


@router.post(
    "/simulator",
    response_model=SimulatorResponse,
    summary="What-If Simulator: outcome prediction and historical comps",
)
def run_what_if_simulation(
    stat_line: SimulatorInput,
    k: Annotated[int, Query(ge=1, le=10, description="Number of historical comps to return")] = 5,
):
    prediction = prediction_service.predict_stat_line(stat_line)
    comparisons = similarity_service.find_comps_for_stat_line(stat_line, k=k)
    return SimulatorResponse(prediction=prediction, comparisons=comparisons)
