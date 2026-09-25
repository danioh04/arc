import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from mangum import Mangum

from arc.api import draft_classes, prospects, simulator, system
from arc.core import config
from arc.ml.bundle import read_bundle
from arc.services.cache import close_cache, init_cache, set_cache_namespace
from arc.services.prediction import ModelUnavailableError, prediction_service
from arc.services.similarity import similarity_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def warmup() -> None:
    init_cache()
    try:
        bundle = read_bundle(config.ARTIFACTS_DIR)
    except FileNotFoundError as error:
        logger.warning(f"{error} Predictions are unavailable until the pipeline has run.")
        bundle = None
    except Exception:
        logger.exception("Failed to read the model bundle")
        bundle = None
    if bundle is not None:
        prediction_service.load(bundle)
        similarity_service.load(bundle)
    set_cache_namespace(prediction_service.cache_namespace)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting Arc application.")
    warmup()
    yield
    logger.info("Closing Arc application connections.")
    close_cache()


app = FastAPI(
    title="Arc",
    description="### NBA Career Outcome Prediction Platform",
    openapi_url="/api/v1/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(_request: Request, error: ModelUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")


app.include_router(prospects.router, prefix="/api/v1/prospects", tags=["Prospects"])
app.include_router(draft_classes.router, prefix="/api/v1", tags=["Big Board"])
app.include_router(simulator.router, prefix="/api/v1", tags=["What-If Simulator"])
app.include_router(system.router, prefix="/api/v1", tags=["System"])

handler = Mangum(app, lifespan="off")

if config.IS_LAMBDA:
    warmup()
