from pydantic import BaseModel, Field


class PaginationMeta(BaseModel):
    total_records: int = Field(..., description="Total number of matching records in database")
    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")


class PaginatedResponse[T](BaseModel):
    items: list[T] = Field(..., description="List of items for current page")
    pagination: PaginationMeta


class HealthCheck(BaseModel):
    status: str = Field(..., description="Overall system status (healthy/degraded); the cache is advisory")
    database_connected: bool = Field(..., description="SQLite database connectivity status")
    cache_connected: bool = Field(..., description="Redis/Cache status")
    models_loaded: bool = Field(..., description="Machine learning models availability")
    active_model_version: str = Field(..., description="Active ML model version")
