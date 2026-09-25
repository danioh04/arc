import logging

import redis
from pydantic import BaseModel
from redis.backoff import NoBackoff
from redis.retry import Retry

from arc.core import config

logger = logging.getLogger(__name__)

TTL_SECONDS = 3600

_client: redis.Redis | None = None
_namespace: str = ""


def init_cache() -> None:
    global _client
    if not config.REDIS_URL:
        logger.info("No REDIS_URL configured; running uncached.")
        return
    try:
        client = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
            retry=Retry(NoBackoff(), 0),
        )
        client.ping()
        _client = client
        logger.info("Connected to Redis.")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Redis unavailable ({e}). Running uncached.")


def close_cache() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def cache_is_healthy() -> bool:
    if _client is None:
        return False
    try:
        return bool(_client.ping())
    except Exception:  # noqa: BLE001
        return False


def set_cache_namespace(value: str) -> None:
    global _namespace
    _namespace = value


def _namespaced(key: str) -> str:
    return f"{_namespace}:{key}"


def read_cached[T: BaseModel](key: str, schema_cls: type[T]) -> T | None:
    if _client is None:
        return None
    try:
        raw = _client.get(_namespaced(key))
        if raw is None:
            return None
        return schema_cls.model_validate_json(raw)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Cache read/validation failed for key '{key}': {e}")
        return None


def write_cached(key: str, data: BaseModel) -> None:
    if _client is None:
        return
    try:
        _client.set(_namespaced(key), data.model_dump_json(), ex=TTL_SECONDS)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Cache write failed for key '{key}': {e}")
