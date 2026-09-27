import argparse
import json
import os
import random
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import redis

API = "/api/v1"
WARMUP_REQUESTS = 3
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Target(NamedTuple):
    endpoint: str
    path: str
    cache_key: str


def fetch(base_url: str, path: str) -> tuple[float, Any]:
    request = urllib.request.Request(base_url + path, headers={"Accept": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    return (time.perf_counter() - start) * 1000, json.loads(body)


def cache_namespace(model_card: dict[str, Any]) -> str:
    fingerprint = str(model_card.get("dataset_fingerprint") or "")[:16]
    version = str(model_card.get("model_version") or "unknown")
    return f"{fingerprint}-{version}".strip("-")


def build_targets(base_url: str, namespace: str, prospects: int, seed: int) -> list[Target]:
    _, classes = fetch(base_url, f"{API}/draft-classes")
    targets = [
        Target("big board", f"{API}/draft-classes/{c['draft_year']}/big-board", f"nba:big_board:{c['draft_year']}")
        for c in classes["classes"]
    ]
    _, page = fetch(base_url, f"{API}/prospects?page_size=100")
    ids = [item["prospect_id"] for item in page["items"]]
    for prospect_id in random.Random(seed).sample(ids, min(prospects, len(ids))):
        targets.append(Target("prediction", f"{API}/prospects/{prospect_id}/predict", f"nba:pred:{prospect_id}"))
        targets.append(Target("comparisons", f"{API}/prospects/{prospect_id}/comparisons", f"nba:comps:{prospect_id}"))
    return [t._replace(cache_key=f"{namespace}:{t.cache_key}") for t in targets]


def summarize(label: str, misses: list[float], hits: list[float]) -> None:
    miss, hit = statistics.median(misses), statistics.median(hits)
    print(f"{label:<12} n={len(misses):<3} miss {miss:8.1f} ms   hit {hit:8.1f} ms   {miss / hit:5.1f}x faster")


def main() -> None:
    load_env_file(ENV_FILE)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("LAMBDA_URL", ""), help="Deployed base URL")
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", ""), help="Redis used by the deployment")
    parser.add_argument("--prospects", type=int, default=20, help="Prospects to sample for predictions and comps")
    parser.add_argument("--hits", type=int, default=5, help="Cached requests timed per target")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.url:
        parser.error("pass --url or set LAMBDA_URL")
    base_url = args.url.rstrip("/")
    for _ in range(WARMUP_REQUESTS):
        _, health = fetch(base_url, f"{API}/health")
    if not health.get("cache_connected"):
        raise SystemExit("The deployment reports no Redis connection, so there is no cache to benchmark.")
    cache = None
    if args.redis_url:
        cache = redis.Redis.from_url(args.redis_url, socket_connect_timeout=5, socket_timeout=5)
        cache.ping()
    else:
        print("WARNING: no REDIS_URL, so first requests are not guaranteed misses. Treat the results as unverified.\n")
    _, model_card = fetch(base_url, f"{API}/models/metadata")
    targets = build_targets(base_url, cache_namespace(model_card), args.prospects, args.seed)
    misses: dict[str, list[float]] = defaultdict(list)
    hits: dict[str, list[float]] = defaultdict(list)
    for target in targets:
        if cache is not None:
            cache.delete(target.cache_key)
        miss_ms, _ = fetch(base_url, target.path)
        if cache is not None and not cache.exists(target.cache_key):
            raise SystemExit(f"{target.cache_key} was not cached after a miss; check the key format.")
        misses[target.endpoint].append(miss_ms)
        hits[target.endpoint].extend(fetch(base_url, target.path)[0] for _ in range(args.hits))
    print(f"{base_url}  (medians, network round trip included)\n")
    for endpoint in misses:
        summarize(endpoint, misses[endpoint], hits[endpoint])
    all_misses = [ms for values in misses.values() for ms in values]
    all_hits = [ms for values in hits.values() for ms in values]
    print()
    summarize("all", all_misses, all_hits)


if __name__ == "__main__":
    main()
