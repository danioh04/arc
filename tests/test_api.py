import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from arc.main import app, handler
from arc.services.prediction import prediction_service

ZION = "Zion Williamson"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _api_gateway_event(path: str) -> dict:
    return {
        "version": "2.0",
        "routeKey": f"GET {path}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"host": "example.execute-api.us-east-1.amazonaws.com"},
        "isBase64Encoded": False,
        "requestContext": {
            "accountId": "000000000000",
            "apiId": "test",
            "domainName": "example.execute-api.us-east-1.amazonaws.com",
            "requestId": "test",
            "routeKey": f"GET {path}",
            "stage": "$default",
            "time": "24/Sep/2026:00:00:00 +0000",
            "timeEpoch": 1790000000000,
            "http": {
                "method": "GET",
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "pytest",
            },
        },
    }


def test_the_list_searches_by_alias_and_filters_what_it_displays(client):
    cam = client.get("/api/v1/prospects", params={"search": "Cam Thomas"}).json()
    guards = client.get("/api/v1/prospects", params={"position": "PG", "page_size": 100}).json()["items"]
    strong = client.get("/api/v1/prospects", params={"min_bpm": 10, "page_size": 100}).json()["items"]
    wildcard = client.get("/api/v1/prospects", params={"search": "z_on"}).json()
    assert cam["pagination"]["total_records"] >= 1
    assert all("thomas" in item["player_name"].lower() for item in cam["items"])
    assert guards
    assert {item["primary_position"] for item in guards} == {"G"}
    assert strong
    assert all(item["latest_bpm"] >= 10 for item in strong)
    assert wildcard["pagination"]["total_records"] == 0


def test_an_unknown_position_is_a_422_everywhere(client):
    listing = client.get("/api/v1/prospects", params={"position": "QB"})
    simulator = client.post("/api/v1/simulator", json={"primary_position": "QB"})
    assert listing.status_code == simulator.status_code == 422
    assert listing.json()["detail"].startswith("Unknown position 'QB'. Use G, W, F or C")
    assert any("Unknown position" in error["msg"] for error in simulator.json()["detail"])


def test_a_prospect_resolves_by_uuid_name_or_slug_and_never_by_a_partial_name(client):
    by_name = client.get(f"/api/v1/prospects/{ZION}").json()
    assert client.get(f"/api/v1/prospects/{by_name['prospect_id']}").json()["player_name"] == ZION
    assert client.get("/api/v1/prospects/zion-williamson").json()["player_name"] == ZION
    assert by_name["college_seasons"]
    partial = client.get("/api/v1/prospects/Tho")
    assert partial.status_code == 404
    assert partial.json()["detail"] == "Prospect 'Tho' not found in database."
    assert client.get("/api/v1/prospects/Zion-W").status_code == 404


def test_a_prediction_is_a_tier_distribution_with_shap_factors(client):
    body = client.get(f"/api/v1/prospects/{ZION}/predict").json()
    assert body["projected_tier"] in (1, 2, 3, 4, 5)
    assert [tier["tier"] for tier in body["tier_probabilities"]] == [1, 2, 3, 4, 5]
    assert sum(tier["probability"] for tier in body["tier_probabilities"]) == pytest.approx(1.0, abs=1e-3)
    assert body["top_positive_factors"]
    assert all(f["shap_value"] > 0 and f["contribution_sigma"][0] == "+" for f in body["top_positive_factors"])
    assert all(f["shap_value"] < 0 and f["contribution_sigma"][0] == "-" for f in body["top_negative_factors"])


def test_comparisons_return_k_other_players_closest_first(client):
    comps = client.get(f"/api/v1/prospects/{ZION}/comparisons", params={"k": 3}).json()["comparisons"]
    scores = [comp["similarity_score"] for comp in comps]
    assert len(comps) == 3
    assert ZION not in {comp["player_name"] for comp in comps}
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_draft_classes_list_every_class_and_unknown_ones_are_404(client):
    classes = {row["draft_year"]: row["drafted_count"] for row in client.get("/api/v1/draft-classes").json()["classes"]}
    unknown = client.get("/api/v1/draft-classes/1990/big-board")
    assert set(classes) == set(range(2008, 2027))
    assert all(0 < count <= 60 for count in classes.values())
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "No prospect records found for draft class year 1990."


def test_a_board_ranks_a_class_by_projected_z_and_prices_picks_within_it(client):
    board = client.get("/api/v1/draft-classes/2025/big-board", params={"limit": 500}).json()
    rows = board["board"]
    assert board["total_prospects"] == len(rows) > 30
    assert [row["model_rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert [row["projected_z"] for row in rows] == sorted((row["projected_z"] for row in rows), reverse=True)
    assert sum(row["value_delta"] for row in rows) == 0


def test_the_simulator_projects_a_hypothetical_stat_line(client):
    response = client.post(
        "/api/v1/simulator", params={"k": 2}, json={"player_name": "Test Prospect", "primary_position": "W"}
    )
    body = response.json()
    assert response.status_code == 200
    assert body["prediction"]["player_name"] == "Test Prospect"
    assert body["prediction"]["prospect_id"] is None
    assert len(body["comparisons"]) == 2


def test_model_card_serves_the_measured_metrics(client):
    card = client.get("/api/v1/models/metadata").json()
    evaluation = card["evaluation"]
    assert len(card["feature_names"]) == 16
    assert card["holdout_rows"] > 100
    assert evaluation["spearman_rank_correlation"] > 0
    assert evaluation["ranking"]["star_top10_precision"] > 0
    assert set(evaluation["baselines"]["draft_market_order"]) == {
        "spearman_rank_correlation",
        "star_top10_precision",
        "star_top25_precision",
    }
    assert {"n_estimators", "max_depth", "learning_rate"} <= set(evaluation["hyperparameters"])


def test_a_missing_model_is_a_503_not_a_crash(client, monkeypatch):
    monkeypatch.setattr(prediction_service, "regressor", None)
    health = client.get("/api/v1/health")
    prediction = client.get(f"/api/v1/prospects/{ZION}/predict")
    assert (health.status_code, health.json()["status"]) == (503, "degraded")
    assert prediction.status_code == 503
    assert (
        prediction.json()["detail"]
        == "Machine learning model 'XGBoost Prospect Regressor' is not loaded or unavailable."
    )


@pytest.mark.usefixtures("client")
def test_the_lambda_handler_serves_api_gateway_events():
    health = handler(_api_gateway_event("/api/v1/health"), SimpleNamespace())
    missing = handler(_api_gateway_event("/api/v1/prospects/NotARealPlayer99"), SimpleNamespace())
    body = json.loads(health["body"])
    assert health["statusCode"] == 200
    assert (body["status"], body["database_connected"], body["models_loaded"]) == ("healthy", True, True)
    assert missing["statusCode"] == 404
