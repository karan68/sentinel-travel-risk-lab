from copy import deepcopy

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

LOW_RISK_BOOKING = {
    "booking_id": "BKG-1001",
    "agent_id": "AGT-TRUSTED",
    "account_age_days": 1_200,
    "total_bookings_90d": 500,
    "chargebacks_90d": 0,
    "cancellations_90d": 12,
    "bookings_24h": 4,
    "recent_holds_24h": 2,
    "recent_late_cancellations_90d": 0,
    "seats_requested": 2,
    "hours_until_departure": 240,
    "ip_country": "IN",
    "card_country": "IN",
    "payment_attempts_10m": 1,
    "device_linked_to_fraud": False,
    "card_on_blocklist": False,
    "interaction_duration_seconds": 94,
    "fields_pasted": 1,
    "pointer_events": 46,
}


def test_low_risk_booking_is_approved() -> None:
    response = client.post("/api/assess", json=LOW_RISK_BOOKING)

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "approve"
    assert result["overall_score"] < 0.35
    assert result["engine_mode"] in {"rules_only_untrained", "hybrid_xgboost_rules"}


def test_multiple_payment_signals_require_review() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking.update(
        account_age_days=1,
        ip_country="RU",
        hours_until_departure=1.5,
        payment_attempts_10m=2,
    )

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "review"
    assert result["payment_fraud"]["score"] >= 0.5


def test_blocklisted_payment_is_blocked() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking["card_on_blocklist"] = True

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "block"
    assert result["payment_fraud"]["reasons"][0]["code"] == "card_blocklist"


def test_invalid_history_is_rejected() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking.update(total_bookings_90d=2, chargebacks_90d=3)

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 422


def test_model_metadata_discloses_synthetic_source() -> None:
    response = client.get("/api/model")

    assert response.status_code == 200
    result = response.json()
    assert result["row_count"] == 15_000
    assert "Synthetic" in result["data_disclosure"]
    assert set(result["test_metrics"]) == {"payment_fraud", "inventory_abuse"}


def test_detective_graph_is_labeled_as_demo_data() -> None:
    response = client.get("/api/network/AGT-2048")

    assert response.status_code == 200
    result = response.json()
    assert result["provider"] == "offline_demo"
    assert result["nodes"][0]["id"] == "AGT-2048"
    assert "Synthetic" in result["data_disclosure"]


def test_analyst_brief_uses_offline_evidence_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assessment = client.post("/api/assess", json=LOW_RISK_BOOKING).json()

    response = client.post("/api/brief", json={"assessment": assessment})

    assert response.status_code == 200
    result = response.json()
    assert result["provider"] == "offline_deterministic"
    assert result["text"] == assessment["summary"]
    assert "No external service" in result["data_disclosure"]


def test_local_vite_origin_is_allowed_by_cors() -> None:
    response = client.options(
        "/api/model",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_inventory_abuse_signals_block_without_payment_risk() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking.update(
        seats_requested=32,
        total_bookings_90d=90,
        cancellations_90d=48,
        bookings_24h=42,
        recent_holds_24h=24,
        recent_late_cancellations_90d=8,
    )

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "block"
    assert result["inventory_abuse"]["score"] >= 0.8
    assert result["payment_fraud"]["score"] < 0.35


def test_bot_telemetry_is_separate_from_payment_fraud() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking.update(
        interaction_duration_seconds=2,
        fields_pasted=10,
        pointer_events=0,
    )

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "review"
    assert result["bot_likelihood"]["score"] == 0.8
    assert result["payment_fraud"]["score"] < 0.35


def test_lowercase_country_code_is_rejected() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking["ip_country"] = "in"

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 422


def test_zero_seats_is_rejected() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking["seats_requested"] = 0

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 422


def test_negative_history_count_is_rejected() -> None:
    booking = deepcopy(LOW_RISK_BOOKING)
    booking["chargebacks_90d"] = -1

    response = client.post("/api/assess", json=booking)

    assert response.status_code == 422