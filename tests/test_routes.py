"""Flask route tests for the web application."""

import pytest

from app import app


@pytest.fixture
def client():
    app.config.update(
        TESTING=True,
        CSRF_ENABLED=False,
    )

    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Student Health Risk Prediction System" in response.data


def test_about_page_loads(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert (
    b"A compact prediction system built around a trained XGBoost pipeline."
    in response.data
)


def test_health_route_is_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["ready"] is True


def test_valid_prediction_route(client, fit_record):
    form_data = dict(fit_record)
    form_data["consent"] = "accepted"

    response = client.post("/predict", data=form_data)

    assert response.status_code == 200
    assert b"Fit" in response.data
    assert b"Probability for Every Class" in response.data


def test_consent_is_required(client, fit_record):
    response = client.post("/predict", data=fit_record)

    assert response.status_code == 400
    assert b"not a medical diagnosis" in response.data


def test_invalid_prediction_returns_400(client, fit_record):
    form_data = dict(fit_record)
    form_data["heart_rate"] = "300"
    form_data["consent"] = "accepted"

    response = client.post("/predict", data=form_data)

    assert response.status_code == 400
    assert b"Heart Rate" in response.data


def test_unknown_page_returns_404(client):
    response = client.get("/this-page-does-not-exist")
    assert response.status_code == 404
