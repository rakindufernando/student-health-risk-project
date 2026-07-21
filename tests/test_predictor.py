"""Tests for successful predictions and backend validation."""

import math

import pytest

from services.predictor import (
    get_service_status,
    predict_health_condition,
    prepare_input,
)


def test_model_service_is_ready():
    status = get_service_status()
    assert status["ready"] is True, status["error"]


@pytest.mark.parametrize(
    ("fixture_name", "expected_class"),
    [
        ("fit_record", "fit"),
        ("at_risk_record", "at-risk"),
        ("unhealthy_record", "unhealthy"),
    ],
)
def test_expected_prediction(request, fixture_name, expected_class):
    record = request.getfixturevalue(fixture_name)
    result = predict_health_condition(record)

    assert result["prediction"] == expected_class
    assert result["model_probability"] is not None
    assert 0 <= result["model_probability"] <= 100
    assert set(result["class_probabilities"]) == {
        "at-risk",
        "fit",
        "unhealthy",
    }
    assert math.isclose(
        sum(result["class_probabilities"].values()),
        100,
        abs_tol=0.05,
    )


def test_missing_field_is_rejected(fit_record):
    fit_record.pop("bmi")

    with pytest.raises(ValueError, match="BMI"):
        prepare_input(fit_record)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("sleep_duration", "-1"),
        ("heart_rate", "500"),
        ("bmi", "nan"),
        ("water_intake", "inf"),
        ("step_count", "not-a-number"),
    ],
)
def test_invalid_numeric_values_are_rejected(
    fit_record,
    field,
    invalid_value,
):
    fit_record[field] = invalid_value

    with pytest.raises(ValueError):
        prepare_input(fit_record)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("diet_type", "pizza"),
        ("stress_level", "extreme"),
        ("sleep_quality", "excellent"),
        ("gender", "unknown"),
    ],
)
def test_invalid_categories_are_rejected(
    fit_record,
    field,
    invalid_value,
):
    fit_record[field] = invalid_value

    with pytest.raises(ValueError):
        prepare_input(fit_record)
