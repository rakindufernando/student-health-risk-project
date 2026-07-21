"""Load the saved model, validate input values, and create predictions."""

from __future__ import annotations

import json
import logging
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from config import (
    CATEGORICAL_FIELDS,
    CATEGORICAL_FEATURES,
    LOW_PROBABILITY_THRESHOLD,
    NUMERIC_FIELDS,
    NUMERIC_FEATURES,
    REQUIRED_FEATURES,
)

LOGGER = logging.getLogger(__name__)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_FOLDER = CURRENT_FILE.parents[2]
MODELS_FOLDER = PROJECT_FOLDER / "models"
MODEL_FILE = MODELS_FOLDER / "final_model.joblib"
LABEL_ENCODER_FILE = MODELS_FOLDER / "final_label_encoder.joblib"
METADATA_FILE = MODELS_FOLDER / "model_metadata.json"

final_model: Any | None = None
label_encoder: Any | None = None
model_metadata: dict[str, Any] = {}
service_ready = False
service_error: str | None = None


def _package_version(package_name: str) -> str:
    """Return an installed package version without stopping the app."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def _load_metadata() -> dict[str, Any]:
    """Load human-readable model information when the file is available."""
    if not METADATA_FILE.exists():
        return {
            "model_name": "XGBoost",
            "model_version": "1.0",
            "classes": ["at-risk", "fit", "unhealthy"],
            "validation_balanced_accuracy": 0.9097,
            "macro_f1_score": 0.7541,
        }

    with METADATA_FILE.open("r", encoding="utf-8") as metadata_stream:
        return json.load(metadata_stream)


def load_model_files() -> bool:
    """Load the model files and record whether the service is ready."""
    global final_model
    global label_encoder
    global model_metadata
    global service_ready
    global service_error

    service_ready = False
    service_error = None

    try:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Final model file was not found at {MODEL_FILE}"
            )

        if not LABEL_ENCODER_FILE.exists():
            raise FileNotFoundError(
                f"Label encoder file was not found at {LABEL_ENCODER_FILE}"
            )

        final_model = joblib.load(MODEL_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        model_metadata = _load_metadata()

        if not hasattr(final_model, "predict"):
            raise TypeError("The loaded model does not support prediction")

        if not hasattr(label_encoder, "inverse_transform"):
            raise TypeError("The loaded label encoder is invalid")

        service_ready = True
        LOGGER.info("Prediction model loaded successfully")
        return True

    except Exception as error:
        final_model = None
        label_encoder = None
        service_error = str(error)
        LOGGER.exception("Prediction model could not be loaded")
        return False


def get_service_status() -> dict[str, Any]:
    """Return a simple status object for the application and health route."""
    return {
        "ready": service_ready,
        "error": service_error,
        "model_file": MODEL_FILE.name,
        "model_name": model_metadata.get("model_name", "Unknown"),
        "model_version": model_metadata.get("model_version", "Unknown"),
        "scikit_learn_version": _package_version("scikit-learn"),
        "xgboost_version": _package_version("xgboost"),
    }


def get_model_metadata() -> dict[str, Any]:
    """Return a copy so templates cannot modify the shared dictionary."""
    return dict(model_metadata)


def _required_value(form_data: Mapping[str, Any], feature: str) -> str:
    """Read one required value and reject missing or blank data."""
    raw_value = form_data.get(feature)

    if raw_value is None or str(raw_value).strip() == "":
        label = NUMERIC_FIELDS.get(feature, CATEGORICAL_FIELDS.get(feature, {})).get(
            "label",
            feature.replace("_", " ").title(),
        )
        raise ValueError(f"Please provide a value for {label}")

    return str(raw_value).strip()


def _validate_numeric_value(feature: str, raw_value: str) -> float:
    """Convert one numeric value and check its supported dataset range."""
    rule = NUMERIC_FIELDS[feature]
    label = rule["label"]

    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid number") from error

    if not math.isfinite(numeric_value):
        raise ValueError(f"{label} must be a finite number")

    minimum = float(rule["minimum"])
    maximum = float(rule["maximum"])

    if numeric_value < minimum or numeric_value > maximum:
        raise ValueError(
            f"{label} must be between {minimum:g} and {maximum:g} "
            f"{rule['unit']}"
        )

    return numeric_value


def _validate_categorical_value(feature: str, raw_value: str) -> str:
    """Clean one category and reject unsupported values."""
    rule = CATEGORICAL_FIELDS[feature]
    cleaned_value = raw_value.lower()
    allowed_values = rule["options"]

    if cleaned_value not in allowed_values:
        allowed_labels = ", ".join(allowed_values.values())
        raise ValueError(
            f"{rule['label']} must be one of the following values, "
            f"{allowed_labels}"
        )

    return cleaned_value


def prepare_input(
    form_data: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the submitted values and create a one-row DataFrame."""
    cleaned_data: dict[str, Any] = {}

    for feature in NUMERIC_FEATURES:
        raw_value = _required_value(form_data, feature)
        cleaned_data[feature] = _validate_numeric_value(feature, raw_value)

    for feature in CATEGORICAL_FEATURES:
        raw_value = _required_value(form_data, feature)
        cleaned_data[feature] = _validate_categorical_value(feature, raw_value)

    input_data = pd.DataFrame([cleaned_data], columns=REQUIRED_FEATURES)

    if input_data.columns.tolist() != REQUIRED_FEATURES:
        raise ValueError("The submitted feature order does not match the model schema")

    if input_data.isna().any().any():
        raise ValueError("The submitted information contains a missing value")

    return input_data, cleaned_data


def _calculate_class_probabilities(input_data: pd.DataFrame) -> dict[str, float]:
    """Return model probabilities using readable class names."""
    if final_model is None or label_encoder is None:
        return {}

    if not hasattr(final_model, "predict_proba"):
        return {}

    raw_probabilities = final_model.predict_proba(input_data)[0]
    encoded_classes = getattr(
        final_model,
        "classes_",
        list(range(len(raw_probabilities))),
    )

    readable_classes = label_encoder.inverse_transform(
        [int(class_number) for class_number in encoded_classes]
    )

    return {
        str(class_name): round(float(probability) * 100, 2)
        for class_name, probability in zip(
            readable_classes,
            raw_probabilities,
            strict=True,
        )
    }


def predict_health_condition(form_data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one student record and predict its health condition."""
    if not service_ready or final_model is None or label_encoder is None:
        raise RuntimeError(
            "The prediction service is currently unavailable. "
            "Check the saved model files and package versions."
        )

    input_data, cleaned_data = prepare_input(form_data)

    encoded_prediction = final_model.predict(input_data)[0]
    predicted_class = str(
        label_encoder.inverse_transform([int(encoded_prediction)])[0]
    )

    class_probabilities = _calculate_class_probabilities(input_data)
    model_probability = class_probabilities.get(predicted_class)

    low_probability_warning = (
        model_probability is not None
        and model_probability < LOW_PROBABILITY_THRESHOLD
    )

    return {
        "prediction": predicted_class,
        "model_probability": model_probability,
        "class_probabilities": class_probabilities,
        "low_probability_warning": low_probability_warning,
        "input_data": cleaned_data,
        "model_name": model_metadata.get("model_name", "XGBoost"),
        "model_version": model_metadata.get("model_version", "1.0"),
    }


load_model_files()
