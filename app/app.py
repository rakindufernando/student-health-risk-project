"""Flask application for the Student Health Risk Predictor."""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import time
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, session

from config import (
    CATEGORICAL_FIELDS,
    MAX_REQUEST_SIZE_BYTES,
    NUMERIC_FIELDS,
)
from services.predictor import (
    get_model_metadata,
    get_service_status,
    predict_health_condition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-development-key-change-before-deployment",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_SIZE_BYTES
app.config["CSRF_ENABLED"] = True


def generate_csrf_token() -> str:
    """Create one CSRF token per browser session."""
    token = session.get("csrf_token")

    if token is None:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token

    return token


app.jinja_env.globals["csrf_token"] = generate_csrf_token


@app.before_request
def verify_csrf_token() -> None:
    """Reject forged POST requests while keeping local tests simple."""
    if request.method != "POST":
        return

    if not app.config.get("CSRF_ENABLED", True):
        return

    submitted_token = request.form.get("csrf_token", "")
    stored_token = session.get("csrf_token", "")

    if not submitted_token or not stored_token:
        abort(400, description="The form session has expired. Please try again.")

    if not hmac.compare_digest(submitted_token, stored_token):
        abort(400, description="The form security token is invalid.")


def _template_values() -> dict[str, Any]:
    """Return shared values used by the application templates."""
    return {
        "numeric_fields": NUMERIC_FIELDS,
        "categorical_fields": CATEGORICAL_FIELDS,
        "model_metadata": get_model_metadata(),
        "service_status": get_service_status(),
    }


def render_home(
    error_message: str | None = None,
    previous_data: Any | None = None,
    status_code: int = 200,
):
    """Render the input page without repeating template arguments."""
    return (
        render_template(
            "index.html",
            error_message=error_message,
            previous_data=previous_data,
            **_template_values(),
        ),
        status_code,
    )


@app.route("/")
def home():
    """Display the prediction form."""
    return render_home()


@app.route("/predict", methods=["POST"])
def predict():
    """Validate the form and show a model prediction."""
    if request.form.get("consent") != "accepted":
        return render_home(
            error_message=(
                "Please confirm that you understand this is an educational "
                "prediction and not a medical diagnosis."
            ),
            previous_data=request.form,
            status_code=400,
        )

    service_status = get_service_status()

    if not service_status["ready"]:
        app.logger.error("Prediction requested while the model service was unavailable")
        return render_home(
            error_message=(
                "The prediction service is currently unavailable. "
                "Please check the saved model files and package versions."
            ),
            previous_data=request.form,
            status_code=503,
        )

    try:
        start_time = time.perf_counter()
        prediction_result = predict_health_condition(request.form)
        processing_time_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        app.logger.info(
            "Prediction completed successfully in %.2f milliseconds",
            processing_time_ms,
        )

        return render_template(
            "result.html",
            processing_time_ms=processing_time_ms,
            **prediction_result,
            **_template_values(),
        )

    except ValueError as error:
        app.logger.warning("Prediction input validation failed")
        return render_home(
            error_message=str(error),
            previous_data=request.form,
            status_code=400,
        )

    except RuntimeError:
        app.logger.exception("Prediction service error")
        return render_home(
            error_message=(
                "The prediction service is currently unavailable. "
                "Please try again after checking the model files."
            ),
            previous_data=request.form,
            status_code=503,
        )

    except Exception:
        app.logger.exception("Unexpected prediction error")
        return render_home(
            error_message=(
                "The prediction could not be completed. "
                "Please check the input values and try again."
            ),
            previous_data=request.form,
            status_code=500,
        )


@app.route("/about")
def about():
    """Display model, privacy, and limitation information."""
    return render_template("about.html", **_template_values())


@app.route("/health")
def health():
    """Return a small machine-readable service status response."""
    status = get_service_status()
    response_code = 200 if status["ready"] else 503
    return jsonify(status), response_code


@app.errorhandler(400)
def bad_request(error):
    """Show a readable message for invalid requests."""
    message = getattr(error, "description", "The submitted request is invalid.")
    return render_home(error_message=message, status_code=400)


@app.errorhandler(404)
def page_not_found(error):
    """Show a readable not-found page using the main template."""
    return render_home(
        error_message="The requested page was not found.",
        status_code=404,
    )


@app.errorhandler(413)
def request_too_large(error):
    """Reject unexpectedly large requests."""
    return render_home(
        error_message="The submitted request is too large.",
        status_code=413,
    )


@app.errorhandler(500)
def internal_server_error(error):
    """Hide technical details from the browser."""
    app.logger.error("Internal server error", exc_info=error)
    return render_home(
        error_message="An internal system error occurred. Please try again.",
        status_code=500,
    )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=False,
        use_reloader=False,
    )
