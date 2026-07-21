"""Measure prediction speed using the saved final model."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
APP_FOLDER = PROJECT_FOLDER / "app"
sys.path.insert(0, str(APP_FOLDER))

from services.predictor import predict_health_condition  # noqa: E402

TEST_RECORD = {
    "sleep_duration": "7.68",
    "heart_rate": "70",
    "bmi": "16.00",
    "calorie_expenditure": "2575",
    "step_count": "13804",
    "exercise_duration": "52",
    "water_intake": "1.72",
    "diet_type": "balanced",
    "stress_level": "low",
    "sleep_quality": "good",
    "physical_activity_level": "active",
    "smoking_alcohol": "no",
    "gender": "other",
}

NUMBER_OF_RUNS = 20

# Warm up once so model loading is not included in the prediction average.
predict_health_condition(TEST_RECORD)

prediction_times_ms = []

for _ in range(NUMBER_OF_RUNS):
    start_time = time.perf_counter()
    predict_health_condition(TEST_RECORD)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    prediction_times_ms.append(elapsed_ms)

benchmark_result = {
    "number_of_runs": NUMBER_OF_RUNS,
    "average_ms": round(statistics.mean(prediction_times_ms), 3),
    "median_ms": round(statistics.median(prediction_times_ms), 3),
    "minimum_ms": round(min(prediction_times_ms), 3),
    "maximum_ms": round(max(prediction_times_ms), 3),
}

results_folder = PROJECT_FOLDER / "outputs" / "results"
results_folder.mkdir(parents=True, exist_ok=True)
result_file = results_folder / "prediction_speed_benchmark.json"

with result_file.open("w", encoding="utf-8") as output_stream:
    json.dump(benchmark_result, output_stream, indent=2)

print(json.dumps(benchmark_result, indent=2))
print("Saved", result_file)
