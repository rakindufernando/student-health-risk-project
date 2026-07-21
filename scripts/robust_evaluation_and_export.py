"""Create holdout evaluation, fairness evidence, and a deployment model candidate."""

from __future__ import annotations

import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_FOLDER / "data" / "kaggle" / "raw" / "train.csv"
MODELS_FOLDER = PROJECT_FOLDER / "models"
RESULTS_FOLDER = PROJECT_FOLDER / "outputs" / "results"
FIGURES_FOLDER = PROJECT_FOLDER / "outputs" / "figures"

MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COLUMN = "health_condition"
SAVE_AS_FINAL_MODEL = False

numerical_columns = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

categorical_columns = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

feature_columns = numerical_columns + categorical_columns


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def load_best_parameters() -> dict:
    parameter_file = RESULTS_FOLDER / "xgboost_best_parameters.json"

    if parameter_file.exists():
        with parameter_file.open("r", encoding="utf-8") as input_stream:
            return json.load(input_stream)

    return {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_lambda": 1.0,
    }


def build_model(parameters: dict) -> Pipeline:
    classifier = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **parameters,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def evaluate_predictions(
    true_values: pd.Series,
    predicted_values: np.ndarray,
) -> dict:
    return {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_values, predicted_values)
        ),
        "macro_f1": float(
            f1_score(true_values, predicted_values, average="macro")
        ),
    }


train_data = pd.read_csv(TRAIN_FILE)
X = train_data[feature_columns].copy()
y = train_data[TARGET_COLUMN].copy()

# First reserve an untouched 15 percent internal test set.
X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Then split the remaining 85 percent into 70 percent training and 15 percent validation.
validation_fraction_of_development = 0.15 / 0.85
X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=validation_fraction_of_development,
    stratify=y_development,
    random_state=RANDOM_STATE,
)

print("Training rows", len(X_train))
print("Validation rows", len(X_validation))
print("Internal test rows", len(X_test))

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_validation_encoded = label_encoder.transform(y_validation)
y_test_encoded = label_encoder.transform(y_test)

best_parameters = load_best_parameters()
model = build_model(best_parameters)
training_weights = compute_sample_weight("balanced", y_train_encoded)

start_training = time.perf_counter()
model.fit(
    X_train,
    y_train_encoded,
    classifier__sample_weight=training_weights,
)
training_seconds = time.perf_counter() - start_training

validation_encoded_predictions = model.predict(X_validation).astype(int)
test_encoded_predictions = model.predict(X_test).astype(int)
validation_predictions = label_encoder.inverse_transform(
    validation_encoded_predictions
)
test_predictions = label_encoder.inverse_transform(test_encoded_predictions)

validation_metrics = evaluate_predictions(y_validation, validation_predictions)
test_metrics = evaluate_predictions(y_test, test_predictions)

all_metrics = {
    "training_rows": len(X_train),
    "validation_rows": len(X_validation),
    "test_rows": len(X_test),
    "training_seconds": training_seconds,
    "validation": validation_metrics,
    "internal_test": test_metrics,
    "best_parameters": best_parameters,
}

with (RESULTS_FOLDER / "robust_holdout_metrics.json").open(
    "w",
    encoding="utf-8",
) as output_stream:
    json.dump(all_metrics, output_stream, indent=2)

# Save class-level precision, recall, and F1 evidence.
report_dictionary = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0,
)
report_table = pd.DataFrame(report_dictionary).transpose()
report_table.to_csv(RESULTS_FOLDER / "internal_test_classification_report.csv")

# Save the internal test confusion matrix.
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    labels=label_encoder.classes_,
)
plt.title("Internal Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    FIGURES_FOLDER / "internal_test_confusion_matrix.png",
    dpi=160,
)
plt.close()

# Fairness evidence by gender.
fairness_rows = []
test_with_gender = X_test.copy()
test_with_gender["actual"] = y_test.values
test_with_gender["predicted"] = test_predictions

for gender_name, group in test_with_gender.groupby("gender"):
    group_metrics = evaluate_predictions(
        group["actual"],
        group["predicted"],
    )

    fairness_rows.append(
        {
            "gender": gender_name,
            "records": len(group),
            "accuracy": group_metrics["accuracy"],
            "balanced_accuracy": group_metrics["balanced_accuracy"],
            "macro_f1": group_metrics["macro_f1"],
        }
    )

fairness_table = pd.DataFrame(fairness_rows)
fairness_table.to_csv(
    RESULTS_FOLDER / "gender_fairness_evaluation.csv",
    index=False,
)

# Probability reliability evidence.
test_probabilities = model.predict_proba(X_test)
number_of_classes = len(label_encoder.classes_)
one_hot_targets = np.eye(number_of_classes)[y_test_encoded]
multiclass_brier_score = float(
    np.mean(np.sum((one_hot_targets - test_probabilities) ** 2, axis=1))
)

with (RESULTS_FOLDER / "probability_reliability.json").open(
    "w",
    encoding="utf-8",
) as output_stream:
    json.dump(
        {"multiclass_brier_score": multiclass_brier_score},
        output_stream,
        indent=2,
    )

for class_number, class_name in enumerate(label_encoder.classes_):
    actual_binary = (y_test_encoded == class_number).astype(int)
    predicted_probability = test_probabilities[:, class_number]
    observed_fraction, predicted_mean = calibration_curve(
        actual_binary,
        predicted_probability,
        n_bins=10,
        strategy="quantile",
    )

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(
        predicted_mean,
        observed_fraction,
        marker="o",
        label=class_name,
    )
    plt.title(f"Probability Reliability for {class_name.title()}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_FOLDER / f"calibration_{class_name}.png",
        dpi=160,
    )
    plt.close()

# Single-record prediction speed evidence using 100 records.
speed_sample = X_test.sample(
    n=min(20, len(X_test)),
    random_state=RANDOM_STATE,
)
prediction_times_ms = []

for row_number in range(len(speed_sample)):
    one_record = speed_sample.iloc[[row_number]]
    start_prediction = time.perf_counter()
    model.predict(one_record)
    elapsed_ms = (time.perf_counter() - start_prediction) * 1000
    prediction_times_ms.append(elapsed_ms)

speed_metrics = {
    "records_tested": len(prediction_times_ms),
    "average_ms": statistics.mean(prediction_times_ms),
    "median_ms": statistics.median(prediction_times_ms),
    "minimum_ms": min(prediction_times_ms),
    "maximum_ms": max(prediction_times_ms),
}

with (RESULTS_FOLDER / "holdout_prediction_speed.json").open(
    "w",
    encoding="utf-8",
) as output_stream:
    json.dump(speed_metrics, output_stream, indent=2)

# Save the honest evaluation model first.
joblib.dump(model, MODELS_FOLDER / "evaluation_xgboost.joblib")
joblib.dump(
    label_encoder,
    MODELS_FOLDER / "evaluation_label_encoder.joblib",
)

# Retrain a deployment candidate using training and validation records.
X_deployment = pd.concat([X_train, X_validation], ignore_index=True)
y_deployment = pd.concat([y_train, y_validation], ignore_index=True)
y_deployment_encoded = label_encoder.transform(y_deployment)
deployment_weights = compute_sample_weight(
    "balanced",
    y_deployment_encoded,
)
deployment_model = build_model(best_parameters)
deployment_model.fit(
    X_deployment,
    y_deployment_encoded,
    classifier__sample_weight=deployment_weights,
)

candidate_model_file = MODELS_FOLDER / "final_model_candidate.joblib"
candidate_encoder_file = MODELS_FOLDER / "final_label_encoder_candidate.joblib"
joblib.dump(deployment_model, candidate_model_file)
joblib.dump(label_encoder, candidate_encoder_file)

if SAVE_AS_FINAL_MODEL:
    joblib.dump(deployment_model, MODELS_FOLDER / "final_model.joblib")
    joblib.dump(label_encoder, MODELS_FOLDER / "final_label_encoder.joblib")

metadata = {
    "model_name": "XGBoost",
    "model_type": "Ensemble classification model",
    "model_version": "2.0",
    "training_rows": len(X_deployment),
    "classes": label_encoder.classes_.tolist(),
    "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
    "validation_macro_f1": validation_metrics["macro_f1"],
    "internal_test_balanced_accuracy": test_metrics["balanced_accuracy"],
    "internal_test_macro_f1": test_metrics["macro_f1"],
    "internal_test_accuracy": test_metrics["accuracy"],
    "multiclass_brier_score": multiclass_brier_score,
    "primary_metric": "balanced_accuracy",
    "feature_count": len(feature_columns),
    "feature_order": feature_columns,
    "best_parameters": best_parameters,
    "random_state": RANDOM_STATE,
    "python_version": platform.python_version(),
    "pandas_version": version("pandas"),
    "numpy_version": version("numpy"),
    "scikit_learn_version": version("scikit-learn"),
    "xgboost_version": version("xgboost"),
    "joblib_version": version("joblib"),
    "notes": (
        "The internal test set was kept separate from model tuning. "
        "The web application displays model probability, not medical certainty."
    ),
}

with (MODELS_FOLDER / "model_metadata.json").open(
    "w",
    encoding="utf-8",
) as output_stream:
    json.dump(metadata, output_stream, indent=2)

print()
print("Validation metrics")
print(json.dumps(validation_metrics, indent=2))
print()
print("Internal test metrics")
print(json.dumps(test_metrics, indent=2))
print()
print("Fairness evidence")
print(fairness_table)
print()
print("Deployment candidate saved", candidate_model_file)
print("Set SAVE_AS_FINAL_MODEL to True only after checking the candidate")
