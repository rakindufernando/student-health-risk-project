"""Run a small, explainable XGBoost tuning experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_FOLDER / "data" / "kaggle" / "raw" / "train.csv"
RESULTS_FOLDER = PROJECT_FOLDER / "outputs" / "results"
MODELS_FOLDER = PROJECT_FOLDER / "models"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TUNING_SAMPLE_SIZE = 150_000
TARGET_COLUMN = "health_condition"

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


train_data = pd.read_csv(TRAIN_FILE)
X = train_data.drop(columns=[TARGET_COLUMN, "id"], errors="ignore")
y = train_data[TARGET_COLUMN].copy()

if len(X) > TUNING_SAMPLE_SIZE:
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=TUNING_SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
else:
    X_sample = X.copy()
    y_sample = y.copy()

X_train, X_validation, y_train, y_validation = train_test_split(
    X_sample,
    y_sample,
    test_size=0.20,
    stratify=y_sample,
    random_state=RANDOM_STATE,
)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_validation_encoded = label_encoder.transform(y_validation)
sample_weights = compute_sample_weight("balanced", y_train_encoded)

parameter_options = [
    {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "gamma": 0.1,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "gamma": 0.0,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 180,
        "max_depth": 6,
        "learning_rate": 0.06,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "gamma": 0.1,
        "reg_lambda": 2.0,
    },
]

result_rows = []
best_pipeline = None
best_result = None

for experiment_number, parameters in enumerate(parameter_options, start=1):
    classifier = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **parameters,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )

    start_time = time.perf_counter()
    pipeline.fit(
        X_train,
        y_train_encoded,
        classifier__sample_weight=sample_weights,
    )
    training_seconds = time.perf_counter() - start_time

    validation_encoded_predictions = pipeline.predict(X_validation)
    validation_predictions = label_encoder.inverse_transform(
        validation_encoded_predictions.astype(int)
    )

    balanced_accuracy = balanced_accuracy_score(
        y_validation,
        validation_predictions,
    )
    macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro",
    )

    row = {
        "Experiment": experiment_number,
        "Validation balanced accuracy": balanced_accuracy,
        "Validation macro F1": macro_f1,
        "Training seconds": training_seconds,
        **parameters,
    }
    result_rows.append(row)

    if best_result is None or balanced_accuracy > best_result[
        "Validation balanced accuracy"
    ]:
        best_result = row
        best_pipeline = pipeline

    print(
        "Experiment",
        experiment_number,
        "balanced accuracy",
        round(balanced_accuracy, 4),
        "macro F1",
        round(macro_f1, 4),
    )

results = pd.DataFrame(result_rows).sort_values(
    by=["Validation balanced accuracy", "Validation macro F1"],
    ascending=False,
)
results.to_csv(RESULTS_FOLDER / "xgboost_tuning_results.csv", index=False)

best_parameter_names = list(parameter_options[0].keys())
best_parameters = {
    parameter_name: best_result[parameter_name]
    for parameter_name in best_parameter_names
}

with (RESULTS_FOLDER / "xgboost_best_parameters.json").open(
    "w",
    encoding="utf-8",
) as output_stream:
    json.dump(best_parameters, output_stream, indent=2)

if best_pipeline is not None:
    joblib.dump(
        best_pipeline,
        MODELS_FOLDER / "xgboost_tuned_sample_model.joblib",
    )
    joblib.dump(
        label_encoder,
        MODELS_FOLDER / "xgboost_tuned_sample_label_encoder.joblib",
    )

print()
print("Best parameters")
print(json.dumps(best_parameters, indent=2))
print("Tuning results saved")
