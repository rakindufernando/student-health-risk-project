"""Compare the two strongest tree models using three stratified folds."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_FOLDER / "data" / "kaggle" / "raw" / "train.csv"
RESULTS_FOLDER = PROJECT_FOLDER / "outputs" / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 150_000
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
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_model(model_name: str) -> Pipeline:
    if model_name == "XGBoost":
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


train_data = pd.read_csv(TRAIN_FILE)
X = train_data[numerical_columns + categorical_columns].copy()
y = train_data[TARGET_COLUMN].copy()

if len(X) > SAMPLE_SIZE:
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
else:
    X_sample = X.copy()
    y_sample = y.copy()

X_sample = X_sample.reset_index(drop=True)
y_sample = y_sample.reset_index(drop=True)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_sample)
folds = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_rows = []

for model_name in ["XGBoost", "Random Forest"]:
    for fold_number, (train_index, validation_index) in enumerate(
        folds.split(X_sample, y_encoded),
        start=1,
    ):
        X_train = X_sample.iloc[train_index]
        X_validation = X_sample.iloc[validation_index]
        y_train_encoded = y_encoded[train_index]
        y_validation = y_sample.iloc[validation_index]

        pipeline = build_model(model_name)
        fit_arguments = {}

        if model_name == "XGBoost":
            fit_arguments["classifier__sample_weight"] = compute_sample_weight(
                "balanced",
                y_train_encoded,
            )

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train_encoded, **fit_arguments)
        training_seconds = time.perf_counter() - start_time

        encoded_predictions = pipeline.predict(X_validation).astype(int)
        predictions = label_encoder.inverse_transform(encoded_predictions)

        fold_rows.append(
            {
                "Model": model_name,
                "Fold": fold_number,
                "Balanced Accuracy": balanced_accuracy_score(
                    y_validation,
                    predictions,
                ),
                "Macro F1": f1_score(
                    y_validation,
                    predictions,
                    average="macro",
                ),
                "Training Seconds": training_seconds,
            }
        )

        print(model_name, "fold", fold_number, "completed")

fold_results = pd.DataFrame(fold_rows)
fold_results.to_csv(
    RESULTS_FOLDER / "top_models_cross_validation_folds.csv",
    index=False,
)

summary = (
    fold_results.groupby("Model")
    .agg(
        Mean_Balanced_Accuracy=("Balanced Accuracy", "mean"),
        Standard_Deviation_Balanced_Accuracy=("Balanced Accuracy", "std"),
        Mean_Macro_F1=("Macro F1", "mean"),
        Standard_Deviation_Macro_F1=("Macro F1", "std"),
        Mean_Training_Seconds=("Training Seconds", "mean"),
    )
    .reset_index()
)
summary.to_csv(
    RESULTS_FOLDER / "top_models_cross_validation_summary.csv",
    index=False,
)

print()
print(summary)
