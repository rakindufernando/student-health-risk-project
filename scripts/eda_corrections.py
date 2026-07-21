"""Create corrected EDA evidence without treating the ID as a health feature."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_FOLDER / "data" / "kaggle" / "raw"
EDA_FOLDER = PROJECT_FOLDER / "outputs" / "eda"
RESULTS_FOLDER = PROJECT_FOLDER / "outputs" / "results"

EDA_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_FOLDER / "train.csv"
TEST_FILE = DATA_FOLDER / "test.csv"
TARGET_COLUMN = "health_condition"
IDENTIFIER_COLUMNS = ["id"]

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

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

# Basic dataset evidence
summary_table = pd.DataFrame(
    {
        "Dataset": ["Training", "Kaggle test"],
        "Rows": [len(train_data), len(test_data)],
        "Columns": [train_data.shape[1], test_data.shape[1]],
        "Missing values": [
            int(train_data.isna().sum().sum()),
            int(test_data.isna().sum().sum()),
        ],
    }
)
summary_table.to_csv(RESULTS_FOLDER / "dataset_summary.csv", index=False)

# Correct duplicate checks
exact_duplicates_with_id = int(train_data.duplicated().sum())
feature_duplicates_without_id = int(
    train_data.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    .duplicated()
    .sum()
)

pd.DataFrame(
    {
        "Check": [
            "Exact duplicate rows including ID",
            "Duplicate records after removing ID",
        ],
        "Duplicate count": [
            exact_duplicates_with_id,
            feature_duplicates_without_id,
        ],
    }
).to_csv(RESULTS_FOLDER / "duplicate_check.csv", index=False)

# Missing values by column
missing_table = pd.DataFrame(
    {
        "Training missing": train_data.isna().sum(),
        "Kaggle test missing": test_data.isna().sum(),
    }
)
missing_table.to_csv(RESULTS_FOLDER / "missing_value_summary.csv")

# Target class distribution
class_counts = train_data[TARGET_COLUMN].value_counts()
class_percentages = train_data[TARGET_COLUMN].value_counts(normalize=True) * 100

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")
plt.title("Target Class Distribution")
plt.xlabel("Health condition")
plt.ylabel("Number of records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(EDA_FOLDER / "target_class_distribution.png", dpi=160)
plt.close()

pd.DataFrame(
    {
        "Count": class_counts,
        "Percentage": class_percentages.round(2),
    }
).to_csv(RESULTS_FOLDER / "target_class_distribution.csv")

# Numerical distributions and separate boxplots
for column in numerical_columns:
    plt.figure(figsize=(8, 5))
    train_data[column].hist(bins=30)
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(EDA_FOLDER / f"{column}_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 3.8))
    plt.boxplot(train_data[column].dropna(), vert=False)
    plt.title(f"Boxplot of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(EDA_FOLDER / f"{column}_boxplot.png", dpi=160)
    plt.close()

# Correlation matrix without the ID column
correlation_matrix = train_data[numerical_columns].corr()
plt.figure(figsize=(9, 7))
image = plt.imshow(correlation_matrix, aspect="auto")
plt.colorbar(image)
plt.xticks(
    range(len(numerical_columns)),
    [column.replace("_", " ") for column in numerical_columns],
    rotation=45,
    ha="right",
)
plt.yticks(
    range(len(numerical_columns)),
    [column.replace("_", " ") for column in numerical_columns],
)
plt.title("Numerical Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(EDA_FOLDER / "numerical_correlation_matrix.png", dpi=160)
plt.close()
correlation_matrix.to_csv(RESULTS_FOLDER / "numerical_correlation_matrix.csv")

# Categorical relationships with the target
for column in categorical_columns:
    relationship = pd.crosstab(
        train_data[column],
        train_data[TARGET_COLUMN],
        normalize="index",
    ) * 100

    relationship.plot(kind="bar", stacked=True, figsize=(9, 5))
    plt.title(
        f"Health Condition Percentage by {column.replace('_', ' ').title()}"
    )
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.legend(title="Health condition")
    plt.tight_layout()
    plt.savefig(EDA_FOLDER / f"{column}_target_relationship.png", dpi=160)
    plt.close()

# IQR outlier evidence with a retain decision
outlier_rows = []

for column in numerical_columns:
    q1 = train_data[column].quantile(0.25)
    q3 = train_data[column].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outlier_count = int(
        (
            (train_data[column] < lower_limit)
            | (train_data[column] > upper_limit)
        ).sum()
    )

    outlier_rows.append(
        {
            "Feature": column,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower limit": round(lower_limit, 4),
            "Upper limit": round(upper_limit, 4),
            "Potential outliers": outlier_count,
            "Decision": "Retain and use range validation",
            "Reason": (
                "Values may represent genuine student differences and the "
                "tree-based model can handle non-linear boundaries."
            ),
        }
    )

pd.DataFrame(outlier_rows).to_csv(
    RESULTS_FOLDER / "outlier_analysis.csv",
    index=False,
)

# Training and test distribution comparison
comparison_rows = []

for column in numerical_columns:
    comparison_rows.append(
        {
            "Feature": column,
            "Type": "numerical",
            "Training mean": train_data[column].mean(),
            "Test mean": test_data[column].mean(),
            "Training standard deviation": train_data[column].std(),
            "Test standard deviation": test_data[column].std(),
        }
    )

pd.DataFrame(comparison_rows).to_csv(
    RESULTS_FOLDER / "train_test_numerical_comparison.csv",
    index=False,
)

categorical_comparisons = []

for column in categorical_columns:
    train_percentages = train_data[column].value_counts(normalize=True) * 100
    test_percentages = test_data[column].value_counts(normalize=True) * 100
    all_categories = sorted(set(train_percentages.index) | set(test_percentages.index))

    for category in all_categories:
        categorical_comparisons.append(
            {
                "Feature": column,
                "Category": category,
                "Training percentage": round(train_percentages.get(category, 0), 3),
                "Test percentage": round(test_percentages.get(category, 0), 3),
            }
        )

pd.DataFrame(categorical_comparisons).to_csv(
    RESULTS_FOLDER / "train_test_categorical_comparison.csv",
    index=False,
)

# Clear evidence that EDA influenced the model design
eda_decisions = pd.DataFrame(
    [
        {
            "EDA finding": "Strong target class imbalance",
            "Model decision": (
                "Use balanced accuracy, macro F1, stratified splits, and class weights."
            ),
        },
        {
            "EDA finding": "Mixed numerical and categorical features",
            "Model decision": "Use a ColumnTransformer preprocessing pipeline.",
        },
        {
            "EDA finding": "ID is only an identifier",
            "Model decision": "Remove ID before statistical analysis and modelling.",
        },
        {
            "EDA finding": "Non-linear feature relationships",
            "Model decision": "Compare tree ensembles and an ANN with simple baselines.",
        },
        {
            "EDA finding": "Different numerical scales",
            "Model decision": (
                "Scale numerical values for Logistic Regression and ANN."
            ),
        },
        {
            "EDA finding": "No current missing values",
            "Model decision": (
                "Keep imputation in the pipeline for future input robustness."
            ),
        },
        {
            "EDA finding": "Potential extreme numerical values",
            "Model decision": (
                "Retain plausible records and validate app inputs using dataset ranges."
            ),
        },
    ]
)
eda_decisions.to_csv(RESULTS_FOLDER / "eda_model_decision_summary.csv", index=False)

print("Corrected EDA evidence created")
print("Figures", EDA_FOLDER)
print("Tables", RESULTS_FOLDER)
