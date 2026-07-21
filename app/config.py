"""Shared application settings for the student health predictor."""

NUMERIC_FIELDS = {
    "sleep_duration": {
        "label": "Sleep Duration",
        "minimum": 3.0,
        "maximum": 10.0,
        "step": "0.01",
        "placeholder": "Example 7.68",
        "unit": "hours per day",
    },
    "heart_rate": {
        "label": "Heart Rate",
        "minimum": 50.0,
        "maximum": 107.7,
        "step": "0.1",
        "placeholder": "Example 77.5",
        "unit": "beats per minute",
    },
    "bmi": {
        "label": "BMI",
        "minimum": 16.0,
        "maximum": 34.82,
        "step": "0.01",
        "placeholder": "Example 26.49",
        "unit": "BMI value",
    },
    "calorie_expenditure": {
        "label": "Calorie Expenditure",
        "minimum": 1200.0,
        "maximum": 3580.0,
        "step": "1",
        "placeholder": "Example 2734",
        "unit": "kilocalories per day",
    },
    "step_count": {
        "label": "Daily Step Count",
        "minimum": 1002.0,
        "maximum": 14999.0,
        "step": "1",
        "placeholder": "Example 14751",
        "unit": "steps per day",
    },
    "exercise_duration": {
        "label": "Exercise Duration",
        "minimum": 0.0,
        "maximum": 99.8,
        "step": "0.1",
        "placeholder": "Example 42.9",
        "unit": "minutes per day",
    },
    "water_intake": {
        "label": "Water Intake",
        "minimum": 0.5,
        "maximum": 4.72,
        "step": "0.01",
        "placeholder": "Example 2.42",
        "unit": "litres per day",
    },
}

CATEGORICAL_FIELDS = {
    "diet_type": {
        "label": "Diet Type",
        "options": {
            "balanced": "Balanced",
            "non-veg": "Non-Vegetarian",
            "veg": "Vegetarian",
        },
    },
    "stress_level": {
        "label": "Stress Level",
        "options": {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
        },
    },
    "sleep_quality": {
        "label": "Sleep Quality",
        "options": {
            "good": "Good",
            "average": "Average",
            "poor": "Poor",
        },
    },
    "physical_activity_level": {
        "label": "Physical Activity Level",
        "options": {
            "active": "Active",
            "moderate": "Moderate",
            "sedentary": "Sedentary",
        },
    },
    "smoking_alcohol": {
        "label": "Smoking or Alcohol",
        "options": {
            "no": "No",
            "occasional": "Occasional",
            "yes": "Yes",
        },
    },
    "gender": {
        "label": "Gender",
        "options": {
            "male": "Male",
            "female": "Female",
            "other": "Other",
        },
    },
}

NUMERIC_FEATURES = list(NUMERIC_FIELDS.keys())
CATEGORICAL_FEATURES = list(CATEGORICAL_FIELDS.keys())
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LOW_PROBABILITY_THRESHOLD = 60.0
MAX_REQUEST_SIZE_BYTES = 1 * 1024 * 1024
