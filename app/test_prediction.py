"""Simple manual check for the three known student records."""

from services.predictor import predict_health_condition

TEST_RECORDS = [
    (
        "Fit test",
        "fit",
        {
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
        },
    ),
    (
        "At-risk test",
        "at-risk",
        {
            "sleep_duration": "6.79",
            "heart_rate": "77.5",
            "bmi": "26.49",
            "calorie_expenditure": "2734",
            "step_count": "14751",
            "exercise_duration": "42.9",
            "water_intake": "2.42",
            "diet_type": "non-veg",
            "stress_level": "low",
            "sleep_quality": "average",
            "physical_activity_level": "active",
            "smoking_alcohol": "no",
            "gender": "other",
        },
    ),
    (
        "Unhealthy test",
        "unhealthy",
        {
            "sleep_duration": "4.66",
            "heart_rate": "73.2",
            "bmi": "31.07",
            "calorie_expenditure": "2125",
            "step_count": "12496",
            "exercise_duration": "48.9",
            "water_intake": "2.12",
            "diet_type": "veg",
            "stress_level": "high",
            "sleep_quality": "poor",
            "physical_activity_level": "moderate",
            "smoking_alcohol": "yes",
            "gender": "other",
        },
    ),
]


for test_name, expected_class, record in TEST_RECORDS:
    result = predict_health_condition(record)
    actual_class = result["prediction"]
    test_passed = actual_class == expected_class

    print(test_name)
    print("Expected", expected_class)
    print("Actual", actual_class)
    print("Model probability", result["model_probability"])
    print("Passed", test_passed)
    print()
