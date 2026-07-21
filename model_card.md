# Model Card

## Model name

Student Health Risk XGBoost Classifier

## Purpose

The model predicts one of three educational health-condition classes from student health and lifestyle information.

## Output classes

- At-risk
- Fit
- Unhealthy

## Intended use

- Academic machine learning demonstration
- Local Flask application demonstration
- Comparison of classical, ensemble, and deep learning methods

## Unsupported use

- Medical diagnosis
- Clinical decision making
- Automatic treatment recommendations
- Decisions about insurance, education, or employment

## Input features

Numerical inputs

- Sleep duration
- Heart rate
- BMI
- Calorie expenditure
- Step count
- Exercise duration
- Water intake

Categorical inputs

- Diet type
- Stress level
- Sleep quality
- Physical activity level
- Smoking or alcohol
- Gender

## Recorded model evaluation

- Primary metric is balanced accuracy
- Validation balanced accuracy is approximately 0.9097 in the existing comparison
- Macro F1 is approximately 0.7541 in the existing comparison
- An internal holdout result must be generated using `scripts/robust_evaluation_and_export.py`

## Why balanced accuracy is used

The At-risk class is much larger than the Fit and Unhealthy classes. Normal accuracy can therefore look high even when minority classes are predicted poorly. Balanced accuracy gives equal importance to recall for each class.

## Probability statement

The web application displays model probability. The probability is not medical certainty. Probability reliability evidence is generated using calibration plots and a multiclass Brier-style score.

## Fairness review

Performance is evaluated separately for Male, Female, and Other records. The results are saved in

```text
outputs/results/gender_fairness_evaluation.csv
```

Differences between groups must be reported honestly as limitations.

## Privacy

The Flask application does not save submitted values in a database and does not print health values in the log.

## Main limitations

- Synthetic or competition data may not represent real clinical populations.
- Model predictions can be wrong.
- The model accepts only values within the training-data ranges.
- Feature importance does not prove medical causation.
- Results should be reviewed as educational model outputs only.
