# Student Health Risk Predictor

## Project overview

Student Health Risk Predictor is a Flask web application developed for the CIS 6005 Computational Intelligence assignment. The system uses a trained XGBoost classification model to predict a student's health condition from lifestyle and health-related input values.

The application predicts one of the following classes.

- `fit`
- `at-risk`
- `unhealthy`

## Kaggle competition

- Competition  Playground Series, Season 6 Episode 7, Predicting Student Health Risk
- Problem type  Supervised multiclass classification
- Target column  `health_condition`
- Evaluation metric  Balanced Accuracy Score

## Project objectives

- Explore and understand the competition dataset using exploratory data analysis
- Prepare numerical and categorical features for machine learning
- Train and compare multiple classification models
- Tune and evaluate the selected XGBoost model
- Save the trained preprocessing pipeline and label encoder
- Integrate the trained model into a working Flask web application
- Validate new input values before prediction
- Display the predicted class and model probabilities
- Test model reliability, fairness, and prediction performance

## Main application features

- Responsive student health prediction form
- Server-side numerical range validation
- Server-side category validation
- Rejection of missing, NaN, and infinite values
- Exact model feature order checking
- Saved model availability checking
- User-friendly validation and prediction errors
- CSRF protection for the prediction form
- Prediction probability for each class
- Low-probability result warning
- Dynamic model metadata display
- Application health status endpoint
- Automated tests using pytest
- Prediction speed benchmarking

## Input features

The prediction form uses the following features.

- Sleep duration
- Heart rate
- Body Mass Index
- Calorie expenditure
- Step count
- Exercise duration
- Water intake
- Diet type
- Stress level
- Sleep quality
- Physical activity level
- Smoking or alcohol use
- Gender

## Technologies used

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- XGBoost
- joblib
- HTML
- CSS
- JavaScript
- pytest

## Project structure

```text
student-health-project
├── app
├── data
│   └── kaggle
│       └── raw
├── models
├── notebooks
├── outputs
├── scripts
├── tests
├── requirements.txt
├── requirements-notebooks.txt
└── pytest.ini
```

## Installation

Open the project folder in VS Code and create a virtual environment.

```bash
python -m venv .venv
```

Activate the virtual environment on Windows.

```bash
.venv\Scripts\activate
```

Install the application dependencies.

```bash
pip install -r requirements.txt
```

Install the notebook dependencies when running the model development notebooks.

```bash
pip install -r requirements-notebooks.txt
```

The saved model was created using scikit-learn 1.9.0. The same version should be used to avoid model compatibility issues.

## Running the Flask application

Set a local Flask secret key.

```bash
set FLASK_SECRET_KEY=my-local-secret-key
```

Start the application from the project root.

```bash
python app/app.py
```

Open the application in a browser.

```text
http://127.0.0.1:5000
```

The application health status is available at the following address.

```text
http://127.0.0.1:5000/health
```

## Model development and evaluation

The project includes scripts for the main machine learning development and evaluation stages.

### Exploratory data analysis corrections

```bash
python scripts/eda_corrections.py
```

### XGBoost hyperparameter tuning

```bash
python scripts/tune_xgboost.py
```

### Cross-validation of the strongest tree-based models

```bash
python scripts/cross_validate_top_models.py
```

### Internal holdout, fairness, reliability, and model export

```bash
python scripts/robust_evaluation_and_export.py
```

The evaluation script can save a deployment candidate using the following files.

```text
models/final_model_candidate.joblib
models/final_label_encoder_candidate.joblib
```

After the candidate model is evaluated, the script can export the final deployment files.

```text
models/final_model.joblib
models/final_label_encoder.joblib
```

### Prediction speed benchmark

```bash
python scripts/benchmark_predictions.py
```

## Testing

Run all automated tests from the project root.

```bash
pytest
```

Run the three manual prediction checks.

```bash
python app/test_prediction.py
```

The tests check examples for all three output classes.

- Fit test returns `fit`
- At-risk test returns `at-risk`
- Unhealthy test returns `unhealthy`

## Data privacy

- The application does not use a database
- Submitted prediction values are not stored
- Submitted health values are not printed in the server log
- Each result is generated only for the current request

## Limitations

- Predictions are limited to the input ranges and categories supported by the trained model
- Model performance depends on the quality and representativeness of the competition dataset
- The model may produce incorrect classifications for some records
- Performance across demographic groups should be reviewed using the saved fairness evaluation results
- Future versions can be improved using additional representative data, further tuning, and comparison with deep learning methods

