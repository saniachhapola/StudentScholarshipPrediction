"""
Model evaluation module for the Student Scholarship Prediction System.

Responsibilities:
- Load and preprocess the dataset
- Split the dataset using the same configuration as training
- Load the saved Linear Regression model
- Generate predictions for the testing dataset
- Calculate evaluation metrics
- Display actual and predicted scholarship amounts

This file does not train or save the model.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    from src.config import MODEL_FILE_PATH
except ModuleNotFoundError:
    from config import MODEL_FILE_PATH

try:
    from src.preprocess import get_preprocessed_data
except ModuleNotFoundError:
    from preprocess import get_preprocessed_data

try:
    from src.train_model import split_dataset
except ModuleNotFoundError:
    from train_model import split_dataset


# =========================================================
# EVALUATION RESULT
# =========================================================

@dataclass(frozen=True)
class EvaluationResult:
    """
    Store the calculated model evaluation metrics.

    Attributes:
        mean_absolute_error:
            Average absolute prediction error.

        mean_squared_error:
            Average squared prediction error.

        root_mean_squared_error:
            Square root of mean squared error.

        r2_score:
            Coefficient of determination.
    """

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float


# =========================================================
# LOAD MODEL
# =========================================================

def load_model() -> LinearRegression:
    """
    Load the trained Linear Regression model.

    Returns:
        Trained LinearRegression model.

    Raises:
        FileNotFoundError:
            If model file does not exist.

        ValueError:
            If model cannot be loaded.
    """

    if not MODEL_FILE_PATH.exists():
        raise FileNotFoundError(
            "The trained model file was not found at: "
            f"{MODEL_FILE_PATH}. "
            "Run src/train_model.py first."
        )

    try:
        model = joblib.load(MODEL_FILE_PATH)

    except Exception as error:
        raise ValueError(
            f"Unable to load the trained model: {error}"
        ) from error

    if not isinstance(model, LinearRegression):
        raise ValueError(
            "The saved model is not a "
            "LinearRegression model."
        )

    return model


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def generate_predictions(
    model: LinearRegression,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate scholarship amount predictions.

    Args:
        model:
            Trained Linear Regression model.

        X_test:
            Testing feature dataset.

    Returns:
        NumPy array containing predicted
        scholarship amounts.
    """

    predictions = model.predict(X_test)

    return np.asarray(
        predictions,
        dtype=float,
    )
# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> EvaluationResult:
    """
    Calculate regression evaluation metrics.

    Metrics:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score

    Args:
        y_test:
            Actual scholarship amounts.

        predictions:
            Predicted scholarship amounts.

    Returns:
        EvaluationResult containing all
        calculated metrics.
    """

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    return EvaluationResult(
        mean_absolute_error=float(mae),
        mean_squared_error=float(mse),
        root_mean_squared_error=float(rmse),
        r2_score=float(r2),
    )


# =========================================================
# COMPARISON TABLE
# =========================================================

def create_comparison_table(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Create a comparison table containing
    actual and predicted scholarship amounts.

    Args:
        y_test:
            Actual scholarship amounts.

        predictions:
            Predicted scholarship amounts.

    Returns:
        Comparison DataFrame.
    """

    actual_values = y_test.reset_index(
        drop=True,
    )

    comparison = pd.DataFrame(
        {
            "ActualScholarshipAmount":
                actual_values,

            "PredictedScholarshipAmount":
                np.round(
                    predictions,
                    2,
                ),
        }
    )

    comparison["AbsoluteError"] = np.round(
        np.abs(
            comparison["ActualScholarshipAmount"]
            - comparison[
                "PredictedScholarshipAmount"
            ]
        ),
        2,
    )

    return comparison


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_evaluation_results(
    result: EvaluationResult,
) -> None:
    """
    Display evaluation metrics.

    Args:
        result:
            Evaluation metrics object.
    """

    print("\nModel Evaluation Results")

    print("-" * 50)

    print(
        "Mean Absolute Error (MAE)  : "
        f"{result.mean_absolute_error:.4f}"
    )

    print(
        "Mean Squared Error (MSE)   : "
        f"{result.mean_squared_error:.4f}"
    )

    print(
        "Root Mean Squared Error    : "
        f"{result.root_mean_squared_error:.4f}"
    )

    print(
        "R² Score                   : "
        f"{result.r2_score:.4f}"
    )


# =========================================================
# INTERPRET R² SCORE
# =========================================================

def interpret_r2_score(
    score: float,
) -> str:
    """
    Return a simple interpretation
    of the R² score.

    Args:
        score:
            Calculated R² score.

    Returns:
        Human-readable interpretation.
    """

    if score >= 0.90:
        return "Excellent model performance."

    if score >= 0.75:
        return "Good model performance."

    if score >= 0.50:
        return "Moderate model performance."

    if score >= 0.00:
        return (
            "Weak model performance. "
            "Improvement is required."
        )

    return (
        "Poor model performance. "
        "The model performs worse than "
        "predicting the average target value."
    )
# =========================================================
# EXECUTE EVALUATION PIPELINE
# =========================================================

def execute_evaluation_pipeline() -> tuple[
    EvaluationResult,
    pd.DataFrame,
]:
    """
    Execute the complete model evaluation workflow.

    Steps:
    1. Load and preprocess the dataset.
    2. Split the dataset.
    3. Load the trained model.
    4. Generate predictions.
    5. Calculate evaluation metrics.
    6. Create comparison table.

    Returns:
        Tuple containing:
        - EvaluationResult
        - Comparison DataFrame
    """

    features, target = get_preprocessed_data()

    _, X_test, _, y_test = split_dataset(
        features,
        target,
    )

    model = load_model()

    predictions = generate_predictions(
        model,
        X_test,
    )

    evaluation_result = calculate_metrics(
        y_test,
        predictions,
    )

    comparison_table = create_comparison_table(
        y_test,
        predictions,
    )

    return (
        evaluation_result,
        comparison_table,
    )


# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    try:

        (
            evaluation_result,
            comparison_table,
        ) = execute_evaluation_pipeline()

        display_evaluation_results(
            evaluation_result,
        )

        print(
            "\nPerformance Interpretation: "
            f"{interpret_r2_score(evaluation_result.r2_score)}"
        )

        print(
            "\nActual vs Predicted Scholarship Amount"
        )

        print("-" * 60)

        print(
            comparison_table.to_string(
                index=False,
            )
        )

        print(
            "\nModel evaluation completed successfully."
        )

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as error:

        print(
            f"Model evaluation failed: {error}"
        )