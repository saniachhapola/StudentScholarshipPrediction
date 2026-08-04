"""
Prediction module for the Student Scholarship Prediction System.

Responsibilities:
- Validate user input
- Load the saved Linear Regression model
- Convert student input into the correct feature format
- Predict scholarship amount
- Return a clean numeric prediction

This file does not train or evaluate the model.
"""

from functools import lru_cache

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from src.config import (
        FEATURE_COLUMNS,
        MODEL_FILE_PATH,
        MIN_YEAR,
        MAX_YEAR,
        MIN_CATEGORY,
        MAX_CATEGORY,
        MIN_PERCENTAGE,
        MAX_PERCENTAGE,
        MIN_FAMILY_INCOME,
        MAX_FAMILY_INCOME,
        MIN_ATTENDANCE,
        MAX_ATTENDANCE,
        MIN_BACKLOGS,
        MAX_BACKLOGS,
        MIN_EXTRACURRICULUMSCORE,
        MAX_EXTRACURRICULUMSCORE,
    )
except ModuleNotFoundError:
    from config import (
        FEATURE_COLUMNS,
        MODEL_FILE_PATH,
        MIN_YEAR,
        MAX_YEAR,
        MIN_CATEGORY,
        MAX_CATEGORY,
        MIN_PERCENTAGE,
        MAX_PERCENTAGE,
        MIN_FAMILY_INCOME,
        MAX_FAMILY_INCOME,
        MIN_ATTENDANCE,
        MAX_ATTENDANCE,
        MIN_BACKLOGS,
        MAX_BACKLOGS,
        MIN_EXTRACURRICULUMSCORE,
        MAX_EXTRACURRICULUMSCORE,
    )


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@lru_cache(maxsize=1)
def load_trained_model() -> LinearRegression:
    """
    Load the trained Linear Regression model from disk.

    The model is cached after the first load so that repeated
    predictions do not read the model file every time.

    Returns:
        Trained Linear Regression model.

    Raises:
        FileNotFoundError:
            If the model file does not exist.

        ValueError:
            If the model cannot be loaded.
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
            "The saved file does not contain a valid "
            "LinearRegression model."
        )

    return model


# =========================================================
# VALIDATE NUMERIC VALUE
# =========================================================

def validate_numeric_value(
    value: float,
    field_name: str,
    minimum_value: float,
    maximum_value: float,
) -> float:
    """
    Validate a numeric input value.

    Args:
        value:
            User input value.

        field_name:
            Name of the input field.

        minimum_value:
            Minimum allowed value.

        maximum_value:
            Maximum allowed value.

    Returns:
        Validated numeric value.

    Raises:
        TypeError:
            If the value is not numeric.

        ValueError:
            If the value is outside
            the allowed range.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field_name} must be a numeric value."
        )

    numeric_value = float(value)

    if not minimum_value <= numeric_value <= maximum_value:
        raise ValueError(
            f"{field_name} must be between "
            f"{minimum_value} and {maximum_value}."
        )

    return numeric_value

# =========================================================
# VALIDATE STUDENT INPUTS
# =========================================================

def validate_student_inputs(
    year: int,
    category: int,
    percentage: float,
    family_income: float,
    attendance: float,
    backlogs: int,
    extracurricular_score: int,
) -> dict[str, float]:
    """
    Validate all student input values.

    Args:
        year: Academic year.
        category: Student category.
        percentage: Percentage obtained.
        family_income: Annual family income.
        attendance: Attendance percentage.
        backlogs: Number of backlogs.
        extracurricular_score: Extracurriculum score.

    Returns:
        Dictionary containing validated input values.
    """

    return {
        "Year": validate_numeric_value(
            year,
            "Year",
            MIN_YEAR,
            MAX_YEAR,
        ),
        "Category": validate_numeric_value(
            category,
            "Category",
            MIN_CATEGORY,
            MAX_CATEGORY,
        ),
        "Percentage": validate_numeric_value(
            percentage,
            "Percentage",
            MIN_PERCENTAGE,
            MAX_PERCENTAGE,
        ),
        "FamilyIncome": validate_numeric_value(
            family_income,
            "Family Income",
            MIN_FAMILY_INCOME,
            MAX_FAMILY_INCOME,
        ),
        "Attendance": validate_numeric_value(
            attendance,
            "Attendance",
            MIN_ATTENDANCE,
            MAX_ATTENDANCE,
        ),
        "Backlogs": validate_numeric_value(
            backlogs,
            "Backlogs",
            MIN_BACKLOGS,
            MAX_BACKLOGS,
        ),
        "ExtracurriculumScore": validate_numeric_value(
            extracurricular_score,
            "Extracurriculum Score",
            MIN_EXTRACURRICULUMSCORE,
            MAX_EXTRACURRICULUMSCORE,
        ),
    }


# =========================================================
# CREATE PREDICTION DATAFRAME
# =========================================================

def create_prediction_dataframe(
    student_data: dict[str, float],
) -> pd.DataFrame:
    """
    Convert validated student information into
    a one-row DataFrame.

    The feature order must match the order used
    during model training.

    Args:
        student_data:
            Dictionary containing validated inputs.

    Returns:
        One-row DataFrame ready for prediction.
    """

    return pd.DataFrame(
        [
            [
                student_data[column]
                for column in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS,
    )


# =========================================================
# LIMIT PREDICTION
# =========================================================

def limit_prediction_to_valid_range(
    predicted_amount: float,
) -> float:
    """
    Ensure scholarship amount is never negative.

    Linear Regression may predict values below zero.
    Such predictions are converted to zero.

    Args:
        predicted_amount:
            Raw prediction from the model.

    Returns:
        Non-negative scholarship amount.
    """

    return max(
        0.0,
        float(predicted_amount),
    )
# =========================================================
# PREDICT SCHOLARSHIP AMOUNT
# =========================================================

def predict_scholarship_amount(
    year: int,
    category: int,
    percentage: float,
    family_income: float,
    attendance: float,
    backlogs: int,
    extracurricular_score: int,
) -> float:
    """
    Predict the scholarship amount for a student.

    Steps:
    1. Validate all user inputs.
    2. Convert inputs into a one-row DataFrame.
    3. Load the trained model.
    4. Generate scholarship prediction.
    5. Ensure prediction is non-negative.
    6. Round the prediction to two decimal places.

    Args:
        year: Academic year.
        category: Student category.
        percentage: Percentage obtained.
        family_income: Annual family income.
        attendance: Attendance percentage.
        backlogs: Number of backlogs.
        extracurricular_score: Extracurriculum score.

    Returns:
        Predicted scholarship amount.
    """

    student_data = validate_student_inputs(
        year=year,
        category=category,
        percentage=percentage,
        family_income=family_income,
        attendance=attendance,
        backlogs=backlogs,
        extracurricular_score=extracurricular_score,
    )

    prediction_input = create_prediction_dataframe(
        student_data
    )

    model = load_trained_model()

    predicted_value = model.predict(
        prediction_input
    )[0]

    final_prediction = limit_prediction_to_valid_range(
        predicted_value
    )

    return round(
        final_prediction,
        2,
    )


# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    try:

        predicted_amount = predict_scholarship_amount(
            year=2,
            category=1,
            percentage=82.5,
            family_income=450000,
            attendance=90.0,
            backlogs=0,
            extracurricular_score=8,
        )

        print(
            "Student Scholarship Prediction"
        )

        print("-" * 40)

        print(
            f"Predicted Scholarship Amount: "
            f"₹{predicted_amount:,.2f}"
        )

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
    ) as error:

        print(
            f"Prediction failed: {error}"
        )