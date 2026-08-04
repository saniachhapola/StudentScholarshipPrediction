"""
Streamlit application for the Student Scholarship Prediction System.

Responsibilities:
- Display the project interface
- Accept student scholarship information
- Validate user input
- Call the prediction function
- Display predicted scholarship amount
- Show beginner-friendly error messages

This file does not train or evaluate the Machine Learning model.
"""

import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Add project root directory to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.predict import predict_scholarship_amount  # noqa: E402


# =========================================================
# PAGE CONFIGURATION
# =========================================================

def configure_page() -> None:
    """
    Configure the Streamlit page.
    """

    st.set_page_config(
        page_title="Student Scholarship Prediction",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="expanded",
    )


# =========================================================
# HEADER
# =========================================================

def display_header() -> None:
    """
    Display application title.
    """

    st.title("🎓 Student Scholarship Prediction System")

    st.write(
        "Enter the student's academic details to predict the "
        "estimated scholarship amount."
    )

    st.info(
        "This application uses a trained Linear Regression model "
        "to estimate scholarship amount."
    )


# =========================================================
# SIDEBAR
# =========================================================

def display_sidebar() -> None:
    """
    Display project information.
    """

    with st.sidebar:

        st.header("Project Information")

        st.write(
            """
            **Project:** Student Scholarship Prediction

            **Algorithm:** Linear Regression

            **Framework:** Streamlit

            **Library:** Scikit-learn
            """
        )

        st.divider()

        st.subheader("Input Features")

        st.write(
            """
            - Academic Year
            - Category
            - Percentage
            - Family Income
            - Attendance
            - Number of Backlogs
            - Extracurriculum Score
            """
        )

        st.divider()

        st.caption(
            "Prediction is generated using historical student data."
        )


# =========================================================
# INPUTS
# =========================================================

def get_student_inputs() -> tuple[
    int,
    int,
    float,
    int,
    float,
    int,
    int,
]:
    """
    Display input controls.

    Returns:
        Tuple containing all feature values.
    """

    st.subheader("Student Information")

    left_column, right_column = st.columns(2)

    with left_column:

        year = st.selectbox(
            label="Academic Year",
            options=[1, 2, 3, 4],
            help="Select current academic year.",
        )

        percentage = st.number_input(
            label="Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.1,
            help="Enter student's percentage.",
        )

        attendance = st.number_input(
            label="Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.1,
            help="Enter attendance percentage.",
        )

        extracurricular_score = st.number_input(
            label="Extracurriculum Score",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Enter extracurricular performance score.",
        )

    with right_column:

        category = st.selectbox(
            label="Category",
            options=[
                ("General", 0),
                ("OBC", 1),
                ("SC", 2),
                ("ST", 3),
                ("EWS", 4),
            ],
            format_func=lambda value: value[0],
            help="Select student category.",
        )[1]

        family_income = st.number_input(
            label="Family Income (₹)",
            min_value=0,
            max_value=2000000,
            value=500000,
            step=1000,
            help="Annual family income.",
        )

        backlogs = st.number_input(
            label="Number of Backlogs",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Enter total number of backlogs.",
        )

    return (
        year,
        category,
        percentage,
        family_income,
        attendance,
        backlogs,
        extracurricular_score,
    )
# =========================================================
# INPUT SUMMARY
# =========================================================

def display_input_summary(
    year: int,
    category: int,
    percentage: float,
    family_income: int,
    attendance: float,
    backlogs: int,
    extracurricular_score: int,
) -> None:
    """
    Display the entered student information.

    Args:
        year: Academic year.
        category: Student category.
        percentage: Percentage obtained.
        family_income: Annual family income.
        attendance: Attendance percentage.
        backlogs: Number of backlogs.
        extracurricular_score: Extracurriculum score.
    """

    category_names = {
        0: "General",
        1: "OBC",
        2: "SC",
        3: "ST",
        4: "EWS",
    }

    with st.expander("Review Entered Information"):

        summary_data = {
            "Academic Year": year,
            "Category": category_names.get(category, "Unknown"),
            "Percentage": f"{percentage:.2f} %",
            "Family Income": f"₹ {family_income:,}",
            "Attendance": f"{attendance:.2f} %",
            "Backlogs": backlogs,
            "Extracurriculum Score": extracurricular_score,
        }

        st.table(summary_data)


# =========================================================
# SCHOLARSHIP CATEGORY
# =========================================================

def classify_scholarship(
    scholarship_amount: float,
) -> str:
    """
    Return scholarship category based on
    predicted scholarship amount.

    Args:
        scholarship_amount: Predicted scholarship amount.

    Returns:
        Scholarship category.
    """

    if scholarship_amount >= 75000:
        return "High Scholarship"

    if scholarship_amount >= 50000:
        return "Medium Scholarship"

    if scholarship_amount >= 25000:
        return "Low Scholarship"

    if scholarship_amount > 0:
        return "Minimal Scholarship"

    return "Not Eligible"


# =========================================================
# DISPLAY PREDICTION
# =========================================================

def display_prediction_result(
    scholarship_amount: float,
) -> None:
    """
    Display scholarship prediction.

    Args:
        scholarship_amount:
            Predicted scholarship amount.
    """

    scholarship_category = classify_scholarship(
        scholarship_amount
    )

    st.success(
        "Prediction completed successfully."
    )

    amount_column, category_column = st.columns(2)

    with amount_column:

        st.metric(
            label="Predicted Scholarship",
            value=f"₹ {scholarship_amount:,.2f}",
        )

    with category_column:

        st.metric(
            label="Scholarship Category",
            value=scholarship_category,
        )

    if scholarship_amount <= 0:

        st.warning(
            "The student is not eligible for scholarship "
            "based on the provided information."
        )

    elif scholarship_amount < 25000:

        st.info(
            "The student may receive a small scholarship amount."
        )

    elif scholarship_amount < 50000:

        st.success(
            "The student is eligible for a moderate scholarship."
        )

    else:

        st.success(
            "The student is eligible for a high scholarship."
        )

    st.progress(
        min(
            int(
                scholarship_amount / 1000
            ),
            100,
        )
    )

    #---------------------------------------------
# =========================================================
# MAIN APPLICATION
# =========================================================

def main() -> None:
    """
    Run the Student Scholarship Prediction
    Streamlit application.
    """

    configure_page()

    display_header()

    display_sidebar()

    with st.form("scholarship_prediction_form"):

        (
            year,
            category,
            percentage,
            family_income,
            attendance,
            backlogs,
            extracurricular_score,
        ) = get_student_inputs()

        submit_button = st.form_submit_button(
            label="Predict Scholarship Amount",
            type="primary",
            use_container_width=True,
        )

    display_input_summary(
        year=year,
        category=category,
        percentage=percentage,
        family_income=family_income,
        attendance=attendance,
        backlogs=backlogs,
        extracurricular_score=extracurricular_score,
    )

    if submit_button:

        try:

            # Scholarship eligibility rule
            if backlogs > 0:

                st.warning(
                    "Student has one or more backlogs."
                )

                st.error(
                    "Scholarship Amount: ₹0.00"
                )

                st.info(
                    "Students with backlogs are not eligible for scholarship."
                )

            else:

                with st.spinner(
                    "Predicting scholarship amount..."
                ):

                    predicted_amount = (
                        predict_scholarship_amount(
                            year=year,
                            category=category,
                            percentage=percentage,
                            family_income=family_income,
                            attendance=attendance,
                            backlogs=backlogs,
                            extracurricular_score=(
                                extracurricular_score
                            ),
                        )
                    )

                display_prediction_result(
                    predicted_amount
                )

        except FileNotFoundError as error:

            st.error(str(error))

            st.warning(
                "Train the Machine Learning model "
                "before generating predictions."
            )

            st.code(
                "python src/train_model.py",
                language="bash",
            )

        except (TypeError, ValueError) as error:

            st.error(
                f"Invalid input or prediction error: {error}"
            )

        except Exception as error:

            st.error(
                "An unexpected error occurred while "
                "generating the prediction."
            )

            st.exception(error)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()