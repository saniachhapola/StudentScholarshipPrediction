"""
Data preprocessing module for the Scholarship Eligibility Prediction System.

Responsibilities:
- Load the scholarship eligibility dataset
- Validate required columns
- Standardize column names
- Remove duplicate records
- Convert numeric columns
- Encode categorical values
- Handle missing values
- Validate acceptable value ranges
- Separate features and target variable

This file does not train or evaluate the Machine Learning model.
"""


from pathlib import Path

import pandas as pd


try:
    from src.config import (
        DATA_FILE_PATH,
        FEATURE_COLUMNS,
        REQUIRED_COLUMNS,
        TARGET_COLUMN,
        MIN_PERCENTAGE,
        MAX_PERCENTAGE,
        MIN_ATTENDANCE,
        MAX_ATTENDANCE,
        MIN_EXTRACURRICULUMSCORE,
        MAX_EXTRACURRICULUMSCORE,
        MIN_YEAR,
        MAX_YEAR,
        MIN_CATEGORY,
        MAX_CATEGORY,
        MIN_AGE,
        MAX_AGE,
    )

except ModuleNotFoundError:

    from config import (
        DATA_FILE_PATH,
        FEATURE_COLUMNS,
        REQUIRED_COLUMNS,
        TARGET_COLUMN,
        MIN_PERCENTAGE,
        MAX_PERCENTAGE,
        MIN_ATTENDANCE,
        MAX_ATTENDANCE,
        MIN_EXTRACURRICULUMSCORE,
        MAX_EXTRACURRICULUMSCORE,
        MIN_YEAR,
        MAX_YEAR,
        MIN_CATEGORY,
        MAX_CATEGORY,
        MIN_AGE,
        MAX_AGE,
    )



# =========================================================
# COLUMN NAME ALIASES
# =========================================================


COLUMN_NAME_ALIASES = {

    "studentid": "StudentID",
    "age": "Age",
    "gender": "Gender",
    "year": "Year",
    "category": "Category",
    "percentage": "Percentage",
    "familyincome": "FamilyIncome",
    "income": "FamilyIncome",
    "attendance": "Attendance",
    "backlogs": "Backlogs",
    "extracurriculumscore": "ExtracurriculumScore",

   
    "scholarshipamount": "ScholarshipAmount",

}



def _normalize_column_key(column_name: str) -> str:
    """
    Create lowercase compact key for column matching.
    """

    return "".join(
        character.lower()
        for character in str(column_name)
        if character.isalnum()
    )

# =========================================================
# STANDARDIZE COLUMN NAMES
# =========================================================


def standardize_column_names(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Rename different column formats into standard names.
    """
    dataframe = dataframe.copy()
    renamed_columns = {}

    for column in dataframe.columns:
        normalized = _normalize_column_key(column)

        if normalized in COLUMN_NAME_ALIASES:
            required_name = COLUMN_NAME_ALIASES[normalized]

            if column != required_name:
                renamed_columns[column] = required_name

    if renamed_columns:
        dataframe = dataframe.rename(
            columns=renamed_columns
        )
    return dataframe


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(
    file_path: Path = DATA_FILE_PATH
) -> pd.DataFrame:
    """
    Load CSV dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    try:
        dataframe = pd.read_csv(file_path)

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "Dataset file is empty."
        ) from error

    except pd.errors.ParserError as error:
        raise ValueError(
            "Invalid CSV format."
        ) from error

    if dataframe.empty:
        raise ValueError(
            "Dataset contains no records."

        )

    return dataframe

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

def validate_required_columns(
    dataframe: pd.DataFrame
) -> None:
    """
    Check required columns exist.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicate_records(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove duplicate student records.
    """
    return dataframe.drop_duplicates().copy()

# =========================================================
# CONVERT NUMERIC COLUMNS
# =========================================================

def convert_numeric_columns(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert numeric columns into numeric datatype.
    """
    dataframe = dataframe.copy()

    numeric_columns = [
        "StudentID",
        "Age",
        "Year",
        "Category",
        "Percentage",
        "FamilyIncome",
        "Attendance",
        "Backlogs",
        "ExtracurriculumScore",
      
        "ScholarshipAmount"
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

    return dataframe

# =========================================================
# ENCODE GENDER
# =========================================================

def encode_gender(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert gender into numeric values.

    Male = 1
    Female = 0
    """
    dataframe = dataframe.copy()

    dataframe["Gender"] = (
        dataframe["Gender"]
        .astype(str)
        .str.lower()
        .map(
            {

                "male": 1,

                "female": 0
            }
        )
    )
    return dataframe

# ========================================================
# HANDLE MISSING VALUES
# =========================================================

def handle_missing_values(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace missing values with median.
    """
    dataframe = dataframe.copy()

    for column in REQUIRED_COLUMNS:
        if dataframe[column].isna().all():
            raise ValueError(
                f"{column} has no valid values."
            )
        dataframe[column] = dataframe[column].fillna(
            dataframe[column].median()
        )

    return dataframe

# =========================================================
# VALIDATE DATA RANGE
# =========================================================

def validate_data_ranges(
    dataframe: pd.DataFrame
) -> None:
    """
    Validate input ranges.
    """
    validations = [
        (
            "Age",
            MIN_AGE,
            MAX_AGE
        ),
        (
            "Year",
            MIN_YEAR,
            MAX_YEAR
        ),

        (
            "Category",
            MIN_CATEGORY,
            MAX_CATEGORY
        ),

        (
            "Percentage",
            MIN_PERCENTAGE,
            MAX_PERCENTAGE
        ),

        (
            "Attendance",
            MIN_ATTENDANCE,
            MAX_ATTENDANCE
        ),

        (
            "ExtracurriculumScore",
            MIN_EXTRACURRICULUMSCORE,
            MAX_EXTRACURRICULUMSCORE
        )

    ]

    for column, minimum, maximum in validations:

        if not dataframe[column].between(
            minimum,
            maximum
        ).all():
            raise ValueError(
                f"{column} must be between "
                f"{minimum} and {maximum}"
            )
        if (dataframe["ScholarshipAmount"] < 0).any():
            raise ValueError(
                "Scholarship amount cannot be negative."
            )

# =========================================================
# COMPLETE PREPROCESSING PIPELINE
# =========================================================

def preprocess_dataset(
    file_path: Path = DATA_FILE_PATH
) -> pd.DataFrame:
    """
    Execute complete preprocessing workflow.
    """
    dataframe = load_dataset(file_path)
    dataframe = standardize_column_names(
        dataframe
    )
    validate_required_columns(
        dataframe
    )
    dataframe = dataframe[
        REQUIRED_COLUMNS
    ].copy()

    dataframe = remove_duplicate_records(
        dataframe
    )

    dataframe = convert_numeric_columns(
        dataframe
    )

    dataframe = encode_gender(
        dataframe
    )

    dataframe = handle_missing_values(
        dataframe
    )

    validate_data_ranges(
        dataframe
    )

    dataframe.reset_index(
        drop=True,
        inplace=True
    )
    return dataframe

# =========================================================
# FEATURES AND TARGET
# =========================================================

def prepare_features_and_target(
    dataframe: pd.DataFrame
):
    """
    Separate features and target.
    """

    X = dataframe[
        FEATURE_COLUMNS
    ].copy()

    y = dataframe[
        TARGET_COLUMN
    ].copy()

    return X, y

def get_preprocessed_data(
    file_path: Path = DATA_FILE_PATH
):
    """
    Main function called by training module.
    """

    dataframe = preprocess_dataset(
        file_path
    )

    return prepare_features_and_target(
        dataframe
    )

# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":
    try:
        clean_data = preprocess_dataset()

        print(
            "Dataset preprocessing completed successfully."
        )

        print(
            f"Total records: {len(clean_data)}"
        )

        print(
            f"Total columns: {len(clean_data.columns)}"
        )

        print("\nFirst five records:")
        print(
            clean_data.head()
        )

    except (
        FileNotFoundError,
        ValueError
    ) as error:

        print(
            f"Preprocessing failed: {error}"
        )