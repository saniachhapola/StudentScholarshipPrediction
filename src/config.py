"""
Central configuration for the Student Scholarship Prediction System.

This module contains:
- Project directory paths
- Dataset and model file paths
- Feature and target column names
- Model training configuration
- Validation ranges

No preprocessing, model training, or prediction logic should be
written in this file.
"""


from pathlib import Path
from typing import Final



# =========================================================
# PROJECT DIRECTORIES
# =========================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

MODEL_DIR: Final[Path] = PROJECT_ROOT / "model"

APP_DIR: Final[Path] = PROJECT_ROOT / "app"

ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"



# =========================================================
# FILE PATHS
# =========================================================

DATA_FILE_PATH: Final[Path] = (
    DATA_DIR / "student_scholarship.csv"
)


MODEL_FILE_PATH: Final[Path] = (
    MODEL_DIR / "scholarship_model.pkl"
)



# =========================================================
# DATASET CONFIGURATION
# =========================================================


FEATURE_COLUMNS: Final[list[str]] = [

    
    "Year",
    "Category",
    "Percentage",
    "FamilyIncome",
    "Attendance",
    "Backlogs",
    "ExtracurriculumScore",
   

]


TARGET_COLUMN: Final[str] = (

    "ScholarshipAmount"

)



REQUIRED_COLUMNS: Final[list[str]] = [

    "StudentID","Age","Gender",

    *FEATURE_COLUMNS,

    TARGET_COLUMN

]



# =========================================================
# MODEL TRAINING CONFIGURATION
# =========================================================


TEST_SIZE: Final[float] = 0.20


RANDOM_STATE: Final[int] = 42



# =========================================================
# VALIDATION RANGES
# =========================================================


MIN_AGE: Final[int] = 15
MAX_AGE: Final[int] = 30


MIN_YEAR: Final[int] = 1
MAX_YEAR: Final[int] = 4


# General=0, OBC=1, SC=3, ST=4

MIN_CATEGORY: Final[int] = 0
MAX_CATEGORY: Final[int] = 4


MIN_PERCENTAGE: Final[float] = 0.0
MAX_PERCENTAGE: Final[float] = 100.0


MIN_FAMILY_INCOME: Final[int] = 0
MAX_FAMILY_INCOME: Final[int] = 2000000


MIN_ATTENDANCE: Final[float] = 0.0
MAX_ATTENDANCE: Final[float] = 100.0


MIN_BACKLOGS: Final[int] = 0
MAX_BACKLOGS: Final[int] = 10


MIN_EXTRACURRICULUMSCORE: Final[int] = 0
MAX_EXTRACURRICULUMSCORE: Final[int] = 10


#MIN_CERTIFICATES: Final[int] = 0
#MAX_CERTIFICATES: Final[int] = 20



# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================


def create_required_directories() -> None:
    """
    Create project directories if they do not exist.
    """


    directories = [

        DATA_DIR,

        MODEL_DIR,

        APP_DIR,

        ASSETS_DIR,

    ]


    for directory in directories:

        directory.mkdir(

            parents=True,

            exist_ok=True

        )