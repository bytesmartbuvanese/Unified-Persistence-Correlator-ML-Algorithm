
"""
UNIFIED PERSISTENCE CORRELATOR
OULAD DATA PREPROCESSING PIPELINE

This script validates and cleans the source tables, aggregates the large VLE
activity file in chunks, and produces both longitudinal weekly and
student-course feature tables. Outcome fields are retained for analysis but
must be excluded from predictors.

Input:
    raw_data/

Output:
    processed_data/
        cleaned_student_info.csv
        cleaned_student_registration.csv
        cleaned_courses.csv
        cleaned_assessments.csv
        cleaned_student_assessment.csv
        cleaned_vle.csv
        student_course_features.csv
        weekly_student_features.csv
        preprocessing_summary.txt

The large studentVle.csv is processed in chunks.
SQLite is used to combine daily activity across chunks accurately.

Run ``python preprocess_data.py`` from the project directory after placing
the seven OULAD source CSVs listed in ``REQUIRED_FILES`` in ``raw_data/``.
The workflow cleans student, registration, course, assessment, and VLE
records; aggregates VLE activity into weekly student-course observations;
joins registration, course, and assessment information; validates the
resulting keys and feature tables; and records progress in
``preprocessing_summary.txt``. Its weekly output is intended for longitudinal
feature engineering. The student-course table contains full-course
aggregations and therefore is not, by itself, a time-safe early-week
prediction input.
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "processed_data"

OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 200_000

STUDENT_KEY = [
    "id_student",
    "code_module",
    "code_presentation"
]

REQUIRED_FILES = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "courses.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "studentVle.csv",
    "vle.csv"
]


# ============================================================
# UTILITIES
# ============================================================

def log(message):
    """Print a progress message and append it to the run summary."""
    print(message)
    SUMMARY.append(str(message))


SUMMARY = []


def save_csv(df, filename):
    """Write a dataframe under the configured output directory.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to serialize.
    filename : str
        Output filename relative to ``OUTPUT_DIR``.

    Returns
    -------
    pathlib.Path
        Path of the written CSV file.
    """

    path = OUTPUT_DIR / filename

    df.to_csv(path, index=False)

    log(f"Saved: {filename} | Shape: {df.shape}")

    return path


def read_csv(filename):
    """Read one required CSV from the raw-data directory.

    Parameters
    ----------
    filename : str
        Input filename relative to ``RAW_DIR``.

    Returns
    -------
    pandas.DataFrame
        Loaded source table.

    Raises
    ------
    FileNotFoundError
        If the requested source file is absent.
    """

    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def replace_unknowns(df):
    """Replace the configured textual missing-value markers with ``NaN``."""

    return df.replace(
        ["?", "NA", "N/A", "NULL", ""],
        np.nan
    )


def numeric_columns(df, columns):
    """Convert present columns to numeric values, coercing invalid entries."""

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def validate_unique(df, keys, name):
    """Log duplicate-key counts and reject non-unique records.

    Raises
    ------
    ValueError
        If more than one row has the same values for ``keys``.
    """

    duplicates = df.duplicated(keys).sum()

    log(
        f"{name}: duplicate key records = {duplicates:,}"
    )

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate records for {keys}. "
            "Resolve these before merging."
        )


def validate_foreign_key(left, right, key, name):
    """Report distinct keys in ``left`` that do not occur in ``right``.

    Parameters
    ----------
    left, right : pandas.DataFrame
        Tables to compare.
    key : str or list of str
        Column name or composite key used for the comparison.
    name : str
        Relationship label included in the progress message.

    Returns
    -------
    pandas.MultiIndex
        Distinct unmatched keys from the left-hand table.
    """
    keys = [key] if isinstance(key, str) else key

    left_keys = pd.MultiIndex.from_frame(
        left[keys].drop_duplicates()
    )

    right_keys = pd.MultiIndex.from_frame(
        right[keys].drop_duplicates()
    )

    unmatched = left_keys.difference(right_keys)

    log(
        f"{name}: "
        f"unmatched keys = {len(unmatched):,}"
    )

    return unmatched


# ============================================================
# 1. STUDENT INFORMATION
# ============================================================

def preprocess_student_info():
    """Clean student attributes and derive binary outcome labels.

    Preserves ``final_result`` as an outcome and derives separate withdrawal
    and successful-completion targets before saving the cleaned table.

    Returns
    -------
    pandas.DataFrame
        Validated student-course records with outcome labels.
    """

    log("\nSTEP 1: STUDENT INFORMATION")

    df = replace_unknowns(
        read_csv("studentInfo.csv")
    )

    df["id_student"] = pd.to_numeric(
        df["id_student"],
        errors="raise"
    ).astype("int64")

    df["num_of_prev_attempts"] = pd.to_numeric(
        df["num_of_prev_attempts"],
        errors="coerce"
    )

    df["studied_credits"] = pd.to_numeric(
        df["studied_credits"],
        errors="coerce"
    )

    # Preserve final_result as the outcome label.
    df["final_result"] = df["final_result"].astype("string").str.strip()

    allowed_results = [
        "Pass",
        "Distinction",
        "Fail",
        "Withdrawn"
    ]

    invalid = ~df["final_result"].isin(allowed_results)

    if invalid.any():
        raise ValueError(
            f"Unexpected final_result values: "
            f"{df.loc[invalid, 'final_result'].unique()}"
        )

    validate_unique(df, STUDENT_KEY, "studentInfo")

    # Binary withdrawal event.
    df["target_withdrawn"] = (
        df["final_result"] == "Withdrawn"
    ).astype("int8")

    # Completion is a separate outcome, not the same as
    # withdrawal prediction.
    df["target_completed"] = (
        df["final_result"].isin(["Pass", "Distinction"])
    ).astype("int8")

    save_csv(df, "cleaned_student_info.csv")

    return df


# ============================================================
# 2. STUDENT REGISTRATION
# ============================================================

def preprocess_registration():
    """Clean registration dates and record unregistration-date availability.

    Returns
    -------
    pandas.DataFrame
        Unique registration records keyed by student and course presentation.
    """

    log("\nSTEP 2: STUDENT REGISTRATION")

    df = replace_unknowns(
        read_csv("studentRegistration.csv")
    )

    df["id_student"] = pd.to_numeric(
        df["id_student"],
        errors="raise"
    ).astype("int64")

    df = numeric_columns(
        df,
        [
            "date_registration",
            "date_unregistration"
        ]
    )

    validate_unique(df, STUDENT_KEY, "studentRegistration")

    # Missing unregistration dates mean no recorded
    # unregistration date, not zero.
    df["has_unregistration_date"] = (
        df["date_unregistration"].notna()
    ).astype("int8")

    save_csv(df, "cleaned_student_registration.csv")

    return df


# ============================================================
# 3. COURSE INFORMATION
# ============================================================

def preprocess_courses():
    """Clean course-presentation metadata and validate its composite key.

    Returns
    -------
    pandas.DataFrame
        Course metadata with numeric presentation lengths.
    """

    log("\nSTEP 3: COURSE INFORMATION")

    df = read_csv("courses.csv")

    df["module_presentation_length"] = pd.to_numeric(
        df["module_presentation_length"],
        errors="coerce"
    )

    validate_unique(
        df,
        ["code_module", "code_presentation"],
        "courses"
    )

    save_csv(df, "cleaned_courses.csv")

    return df


# ============================================================
# 4. ASSESSMENTS
# ============================================================

def preprocess_assessments():
    """Clean assessment metadata and validate assessment identifiers.

    Returns
    -------
    pandas.DataFrame
        Assessment records with numeric dates and weights.
    """

    log("\nSTEP 4: ASSESSMENTS")

    df = replace_unknowns(
        read_csv("assessments.csv")
    )

    df = numeric_columns(
        df,
        [
            "id_assessment",
            "date",
            "weight"
        ]
    )

    df["id_assessment"] = df["id_assessment"].astype("int64")

    validate_unique(
        df,
        ["id_assessment"],
        "assessments"
    )

    save_csv(df, "cleaned_assessments.csv")

    return df


# ============================================================
# 5. STUDENT ASSESSMENT
# ============================================================

def preprocess_student_assessment(assessments):
    """Clean student submissions and attach assessment metadata.

    Submission delay is computed relative to the assessment date; out-of-range
    scores are marked missing, and banked status is retained separately.

    Parameters
    ----------
    assessments : pandas.DataFrame
        Cleaned assessment metadata used for the many-to-one join.

    Returns
    -------
    pandas.DataFrame
        Student submission records enriched with assessment attributes.
    """

    log("\nSTEP 5: STUDENT ASSESSMENT")

    df = replace_unknowns(
        read_csv("studentAssessment.csv")
    )

    df = numeric_columns(
        df,
        [
            "id_assessment",
            "id_student",
            "date_submitted",
            "is_banked",
            "score"
        ]
    )

    df["id_student"] = df["id_student"].astype("int64")

    # Preserve original submission date and score.
    validate_foreign_key(
        df,
        assessments,
        "id_assessment",
        "studentAssessment -> assessments"
    )

    # Add assessment metadata.
    df = df.merge(
        assessments[
            [
                "id_assessment",
                "code_module",
                "code_presentation",
                "assessment_type",
                "date",
                "weight"
            ]
        ],
        on="id_assessment",
        how="left",
        validate="many_to_one"
    )

    # Submission delay: positive means late.
    df["submission_delay_days"] = (
        df["date_submitted"] - df["date"]
    )

    df.loc[
        df["date"].isna(),
        "submission_delay_days"
    ] = np.nan

    # Scores are on the OULAD 0-100 scale.
    df.loc[
        ~df["score"].between(0, 100),
        "score"
    ] = np.nan

    # Preserve banked assessments as a separate feature.
    df["is_banked"] = df["is_banked"].fillna(0).astype("int8")

    save_csv(df, "cleaned_student_assessment.csv")

    return df


# ============================================================
# 6. VLE METADATA
# ============================================================

def preprocess_vle():
    """Clean VLE site metadata and validate site/presentation uniqueness.

    Returns
    -------
    pandas.DataFrame
        VLE metadata with numeric week bounds.
    """

    log("\nSTEP 6: VLE METADATA")

    df = replace_unknowns(
        read_csv("vle.csv")
    )

    df["id_site"] = pd.to_numeric(
        df["id_site"],
        errors="raise"
    ).astype("int64")

    df = numeric_columns(
        df,
        [
            "week_from",
            "week_to"
        ]
    )

    validate_unique(
        df,
        ["id_site", "code_module", "code_presentation"],
        "vle"
    )

    save_csv(df, "cleaned_vle.csv")

    return df


# ============================================================
# 7. LARGE STUDENT VLE PROCESSING
# ============================================================

def process_student_vle():
    """Aggregate the large student VLE log into weekly activity records.

    Source rows are read in chunks, combined by student/course/day in SQLite,
    and then reduced to week-level click, interaction, and active-day totals.
    The database connection is closed even when processing fails.

    Returns
    -------
    pandas.DataFrame
        One row per student-course-week with aggregated activity counts.

    Raises
    ------
    FileNotFoundError
        If the raw VLE activity file is missing.
    """

    log("\nSTEP 7: PROCESSING STUDENT VLE")

    source = RAW_DIR / "studentVle.csv"

    if not source.exists():
        raise FileNotFoundError(source)

    db_path = OUTPUT_DIR / "vle_daily_aggregation.sqlite"

    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)

    connection.execute("""
        CREATE TABLE daily_activity (
            code_module TEXT NOT NULL,
            code_presentation TEXT NOT NULL,
            id_student INTEGER NOT NULL,
            activity_date INTEGER NOT NULL,
            total_clicks INTEGER NOT NULL,
            interaction_records INTEGER NOT NULL,
            PRIMARY KEY (
                code_module,
                code_presentation,
                id_student,
                activity_date
            )
        )
    """)

    upsert_sql = """
        INSERT INTO daily_activity (
            code_module,
            code_presentation,
            id_student,
            activity_date,
            total_clicks,
            interaction_records
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (
            code_module,
            code_presentation,
            id_student,
            activity_date
        )
        DO UPDATE SET
            total_clicks =
                total_clicks + excluded.total_clicks,
            interaction_records =
                interaction_records + excluded.interaction_records
    """

    total_rows = 0

    reader = pd.read_csv(
        source,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8-sig"
    )

    try:

        for chunk_number, chunk in enumerate(reader, 1):

            chunk = replace_unknowns(chunk)

            chunk["id_student"] = pd.to_numeric(
                chunk["id_student"],
                errors="raise"
            ).astype("int64")

            chunk["date"] = pd.to_numeric(
                chunk["date"],
                errors="raise"
            ).astype("int64")

            chunk["sum_click"] = pd.to_numeric(
                chunk["sum_click"],
                errors="raise"
            ).astype("int64")

            # Preserve every activity record.
            # Do not remove repeated-looking events.
            daily = (
                chunk.groupby(
                    [
                        "code_module",
                        "code_presentation",
                        "id_student",
                        "date"
                    ],
                    as_index=False,
                    sort=False
                )
                .agg(
                    total_clicks=("sum_click", "sum"),
                    interaction_records=("sum_click", "size")
                )
            )

            records = [
                (
                    str(row.code_module),
                    str(row.code_presentation),
                    int(row.id_student),
                    int(row.date),
                    int(row.total_clicks),
                    int(row.interaction_records)
                )
                for row in daily.itertuples(index=False)
            ]

            connection.executemany(upsert_sql, records)
            connection.commit()

            total_rows += len(chunk)

            log(
                f"VLE chunk {chunk_number}: "
                f"{total_rows:,} source records processed"
            )

        # Convert daily data into weekly data.
        # Negative dates are handled using floor-style week bins.
        weekly_sql = """
            SELECT
                id_student,
                code_module,
                code_presentation,

                CASE
                    WHEN activity_date < 0
                    THEN CAST((activity_date - 6) / 7 AS INTEGER)
                    ELSE CAST(activity_date / 7 AS INTEGER)
                END AS week_number,

                SUM(total_clicks) AS weekly_total_clicks,

                SUM(interaction_records)
                    AS weekly_interaction_records,

                COUNT(*) AS weekly_active_days

            FROM daily_activity

            GROUP BY
                id_student,
                code_module,
                code_presentation,
                week_number

            ORDER BY
                id_student,
                code_module,
                code_presentation,
                week_number
        """

        weekly = pd.read_sql_query(
            weekly_sql,
            connection
        )

    finally:
        connection.close()

    log(
        f"Total source VLE records processed: {total_rows:,}"
    )

    log(
        f"Weekly student records generated: {len(weekly):,}"
    )

    return weekly


# ============================================================
# 8. WEEKLY LONGITUDINAL FEATURES
# ============================================================

def create_weekly_features(
    weekly,
    students,
    registration,
    courses,
):
    """Construct a complete longitudinal panel with causal weekly features.

    The roster and course duration define calendar weeks, including weeks
    without VLE activity. Cumulative and prior-week fields are computed before
    static student attributes and outcome labels are joined.

    Parameters
    ----------
    weekly : pandas.DataFrame
        Aggregated activity records by student-course-week.
    students : pandas.DataFrame
        Cleaned student attributes and outcomes.
    registration : pandas.DataFrame
        Registration and unregistration dates.
    courses : pandas.DataFrame
        Course-presentation lengths.

    Returns
    -------
    pandas.DataFrame
        Complete weekly panel saved as ``weekly_student_features.csv``.
    """
    log("\nSTEP 8: CREATING WEEKLY FEATURES")

    # Build a complete calendar-week panel for every enrollment. VLE files
    # only contain observed events, so a left join alone would otherwise
    # remove students with no clicks and skip inactive weeks.
    course_weeks = courses[
        ["code_module", "code_presentation", "module_presentation_length"]
    ].copy()
    course_weeks["last_course_week"] = (
        np.ceil(course_weeks["module_presentation_length"] / 7).astype(int) - 1
    )

    roster = students[STUDENT_KEY].merge(
        registration[STUDENT_KEY + ["date_registration", "date_unregistration"]],
        on=STUDENT_KEY,
        how="left",
        validate="one_to_one",
    ).merge(
        course_weeks,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )

    observed_span = weekly.groupby(STUDENT_KEY).agg(
        first_observed_week=("week_number", "min"),
        last_observed_week=("week_number", "max"),
    ).reset_index()
    roster = roster.merge(
        observed_span,
        on=STUDENT_KEY,
        how="left",
        validate="one_to_one",
    )
    roster["first_week"] = np.minimum(
        roster["first_observed_week"].fillna(0), 0
    ).astype(int)
    roster["last_week"] = np.maximum(
        roster["last_course_week"].fillna(0),
        roster["last_observed_week"].fillna(roster["last_course_week"]).fillna(0),
    ).astype(int)

    week_counts = (roster["last_week"] - roster["first_week"] + 1).clip(lower=1)
    calendar = roster.loc[roster.index.repeat(week_counts.to_numpy())].copy().reset_index(drop=True)
    calendar["week_number"] = np.concatenate([
        np.arange(start, end + 1, dtype=np.int16)
        for start, end in zip(
            roster["first_week"].to_numpy(),
            roster["last_week"].to_numpy(),
        )
    ])
    weekly = calendar.merge(
        weekly[
            STUDENT_KEY + [
                "week_number",
                "weekly_total_clicks",
                "weekly_interaction_records",
                "weekly_active_days",
            ]
        ],
        on=STUDENT_KEY + ["week_number"],
        how="left",
        validate="one_to_one",
    )
    activity_columns = [
        "weekly_total_clicks",
        "weekly_interaction_records",
        "weekly_active_days",
    ]
    weekly[activity_columns] = weekly[activity_columns].fillna(0)
    weekly[activity_columns] = weekly[activity_columns].astype("int64")
    weekly.drop(
        columns=[
            "module_presentation_length",
            "last_course_week",
            "first_observed_week",
            "last_observed_week",
            "first_week",
            "last_week",
        ],
        inplace=True,
    )
    weekly = weekly.sort_values(
        STUDENT_KEY + ["week_number"]
    ).reset_index(drop=True)

    # Cumulative information available up to the current week.
    group_keys = STUDENT_KEY

    weekly["cumulative_clicks"] = (
        weekly.groupby(group_keys)["weekly_total_clicks"].cumsum()
    )

    weekly["cumulative_interactions"] = (
        weekly.groupby(group_keys)[
            "weekly_interaction_records"
        ].cumsum()
    )

    weekly["cumulative_active_days"] = (
        weekly.groupby(group_keys)["weekly_active_days"].cumsum()
    )

    weekly["active_week_flag"] = (
        weekly["weekly_active_days"] > 0
    ).astype("int8")
    weekly["cumulative_active_weeks"] = (
        weekly.groupby(group_keys)["active_week_flag"].cumsum()
    )

    weekly["average_clicks_per_active_week"] = (
        weekly["cumulative_clicks"]
        / weekly["cumulative_active_weeks"].replace(0, np.nan)
    )
    weekly["average_clicks_per_active_week"] = (
        weekly["average_clicks_per_active_week"].fillna(0)
    )

    weekly["previous_week_clicks"] = (
        weekly.groupby(group_keys)["weekly_total_clicks"]
        .shift(1)
    )

    weekly["click_change_from_previous_week"] = (
        weekly["weekly_total_clicks"]
        - weekly["previous_week_clicks"]
    )

    weekly["previous_week_clicks"] = (
        weekly["previous_week_clicks"].fillna(0)
    )

    weekly["click_change_from_previous_week"] = (
        weekly["click_change_from_previous_week"].fillna(0)
    )

    # Join only static student information and labels.
    static_columns = STUDENT_KEY + [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
        "target_withdrawn",
        "target_completed"
    ]

    weekly = weekly.merge(
        students[static_columns],
        on=STUDENT_KEY,
        how="left",
        validate="many_to_one"
    )

    # Time since registration is useful for longitudinal analysis.
    weekly["days_since_registration"] = (
        weekly["week_number"] * 7
        - weekly["date_registration"]
    )

    # Do not use final_result or target columns as model features.
    save_csv(weekly, "weekly_student_features.csv")

    return weekly


# ============================================================
# 9. STUDENT-COURSE FEATURE DATASET
# ============================================================

def create_student_course_features(
    students,
    registration,
    courses,
    weekly,
    assessment_data
):
    """Combine static attributes, full-course engagement, and assessments.

    The aggregate represents the whole student-course record and therefore
    should not be used as an early-week predictor without time-aware
    reconstruction.

    Parameters
    ----------
    students : pandas.DataFrame
        Cleaned student attributes and outcome labels.
    registration : pandas.DataFrame
        Registration dates keyed by student and course presentation.
    courses : pandas.DataFrame
        Course-presentation metadata.
    weekly : pandas.DataFrame
        Enriched weekly records used for course-level engagement totals.
    assessment_data : pandas.DataFrame
        Cleaned student-assessment submissions.

    Returns
    -------
    pandas.DataFrame
        Unique student-course feature table saved to disk.
    """
    log("\nSTEP 9: CREATING STUDENT-COURSE FEATURES")

    # --------------------------------------------------------
    # Engagement aggregation
    # --------------------------------------------------------

    engagement = (
        weekly.groupby(
            STUDENT_KEY,
            as_index=False
        )
        .agg(
            total_clicks=("weekly_total_clicks", "sum"),
            total_interactions=(
                "weekly_interaction_records", "sum"
            ),
            total_active_days=("weekly_active_days", "sum"),
            active_weeks=("active_week_flag", "sum"),
            average_weekly_clicks=(
                "weekly_total_clicks", "mean"
            ),
            maximum_weekly_clicks=(
                "weekly_total_clicks", "max"
            )
        )
    )

    # --------------------------------------------------------
    # Academic assessment aggregation
    # --------------------------------------------------------

    assessment_data = assessment_data.copy()

    assessment_data["scored_assessment"] = (
        assessment_data["score"].notna().astype("int8")
    )

    assessment_data["late_submission"] = (
        assessment_data["submission_delay_days"] > 0
    ).astype("int8")

    assessment_features = (
        assessment_data.groupby(
            STUDENT_KEY,
            as_index=False
        )
        .agg(
            assessment_count=("id_assessment", "nunique"),
            scored_assessment_count=(
                "scored_assessment", "sum"
            ),
            average_assessment_score=("score", "mean"),
            maximum_assessment_score=("score", "max"),
            minimum_assessment_score=("score", "min"),
            average_submission_delay=(
                "submission_delay_days", "mean"
            ),
            late_submission_count=("late_submission", "sum"),
            banked_assessment_count=("is_banked", "sum")
        )
    )

    # --------------------------------------------------------
    # Merge static information
    # --------------------------------------------------------

    master = students.merge(
        registration,
        on=STUDENT_KEY,
        how="left",
        suffixes=("", "_registration"),
        validate="one_to_one"
    )

    master = master.merge(
        courses,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one"
    )

    master = master.merge(
        engagement,
        on=STUDENT_KEY,
        how="left",
        validate="one_to_one"
    )

    master = master.merge(
        assessment_features,
        on=STUDENT_KEY,
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Students with no recorded VLE activity
    # --------------------------------------------------------

    engagement_columns = [
        "total_clicks",
        "total_interactions",
        "total_active_days",
        "active_weeks",
        "average_weekly_clicks",
        "maximum_weekly_clicks"
    ]

    master[engagement_columns] = (
        master[engagement_columns].fillna(0)
    )

    # No recorded assessment is not the same as a zero score.
    count_columns = [
        "assessment_count",
        "scored_assessment_count",
        "late_submission_count",
        "banked_assessment_count"
    ]

    master[count_columns] = (
        master[count_columns].fillna(0)
    )

    master["assessment_participation_rate"] = np.where(
        master["assessment_count"] > 0,
        master["scored_assessment_count"]
        / master["assessment_count"],
        np.nan
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validate_unique(
        master,
        STUDENT_KEY,
        "student_course_features"
    )

    log(
        f"Student-course records: {len(master):,}"
    )

    log(
        f"Student-course feature count: {len(master.columns)}"
    )

    save_csv(
        master,
        "student_course_features.csv"
    )

    return master


# ============================================================
# 10. FINAL DATA VALIDATION
# ============================================================

def final_validation(master, weekly):
    """Log dataset sizes, duplicate counts, outcomes, and missing values.

    This reporting step explicitly reminds users that outcome fields are
    labels rather than predictor features.
    """
    log("\nSTEP 10: FINAL VALIDATION")

    log(f"Master dataset shape: {master.shape}")
    log(f"Weekly dataset shape: {weekly.shape}")

    log(
        f"Master exact duplicate rows: "
        f"{master.duplicated().sum():,}"
    )

    log(
        f"Weekly exact duplicate rows: "
        f"{weekly.duplicated().sum():,}"
    )

    log("\nFINAL RESULT DISTRIBUTION")

    distribution = master["final_result"].value_counts(
        dropna=False
    )

    for label, count in distribution.items():

        log(
            f"{label}: {count:,} "
            f"({count / len(master) * 100:.2f}%)"
        )

    log("\nMISSING VALUES IN MASTER DATASET")

    missing = master.isna().sum()

    for column, count in missing.items():

        if count:

            log(
                f"{column}: {count:,} "
                f"({count / len(master) * 100:.2f}%)"
            )

    log("\nTARGET VALIDATION")

    log(
        f"Withdrawn target positives: "
        f"{master['target_withdrawn'].sum():,}"
    )

    log(
        f"Completed target positives: "
        f"{master['target_completed'].sum():,}"
    )

    # Verify target leakage fields are not accidentally
    # removed or confused with predictors.
    log(
        "IMPORTANT: final_result, target_withdrawn and "
        "target_completed are outcome fields. "
        "Exclude them from predictor features."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Run all preprocessing stages and save the summary report.

    Required source files are checked first; smaller tables are processed
    before the chunked VLE activity log and derived feature tables.
    """
    log("=" * 75)
    log("UNIFIED PERSISTENCE CORRELATOR")
    log("OULAD PREPROCESSING PIPELINE")
    log("=" * 75)

    log(f"Raw data folder: {RAW_DIR}")
    log(f"Output folder: {OUTPUT_DIR}")

    # Check all required files.
    for filename in REQUIRED_FILES:

        if not (RAW_DIR / filename).exists():

            raise FileNotFoundError(
                f"Required file not found: {filename}"
            )

    # Process smaller tables first.
    students = preprocess_student_info()

    registration = preprocess_registration()

    courses = preprocess_courses()

    assessments = preprocess_assessments()

    student_assessment = preprocess_student_assessment(
        assessments
    )

    vle_metadata = preprocess_vle()

    # Stream the large VLE file.
    weekly = process_student_vle()

    # Validate student activity against the student roster.
    student_keys = students[STUDENT_KEY].drop_duplicates()

    validate_foreign_key(
        weekly,
        student_keys,
        STUDENT_KEY,
        "Weekly activity -> studentInfo"
    )

    # Create weekly longitudinal records.
    weekly = create_weekly_features(
        weekly,
        students,
        registration,
        courses,
    )

    # Create one-row-per-student-course dataset.
    master = create_student_course_features(
        students,
        registration,
        courses,
        weekly,
        student_assessment
    )

    final_validation(
        master,
        weekly
    )

    # Save processing report.
    summary_path = OUTPUT_DIR / "preprocessing_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(SUMMARY))

    log("\nPREPROCESSING COMPLETED SUCCESSFULLY")

    print("\n" + "=" * 75)
    print("PREPROCESSING COMPLETED")
    print("=" * 75)

    print(f"Output directory: {OUTPUT_DIR}")
    print("Primary output: student_course_features.csv")
    print("Longitudinal output: weekly_student_features.csv")
    print("Summary: preprocessing_summary.txt")


if __name__ == "__main__":
    main()
