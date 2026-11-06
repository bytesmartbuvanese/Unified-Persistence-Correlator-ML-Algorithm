
"""Build predictor-ready feature tables from preprocessed OULAD data.

The pipeline reads ``processed_data/weekly_student_features.csv`` and
``processed_data/student_course_features.csv``, validates their student-course
keys, and derives engagement, synthetic affect, and heuristic hazard
indicators. Weekly calculations preserve the longitudinal rows and use
information available through each observed week; full-course student-course
aggregates describe the complete course and should not be treated as
early-week predictors without a time-aware reconstruction. Outcome labels
remain attached for training and analysis, but are documented as labels rather
than predictor features.

Run ``python feature_engineering.py`` from this project directory after
``preprocess_data.py``. The script writes the engineered weekly and
student-course tables, feature configuration and dictionary CSVs, and a
summary report under ``engineered_data/``. Synthetic affect indicators and
the hazard warning score are derived proxies, not measured emotional states
or trained probabilities.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# UNIFIED PERSISTENCE CORRELATOR
# FEATURE ENGINEERING PIPELINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "engineered_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

WEEKLY_FILE = os.path.join(
    INPUT_DIR, "weekly_student_features.csv"
)

MASTER_FILE = os.path.join(
    INPUT_DIR, "student_course_features.csv"
)

# Composite student-course key
KEYS = [
    "id_student",
    "code_module",
    "code_presentation"
]

# Never use these as predictor features.
TARGET_COLUMNS = [
    "final_result",
    "target_withdrawn",
    "target_completed"
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def log(message):
    """Write a progress message to standard output."""
    print(message)


def load_csv(path, name):
    """Read an input CSV and report its dimensions.

    Parameters
    ----------
    path : str or path-like
        CSV file to read.
    name : str
        Human-readable dataset name used in messages and errors.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    df = pd.read_csv(path, low_memory=False)

    log(
        f"Loaded {name}: "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns"
    )

    return df


def find_column(df, candidates):
    """
    Find the first matching column from a list of candidate names.

    Matching ignores capitalization and surrounding whitespace.

    Parameters
    ----------
    df : pandas.DataFrame
        Table whose column names are inspected.
    candidates : iterable of str
        Names to try in priority order.

    Returns
    -------
    object or None
        Original matching column label, or ``None`` if there is no match.
    """
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def numeric_column(df, candidates, default=0):
    """
    Return a numeric Series from the first matching column.

    Non-numeric values become missing and are filled with ``default``. When
    no candidate exists, a default-valued Series aligned to ``df`` is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table.
    candidates : iterable of str
        Candidate column names in priority order.
    default : scalar, default=0
        Replacement value for missing or unparseable entries.

    Returns
    -------
    pandas.Series
        Numeric values indexed like ``df``.
    """
    col = find_column(df, candidates)

    if col is None:
        return pd.Series(
            default,
            index=df.index,
            dtype="float64"
        )

    return pd.to_numeric(
        df[col], errors="coerce"
    ).fillna(default)


def saturating_scale(series, reference):
    """Map nonnegative activity to ``[0, 1]`` using a fixed scale.

    Missing and negative values are treated as zero; the reference controls
    the scale's saturation point without relying on cohort-wide statistics.
    """
    values = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    return values / (values + reference)


def signed_scale(series, reference):
    """Map signed activity trends to ``[0, 1]`` without cohort statistics.

    Missing values are treated as zero, which maps to the midpoint.
    """
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return 0.5 + 0.5 * np.tanh(values / reference)


def safe_divide(numerator, denominator):
    """Divide aligned values, replacing zero-denominator and infinite results with zero."""
    denominator = denominator.replace(0, np.nan)

    return (
        numerator / denominator
    ).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)


def validate_keys(df, name):
    """Require the configured student-course key columns in a table.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to validate.
    name : str
        Dataset label included in validation errors.

    Raises
    ------
    ValueError
        If any column in ``KEYS`` is absent.
    """
    missing = [
        col for col in KEYS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing key columns: {missing}"
        )


# ============================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================

log("\n" + "=" * 65)
log("STEP 1: LOADING PREPROCESSED DATA")
log("=" * 65)

weekly = load_csv(
    WEEKLY_FILE,
    "Weekly student features"
)

master = load_csv(
    MASTER_FILE,
    "Student-course features"
)

validate_keys(weekly, "Weekly dataset")
validate_keys(master, "Master dataset")


# ============================================================
# STEP 2: SORT WEEKLY DATA
# ============================================================

log("\nSTEP 2: SORTING WEEKLY RECORDS")

week_col = find_column(
    weekly,
    ["week_number", "week", "activity_week"]
)

if week_col is None:
    raise ValueError(
        "No week_number/week column found in weekly dataset. "
        "Check the columns in weekly_student_features.csv."
    )

weekly[week_col] = pd.to_numeric(
    weekly[week_col],
    errors="coerce"
)

weekly = weekly.dropna(
    subset=KEYS + [week_col]
).copy()

weekly[week_col] = weekly[week_col].astype(int)

weekly = weekly.sort_values(
    KEYS + [week_col]
).reset_index(drop=True)

if weekly.duplicated(
    KEYS + [week_col]
).any():
    raise ValueError(
        "Duplicate student-course-week records detected."
    )

log(f"Weekly records available: {len(weekly):,}")


# ============================================================
# STEP 3: IDENTIFY ORIGINAL ENGAGEMENT COLUMNS
# ============================================================

log("\nSTEP 3: IDENTIFYING ENGAGEMENT COLUMNS")

clicks = numeric_column(
    weekly,
    [
        "total_clicks",
        "sum_click",
        "weekly_clicks",
        "clicks"
    ]
)

interactions = numeric_column(
    weekly,
    [
        "interaction_records",
        "interaction_count",
        "total_interactions"
    ]
)

active_days = numeric_column(
    weekly,
    [
        "active_days",
        "weekly_active_days"
    ]
)

cumulative_clicks = numeric_column(
    weekly,
    [
        "cumulative_clicks"
    ],
    default=np.nan
)

cumulative_interactions = numeric_column(
    weekly,
    [
        "cumulative_interactions"
    ],
    default=np.nan
)

cumulative_active_days = numeric_column(
    weekly,
    [
        "cumulative_active_days"
    ],
    default=np.nan
)

previous_clicks = numeric_column(
    weekly,
    [
        "previous_week_clicks",
        "previous_clicks"
    ],
    default=np.nan
)

# Calculate cumulative values if not already present.
grouped = weekly.groupby(
    KEYS, sort=False
)

if cumulative_clicks.isna().all():
    cumulative_clicks = grouped[clicks.name].cumsum() \
        if clicks.name else clicks.groupby(
            [weekly[k] for k in KEYS]
        ).cumsum()

if cumulative_interactions.isna().all():
    cumulative_interactions = interactions.groupby(
        [weekly[k] for k in KEYS]
    ).cumsum()

if cumulative_active_days.isna().all():
    cumulative_active_days = active_days.groupby(
        [weekly[k] for k in KEYS]
    ).cumsum()

# Use the original values if they exist.
weekly["engagement_weekly_clicks"] = clicks
weekly["engagement_interactions"] = interactions
weekly["engagement_active_days"] = active_days

weekly["engagement_cumulative_clicks"] = (
    cumulative_clicks
)

weekly["engagement_cumulative_interactions"] = (
    cumulative_interactions
)

weekly["engagement_cumulative_active_days"] = (
    cumulative_active_days
)


# ============================================================
# STEP 4: ENGAGEMENT FEATURE ENGINEERING
# ============================================================

log("\nSTEP 4: CREATING ENGAGEMENT FEATURES")

grouped = weekly.groupby(
    KEYS, sort=False
)

# Weekly activity intensity.
weekly["engagement_click_intensity"] = np.log1p(
    weekly["engagement_weekly_clicks"].clip(lower=0)
)

# Interaction intensity.
weekly["engagement_interaction_intensity"] = np.log1p(
    weekly["engagement_interactions"].clip(lower=0)
)

# Activity per active day.
weekly["engagement_clicks_per_active_day"] = (
    safe_divide(
        weekly["engagement_weekly_clicks"],
        weekly["engagement_active_days"]
    )
)

# Change in weekly activity.
weekly["engagement_previous_clicks"] = (
    previous_clicks
)

weekly["engagement_click_change"] = (
    grouped["engagement_weekly_clicks"].diff()
)

weekly["engagement_click_change"] = (
    weekly["engagement_click_change"].fillna(0)
)

weekly["engagement_click_change_ratio"] = (
    safe_divide(
        weekly["engagement_click_change"],
        weekly["engagement_previous_clicks"].abs()
    )
)

# The panel has explicit zero-activity calendar weeks, so these windows are
# the current and previous two calendar weeks.
weekly["engagement_rolling_3week_clicks"] = (
    grouped["engagement_weekly_clicks"]
    .transform(
        lambda s: s.rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

# Activity variability: rolling standard deviation of calendar-week clicks.
weekly["engagement_rolling_3week_variability"] = (
    grouped["engagement_weekly_clicks"]
    .transform(
        lambda s: s.rolling(
            window=3,
            min_periods=2
        ).std()
    ).fillna(0)
)

# Activity consistency is the recent share of active calendar weeks.
weekly["engagement_consistency"] = (
    weekly["engagement_weekly_clicks"].gt(0).astype(float)
    .groupby([weekly[key] for key in KEYS], sort=False)
    .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
)

# Cumulative activity per active week.
weekly["engagement_average_clicks_per_week"] = (
    safe_divide(
        weekly["engagement_cumulative_clicks"],
        weekly["cumulative_active_weeks"]
    )
)

# Activity trend based only on information available so far.
weekly["engagement_click_trend"] = (
    grouped["engagement_weekly_clicks"]
    .transform(
        lambda s: s.diff().rolling(
            window=3,
            min_periods=1
        ).mean()
    ).fillna(0)
)

# Fixed-scale activity intensity, without a full-presentation maximum.
weekly["engagement_relative_activity"] = (
    saturating_scale(weekly["engagement_weekly_clicks"], 100)
)

# Active days are capped at seven for weekly normalization.
weekly["engagement_active_day_ratio"] = (
    weekly["engagement_active_days"]
    .clip(0, 7) / 7
)

# A transparent composite engagement score.
weekly["engagement_score"] = (
    0.30 * saturating_scale(weekly["engagement_weekly_clicks"], 100)
    + 0.20 * saturating_scale(weekly["engagement_interactions"], 20)
    + 0.20 * weekly["engagement_active_day_ratio"]
    + 0.15 * saturating_scale(weekly["engagement_cumulative_clicks"], 1000)
    + 0.15 * weekly["engagement_consistency"].clip(0, 1)
).clip(0, 1)


# ============================================================
# STEP 5: SYNTHETIC AFFECT ESTIMATION
# ============================================================

log("\nSTEP 5: GENERATING SYNTHETIC AFFECT ESTIMATES")

"""
IMPORTANT:
OULAD does not contain direct emotional or affective labels.

The following variables are simulated proxies generated from
observable learning activity.

They are NOT actual measurements of student emotions.
They must not be described as clinically or psychologically
validated measurements.

All calculations use current and past activity only.
"""

activity = saturating_scale(weekly["engagement_weekly_clicks"], 100)

consistency = weekly["engagement_consistency"].clip(0, 1)

trend = signed_scale(weekly["engagement_click_trend"], 100)

active_ratio = weekly[
    "engagement_active_day_ratio"
].clip(0, 1)

variability = saturating_scale(
    weekly["engagement_rolling_3week_variability"], 100
)

# Motivation proxy:
# More regular and sustained activity increases this estimate.
weekly["affect_motivation_estimate"] = (
    0.35 * activity
    + 0.35 * consistency
    + 0.30 * active_ratio
).clip(0, 1)

# Confidence proxy:
# Based on activity consistency and non-negative activity trend.
weekly["affect_confidence_estimate"] = (
    0.40 * consistency
    + 0.30 * activity
    + 0.30 * trend
).clip(0, 1)

# Frustration proxy:
# Activity instability and negative changes increase this proxy.
negative_trend = 1 - trend

weekly["affect_frustration_estimate"] = (
    0.50 * variability
    + 0.50 * negative_trend
).clip(0, 1)

# Stress proxy:
# Simulated from unusual activity variability and low consistency.
weekly["affect_stress_estimate"] = (
    0.50 * variability
    + 0.50 * (1 - consistency)
).clip(0, 1)

# Disengagement proxy:
# Lower activity, fewer active days and negative trends increase it.
weekly["affect_disengagement_estimate"] = (
    0.35 * (1 - activity)
    + 0.35 * (1 - active_ratio)
    + 0.30 * negative_trend
).clip(0, 1)

# Overall synthetic affect indicator.
weekly["affect_wellbeing_proxy"] = (
    0.30 * weekly["affect_motivation_estimate"]
    + 0.30 * weekly["affect_confidence_estimate"]
    + 0.20 * (1 - weekly["affect_frustration_estimate"])
    + 0.20 * (1 - weekly["affect_stress_estimate"])
).clip(0, 1)

weekly["affect_data_type"] = "SYNTHETIC_ESTIMATE"


# ============================================================
# STEP 6: HAZARD / WITHDRAWAL-RISK FEATURES
# ============================================================

log("\nSTEP 6: CREATING HAZARD FEATURES")

# Time progression.
weekly["hazard_week_number"] = weekly[week_col]

weekly["hazard_previous_week_clicks"] = (
    weekly["engagement_previous_clicks"]
)

# Declining activity indicator.
weekly["hazard_activity_decline"] = (
    weekly["engagement_click_change"] < 0
).astype(int)

# Consecutive observed weeks with declining clicks.
weekly["hazard_decline_streak"] = (
    weekly.groupby(KEYS, sort=False)[
        "hazard_activity_decline"
    ].transform(
        lambda s: s.groupby(
            (s != s.shift()).cumsum()
        ).cumsum()
    )
)

weekly["hazard_low_activity"] = (
    weekly["engagement_score"] < 0.25
).astype(int)

weekly["hazard_low_consistency"] = (
    weekly["engagement_consistency"] < 0.25
).astype(int)

weekly["hazard_low_active_days"] = (
    weekly["engagement_active_days"] <= 1
).astype(int)

weekly["hazard_activity_drop_ratio"] = (
    -weekly["engagement_click_change_ratio"]
).clip(lower=0)

# Registration duration, if available.
registration_col = find_column(
    weekly,
    ["date_registration"]
)

if registration_col is not None:
    registration = pd.to_numeric(
        weekly[registration_col],
        errors="coerce"
    )

    weekly["hazard_days_since_registration"] = (
        weekly[week_col] * 7 - registration
    ).clip(lower=0)
else:
    weekly["hazard_days_since_registration"] = np.nan

# Transparent early-warning indicator.
# This is a heuristic indicator, not a trained probability.
weekly["hazard_warning_score"] = (
    0.30 * (1 - weekly["engagement_score"])
    + 0.20 * (1 - weekly["engagement_consistency"].clip(0, 1))
    + 0.20 * weekly["hazard_activity_decline"]
    + 0.15 * weekly["hazard_low_active_days"]
    + 0.15 * weekly["affect_disengagement_estimate"]
).clip(0, 1)


# ============================================================
# STEP 7: ATTACH OUTCOME LABELS
# ============================================================

log("\nSTEP 7: ATTACHING WITHDRAWAL LABELS")

for col in TARGET_COLUMNS:
    if col in weekly.columns:
        weekly.drop(
            columns=[col],
            inplace=True
        )

labels = [
    col for col in TARGET_COLUMNS
    if col in master.columns
]

if "target_withdrawn" not in labels:
    raise ValueError(
        "target_withdrawn is missing from master dataset."
    )

label_data = master[
    KEYS + labels
].drop_duplicates(
    subset=KEYS
)

weekly = weekly.merge(
    label_data,
    on=KEYS,
    how="left",
    validate="many_to_one"
)

weekly["target_withdrawn"] = pd.to_numeric(
    weekly["target_withdrawn"],
    errors="coerce"
)

if "target_completed" in weekly.columns:
    weekly["target_completed"] = pd.to_numeric(
        weekly["target_completed"],
        errors="coerce"
    )

log(
    "Weekly records with withdrawal labels: "
    f"{weekly['target_withdrawn'].notna().sum():,}"
)


# ============================================================
# STEP 8: CREATE STUDENT-COURSE FEATURE DATASET
# ============================================================

log("\nSTEP 8: CREATING STUDENT-COURSE FEATURES")

master_engineered = master.copy()

# Numeric conversion for useful assessment features.
assessment_candidates = {
    "average_assessment_score": [
        "average_assessment_score",
        "mean_assessment_score"
    ],
    "maximum_assessment_score": [
        "maximum_assessment_score",
        "max_assessment_score"
    ],
    "minimum_assessment_score": [
        "minimum_assessment_score",
        "min_assessment_score"
    ],
    "average_submission_delay": [
        "average_submission_delay",
        "mean_submission_delay"
    ],
    "assessment_participation_rate": [
        "assessment_participation_rate",
        "participation_rate"
    ]
}

for output_col, candidates in assessment_candidates.items():
    source_col = find_column(
        master_engineered,
        candidates
    )

    if source_col is not None:
        master_engineered[output_col] = pd.to_numeric(
            master_engineered[source_col],
            errors="coerce"
        )

# Assessment availability indicators.
if "average_assessment_score" in master_engineered.columns:
    master_engineered["assessment_score_available"] = (
        master_engineered[
            "average_assessment_score"
        ].notna().astype(int)
    )

if "assessment_participation_rate" in master_engineered.columns:
    master_engineered["assessment_participation_available"] = (
        master_engineered[
            "assessment_participation_rate"
        ].notna().astype(int)
    )

# Course-level engagement aggregates.
weekly_aggregates = weekly.groupby(
    KEYS
).agg(
    total_observed_weekly_clicks=(
        "engagement_weekly_clicks", "sum"
    ),
    mean_weekly_clicks=(
        "engagement_weekly_clicks", "mean"
    ),
    mean_weekly_engagement_score=(
        "engagement_score", "mean"
    ),
    mean_synthetic_motivation=(
        "affect_motivation_estimate", "mean"
    ),
    mean_synthetic_disengagement=(
        "affect_disengagement_estimate", "mean"
    ),
    mean_hazard_warning_score=(
        "hazard_warning_score", "mean"
    ),
    observed_active_weeks=(
        week_col, "nunique"
    )
).reset_index()

master_engineered = master_engineered.merge(
    weekly_aggregates,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

# Preserve missing values in assessment columns.
# Do not treat missing assessment data as a score of zero.


# ============================================================
# STEP 9: PREPARE PREDICTOR COLUMN LISTS
# ============================================================

log("\nSTEP 9: PREPARING MODEL FEATURE LISTS")

# Keys and outcomes are metadata/labels, never predictors.
excluded_columns = (
    KEYS
    + TARGET_COLUMNS
)

weekly_predictors = [
    col for col in weekly.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(weekly[col])
]

master_predictors = [
    col for col in master_engineered.columns
    if col not in excluded_columns
    and pd.api.types.is_numeric_dtype(
        master_engineered[col]
    )
]

# Explicitly remove raw date identifiers and any accidental
# outcome-derived variables.
for forbidden in [
    "date_unregistration",
    "date_registration"
]:
    if forbidden in weekly_predictors:
        weekly_predictors.remove(forbidden)

    if forbidden in master_predictors:
        master_predictors.remove(forbidden)

# These weekly predictors are designed for longitudinal
# prediction. They contain only information available up to
# the current observed week.
#
# The course-level master dataset includes full-course
# assessment and engagement aggregates. Do not use it as an
# early-week predictor dataset without time-aware reconstruction.


# ============================================================
# STEP 10: SAVE OUTPUT DATASETS
# ============================================================

log("\nSTEP 10: SAVING ENGINEERED DATASETS")

weekly_output = os.path.join(
    OUTPUT_DIR,
    "engineered_weekly_features.csv"
)

master_output = os.path.join(
    OUTPUT_DIR,
    "engineered_student_course_features.csv"
)

weekly.to_csv(
    weekly_output,
    index=False
)

master_engineered.to_csv(
    master_output,
    index=False
)

# Save predictor lists.
feature_config = pd.DataFrame({
    "weekly_predictor_features": pd.Series(
        weekly_predictors
    ),
    "student_course_predictor_features": pd.Series(
        master_predictors
    )
})

feature_config.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_config.csv"
    ),
    index=False
)


# ============================================================
# STEP 11: FEATURE DICTIONARY
# ============================================================

feature_descriptions = {
    "engagement_weekly_clicks":
        "Total VLE clicks during the observed week.",

    "engagement_interactions":
        "Number of VLE interaction records during the week.",

    "engagement_active_days":
        "Number of distinct days with VLE activity.",

    "engagement_cumulative_clicks":
        "Cumulative VLE clicks up to the current week.",

    "engagement_cumulative_interactions":
        "Cumulative interactions up to the current week.",

    "engagement_cumulative_active_days":
        "Cumulative active days up to the current week.",

    "engagement_click_intensity":
        "Log-transformed weekly click volume.",

    "engagement_interaction_intensity":
        "Log-transformed weekly interaction volume.",

    "engagement_clicks_per_active_day":
        "Average clicks per active day.",

    "engagement_click_change":
        "Difference between current and previous observed week clicks.",

    "engagement_click_change_ratio":
        "Relative change in weekly clicks.",

    "engagement_rolling_3week_clicks":
        "Rolling average of current and previous observed weekly clicks.",

    "engagement_rolling_3week_variability":
        "Rolling standard deviation of weekly clicks.",

    "engagement_consistency":
        "Activity consistency proxy derived from rolling variability.",

    "engagement_average_clicks_per_week":
        "Cumulative clicks divided by observed weeks.",

    "engagement_click_trend":
        "Recent trend in weekly click activity.",

    "engagement_relative_activity":
        "Relative weekly click activity within a course presentation.",

    "engagement_active_day_ratio":
        "Active days divided by seven.",

    "engagement_score":
        "Composite engagement indicator between zero and one.",

    "affect_motivation_estimate":
        "Synthetic motivation proxy based on activity and consistency.",

    "affect_confidence_estimate":
        "Synthetic confidence proxy based on activity stability.",

    "affect_frustration_estimate":
        "Synthetic frustration proxy based on instability and declining activity.",

    "affect_stress_estimate":
        "Synthetic stress proxy based on variability and inconsistency.",

    "affect_disengagement_estimate":
        "Synthetic disengagement proxy based on low or declining activity.",

    "affect_wellbeing_proxy":
        "Composite synthetic affect indicator.",

    "hazard_week_number":
        "Current observed week number.",

    "hazard_activity_decline":
        "Indicates whether weekly clicks declined.",

    "hazard_decline_streak":
        "Consecutive observed weeks with declining activity.",

    "hazard_low_activity":
        "Indicates whether the engagement score is below the threshold.",

    "hazard_low_consistency":
        "Indicates whether activity consistency is low.",

    "hazard_low_active_days":
        "Indicates one or fewer active days in the week.",

    "hazard_activity_drop_ratio":
        "Magnitude of relative activity decline.",

    "hazard_days_since_registration":
        "Approximate elapsed days since registration, when available.",

    "hazard_warning_score":
        "Heuristic early-warning indicator, not a trained probability."
}

dictionary_rows = []

for feature, description in feature_descriptions.items():
    if feature in weekly.columns:
        if feature.startswith("engagement_"):
            category = "Engagement"
        elif feature.startswith("affect_"):
            category = "Synthetic Affect"
        else:
            category = "Hazard"

        dictionary_rows.append({
            "feature_name": feature,
            "feature_category": category,
            "description": description,
            "data_type": str(weekly[feature].dtype),
            "source": (
                "OULAD-derived"
                if category == "Engagement"
                or category == "Hazard"
                else "Synthetic estimate"
            )
        })

dictionary = pd.DataFrame(
    dictionary_rows
)

dictionary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_dictionary.csv"
    ),
    index=False
)


# ============================================================
# STEP 12: GENERATE SUMMARY REPORT
# ============================================================

log("\nSTEP 12: GENERATING SUMMARY")

summary_path = os.path.join(
    OUTPUT_DIR,
    "feature_engineering_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("=" * 70 + "\n")
    file.write("UNIFIED PERSISTENCE CORRELATOR\n")
    file.write("FEATURE ENGINEERING SUMMARY\n")
    file.write("=" * 70 + "\n\n")

    file.write(
        f"Weekly dataset shape: {weekly.shape}\n"
    )

    file.write(
        f"Student-course dataset shape: "
        f"{master_engineered.shape}\n\n"
    )

    file.write("ENGAGEMENT FEATURES\n")
    file.write("-" * 50 + "\n")

    for col in weekly.columns:
        if col.startswith("engagement_"):
            file.write(col + "\n")

    file.write("\nSYNTHETIC AFFECT FEATURES\n")
    file.write("-" * 50 + "\n")

    for col in weekly.columns:
        if col.startswith("affect_"):
            file.write(col + "\n")

    file.write("\nHAZARD FEATURES\n")
    file.write("-" * 50 + "\n")

    for col in weekly.columns:
        if col.startswith("hazard_"):
            file.write(col + "\n")

    file.write("\nMODEL PREDICTOR FEATURES\n")
    file.write("-" * 50 + "\n")

    file.write(
        f"Weekly predictor count: "
        f"{len(weekly_predictors)}\n"
    )

    for col in weekly_predictors:
        file.write(col + "\n")

    file.write(
        f"\nStudent-course predictor count: "
        f"{len(master_predictors)}\n"
    )

    for col in master_predictors:
        file.write(col + "\n")

    file.write("\nTARGET DISTRIBUTION\n")
    file.write("-" * 50 + "\n")

    if "target_withdrawn" in weekly.columns:
        file.write(
            "Withdrawal labels:\n"
        )

        file.write(
            weekly["target_withdrawn"]
            .value_counts(dropna=False)
            .to_string()
        )

    file.write("\n\nIMPORTANT NOTES\n")
    file.write("-" * 50 + "\n")

    file.write(
        "1. final_result, target_withdrawn and "
        "target_completed are labels, not predictors.\n"
    )

    file.write(
        "2. Affect variables are synthetic estimates, "
        "not actual emotional measurements.\n"
    )

    file.write(
        "3. hazard_warning_score is a heuristic indicator, "
        "not a trained probability.\n"
    )

    file.write(
        "4. Weekly features use current and past observed "
        "activity only.\n"
    )

    file.write(
        "5. Student-course aggregates include full-course "
        "information and must not be used for early-week "
        "prediction without time-aware reconstruction.\n"
    )

    file.write(
        "6. Weekly rows from the same student-course must "
        "remain in the same train/test group.\n"
    )


# ============================================================
# STEP 13: FINAL VALIDATION
# ============================================================

log("\n" + "=" * 65)
log("FINAL FEATURE ENGINEERING VALIDATION")
log("=" * 65)

log(
    f"Weekly engineered dataset: "
    f"{weekly.shape}"
)

log(
    f"Student-course engineered dataset: "
    f"{master_engineered.shape}"
)

log(
    f"Engagement features: "
    f"{sum(c.startswith('engagement_') for c in weekly.columns)}"
)

log(
    f"Synthetic affect features: "
    f"{sum(c.startswith('affect_') for c in weekly.columns)}"
)

log(
    f"Hazard features: "
    f"{sum(c.startswith('hazard_') for c in weekly.columns)}"
)

log(
    f"Weekly predictor features: "
    f"{len(weekly_predictors)}"
)

log(
    f"Student-course predictor features: "
    f"{len(master_predictors)}"
)

assert not any(
    col in weekly_predictors
    for col in TARGET_COLUMNS
)

assert not any(
    col in master_predictors
    for col in TARGET_COLUMNS
)

log("\nAll target-leakage checks passed.")

log("\nGenerated files:")
log("1. engineered_weekly_features.csv")
log("2. engineered_student_course_features.csv")
log("3. feature_config.csv")
log("4. feature_dictionary.csv")
log("5. feature_engineering_summary.txt")

log("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY.")
