
"""Generate synthetic longitudinal records for the ``drop`` model workflow.

The script creates student-level characteristics and repeated monthly
observations, combines the generated engagement, academic, financial, and
support signals into persistence-related scores and a binary
``Persistence_Next_Term`` target, then writes the rows to
``dataset/unified_persistence_dataset.csv``. The default population size,
number of monthly observations, and random seed are controlled by
``N_STUDENTS``, ``N_MONTHS``, and ``RANDOM_SEED`` near the top of the file.

Run ``python generate_synthetic_data.py`` from the ``drop`` directory before
training with ``python train.py``. This is simulated data for exercising the
model pipeline; it is not a source of real student records or evidence about
real-world predictive performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================================
# UNIFIED PERSISTENCE CORRELATOR
# Synthetic Dataset Generator
# ==========================================================

# Configuration
N_STUDENTS = 10000
N_MONTHS = 6
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Signal transformation helpers
# ---------------------------------------------------------------------------

def clip(x, minimum, maximum):
    """Bound numeric values to an inclusive interval.

    Parameters
    ----------
    x : array-like
        Values to bound.
    minimum : float
        Lower bound.
    maximum : float
        Upper bound.

    Returns
    -------
    numpy.ndarray
        Values clipped to the requested interval.
    """
    return np.clip(x, minimum, maximum)


def sigmoid(x):
    """Map logits to probability-scale values.

    Parameters
    ----------
    x : array-like
        Unnormalized scores.

    Returns
    -------
    numpy.ndarray
        Elementwise logistic transform of ``x``.
    """
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# Longitudinal dataset generation
# ---------------------------------------------------------------------------

def generate_dataset():
    """Build, validate, and save synthetic student-month observations.

    Stable student-level traits are combined with month-specific variation to
    generate engagement, academic, financial, and affective signals. The
    module's global random generator and configuration constants control the
    output population and reproducibility.

    Returns
    -------
    pandas.DataFrame
        Generated records sorted by student and observation date.

    Side Effects
    ------------
    Writes the generated dataset to
    ``dataset/unified_persistence_dataset.csv`` and prints a summary of the
    resulting records.
    """

    records = []

    # ------------------------------------------------------
    # 1. GENERATE STUDENT PROFILES
    # ------------------------------------------------------

    student_ids = [
        f"STU{i:05d}"
        for i in range(1, N_STUDENTS + 1)
    ]

    # Stable individual characteristics
    engagement_trait = rng.normal(0, 1, N_STUDENTS)
    academic_trait = rng.normal(0, 1, N_STUDENTS)
    financial_trait = rng.normal(0, 1, N_STUDENTS)
    satisfaction_trait = rng.normal(0, 1, N_STUDENTS)

    base_gpa = clip(
        6.5 + academic_trait * 0.9
        + rng.normal(0, 0.5, N_STUDENTS),
        3, 10
    )

    tuition_fee = clip(
        rng.normal(85000, 18000, N_STUDENTS),
        35000, 160000
    )

    scholarship_amount = np.where(
        rng.random(N_STUDENTS) < 0.35,
        rng.uniform(10000, 60000, N_STUDENTS),
        0
    )

    # ------------------------------------------------------
    # 2. GENERATE SIX MONTHLY OBSERVATIONS
    # ------------------------------------------------------

    dates = pd.date_range(
        start="2026-01-01",
        periods=N_MONTHS,
        freq="MS"
    )

    for month_index, date in enumerate(dates):

        time_factor = month_index / (N_MONTHS - 1)

        # Monthly variations
        engagement_noise = rng.normal(0, 0.35, N_STUDENTS)
        academic_noise = rng.normal(0, 0.30, N_STUDENTS)
        financial_noise = rng.normal(0, 0.25, N_STUDENTS)

        # --------------------------------------------------
        # FEATURE 1: LMS LOGIN COUNT
        # --------------------------------------------------

        engagement = (
            engagement_trait * 0.55
            + engagement_noise
            - 0.12 * time_factor
        )

        lms_login_count = clip(
            35 + engagement * 13
            + rng.normal(0, 9, N_STUDENTS),
            0, 100
        ).astype(int)

        # --------------------------------------------------
        # FEATURE 2: WEEKLY ACTIVE HOURS
        # --------------------------------------------------

        weekly_active_hours = clip(
            10 + engagement * 3.5
            + rng.normal(0, 2.5, N_STUDENTS),
            0, 35
        )

        # --------------------------------------------------
        # FEATURE 3: ASSIGNMENT SUBMISSION RATE
        # --------------------------------------------------

        assignment_submission_rate = clip(
            0.78 + engagement * 0.12
            + rng.normal(0, 0.08, N_STUDENTS),
            0, 1
        )

        # --------------------------------------------------
        # FEATURE 4: LECTURE ATTENDANCE RATE
        # --------------------------------------------------

        lecture_attendance_rate = clip(
            75 + engagement * 17
            + rng.normal(0, 7, N_STUDENTS),
            20, 100
        )

        # --------------------------------------------------
        # FEATURE 5: CURRENT GPA
        # --------------------------------------------------

        current_gpa = clip(
            base_gpa
            + academic_noise * 0.6
            - 0.15 * time_factor
            + rng.normal(0, 0.25, N_STUDENTS),
            0, 10
        )

        # --------------------------------------------------
        # FEATURE 6: FAILED SUBJECT COUNT
        # --------------------------------------------------

        failed_subject_count = clip(
            np.round(
                1.5 - current_gpa * 0.17
                + rng.normal(0, 0.8, N_STUDENTS)
            ),
            0, 8
        ).astype(int)

        # --------------------------------------------------
        # FEATURE 7: TUITION FEE
        # --------------------------------------------------

        monthly_tuition_fee = tuition_fee.copy()

        # --------------------------------------------------
        # FEATURE 8: OUTSTANDING FEE
        # --------------------------------------------------

        fee_payment_delay = clip(
            15 - financial_trait * 8
            + financial_noise * 5
            + rng.normal(0, 8, N_STUDENTS),
            0, 120
        )

        outstanding_fee = clip(
            monthly_tuition_fee * (
                0.15
                + fee_payment_delay / 150
                - scholarship_amount /
                np.maximum(monthly_tuition_fee, 1) * 0.2
                + rng.normal(0, 0.08, N_STUDENTS)
            ),
            0, monthly_tuition_fee
        )

        # --------------------------------------------------
        # FEATURE 9: SCHOLARSHIP AMOUNT
        # --------------------------------------------------

        monthly_scholarship = scholarship_amount.copy()

        # --------------------------------------------------
        # FEATURE 10: FEEDBACK SENTIMENT SCORE
        # Range: -1 to +1
        # --------------------------------------------------

        satisfaction = clip(
            65 + satisfaction_trait * 12
            + engagement * 5
            + rng.normal(0, 10, N_STUDENTS),
            0, 100
        )

        feedback_sentiment_score = clip(
            (satisfaction - 50) / 50
            + rng.normal(0, 0.15, N_STUDENTS),
            -1, 1
        )

        # --------------------------------------------------
        # FEATURE 11: ACADEMIC STRESS SURVEY SCORE
        # Range: 0 to 100
        # --------------------------------------------------

        academic_risk_proxy = clip(
            100 - current_gpa * 10,
            0, 100
        )

        financial_risk_proxy = clip(
            outstanding_fee /
            np.maximum(monthly_tuition_fee, 1) * 100,
            0, 100
        )

        academic_stress_score = clip(
            50
            + academic_risk_proxy * 0.25
            + financial_risk_proxy * 0.15
            - satisfaction * 0.15
            + rng.normal(0, 12, N_STUDENTS),
            0, 100
        )

        # --------------------------------------------------
        # FEATURE 12: SUPPORT REQUEST COUNT
        # --------------------------------------------------

        support_request_count = clip(
            np.round(
                1
                + academic_risk_proxy / 35
                + financial_risk_proxy / 60
                + rng.normal(0, 1.2, N_STUDENTS)
            ),
            0, 15
        ).astype(int)

        # --------------------------------------------------
        # FEATURE 13: COUNSELING SESSION COUNT
        # Restricted institutional data in real life.
        # Synthetic values represent authorized records.
        # --------------------------------------------------

        counseling_session_count = clip(
            np.round(
                academic_stress_score / 30
                + rng.normal(0, 1, N_STUDENTS)
            ),
            0, 10
        ).astype(int)

        # --------------------------------------------------
        # FEATURE 14: ENGAGEMENT RISK SCORE
        # Derived from engagement metrics
        # --------------------------------------------------

        engagement_risk_score = clip(
            0.35 * (100 - lecture_attendance_rate)
            + 0.25 * (100 - assignment_submission_rate * 100)
            + 0.20 * (100 - lms_login_count)
            + 0.20 * (100 - weekly_active_hours / 35 * 100),
            0, 100
        )

        # --------------------------------------------------
        # FEATURE 15: WITHDRAWAL HAZARD SCORE
        # Derived from current observed risk domains
        # --------------------------------------------------

        academic_risk_score = clip(
            0.6 * academic_risk_proxy
            + 0.4 * (failed_subject_count / 8 * 100),
            0, 100
        )

        financial_risk_score = financial_risk_proxy

        affect_risk_score = clip(
            (100 - satisfaction) * 0.6
            + (academic_stress_score * 0.4),
            0, 100
        )

        withdrawal_hazard_score = clip(
            0.30 * engagement_risk_score
            + 0.30 * academic_risk_score
            + 0.20 * financial_risk_score
            + 0.20 * affect_risk_score
            + rng.normal(0, 4, N_STUDENTS),
            0, 100
        )

        # --------------------------------------------------
        # TARGET: PERSISTENCE NEXT TERM
        # 1 = Continues enrollment
        # 0 = Does not continue enrollment
        # --------------------------------------------------

        intervention_effect = (
            support_request_count * 0.01
            + counseling_session_count * 0.015
        )

        combined_risk = (
            0.30 * engagement_risk_score
            + 0.30 * academic_risk_score
            + 0.20 * financial_risk_score
            + 0.20 * affect_risk_score
        )

        logit = (
            2.0
            - combined_risk / 23
            + intervention_effect
            + rng.normal(0, 0.65, N_STUDENTS)
        )

        persistence_probability = sigmoid(logit)

        persistence_next_term = (
            rng.random(N_STUDENTS)
            < persistence_probability
        ).astype(int)

        # --------------------------------------------------
        # 3. BUILD MONTHLY DATAFRAME
        # --------------------------------------------------

        monthly_data = pd.DataFrame({

            "Student_ID": student_ids,
            "Observation_Date": date,

            "LMS_Login_Count": lms_login_count,
            "Weekly_Active_Hours": weekly_active_hours,
            "Assignment_Submission_Rate":
                assignment_submission_rate,
            "Lecture_Attendance_Rate":
                lecture_attendance_rate,

            "Current_GPA": current_gpa,
            "Failed_Subject_Count": failed_subject_count,

            "Tuition_Fee": monthly_tuition_fee,
            "Outstanding_Fee": outstanding_fee,
            "Scholarship_Amount": monthly_scholarship,

            "Feedback_Sentiment_Score":
                feedback_sentiment_score,
            "Academic_Stress_Survey_Score":
                academic_stress_score,

            "Support_Request_Count": support_request_count,
            "Counseling_Session_Count":
                counseling_session_count,

            "Engagement_Risk_Score":
                engagement_risk_score,
            "Withdrawal_Hazard_Score":
                withdrawal_hazard_score,

            "Persistence_Next_Term":
                persistence_next_term

        })

        records.append(monthly_data)

    # ------------------------------------------------------
    # 4. COMBINE MONTHLY OBSERVATIONS
    # ------------------------------------------------------

    dataset = pd.concat(records, ignore_index=True)

    dataset = dataset.sort_values(
        ["Student_ID", "Observation_Date"]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # 5. SAVE DATASET
    # ------------------------------------------------------

    output_file = (
        OUTPUT_DIR / "unified_persistence_dataset.csv"
    )

    dataset.to_csv(output_file, index=False)

    # ------------------------------------------------------
    # 6. DATASET VALIDATION
    # ------------------------------------------------------

    print("\n" + "=" * 55)
    print("UNIFIED PERSISTENCE CORRELATOR")
    print("SYNTHETIC DATASET GENERATED SUCCESSFULLY")
    print("=" * 55)

    print("\nTotal students:",
          dataset["Student_ID"].nunique())

    print("Observation periods:", N_MONTHS)

    print("Total records:", len(dataset))

    print("Total columns:", len(dataset.columns))

    print("\nTarget distribution:")

    print(
        dataset["Persistence_Next_Term"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nMissing values:",
          dataset.isnull().sum().sum())

    print("\nDuplicate student-month records:",
          dataset.duplicated(
              ["Student_ID", "Observation_Date"]
          ).sum())

    print("\nFeature columns:")

    print(dataset.columns.tolist())

    print("\nSample records:")

    print(dataset.head())

    print("\nSaved location:", output_file.resolve())

    return dataset


if __name__ == "__main__":
    df = generate_dataset()

# import numpy as np
# import pandas as pd
# from pathlib import Path

# # ============================================================
# # UNIFIED PERSISTENCE CORRELATOR
# # Longitudinal Synthetic Dataset Generator
# # ============================================================

# # Configuration
# N_STUDENTS = 10_000
# N_MONTHS = 6
# RANDOM_SEED = 42

# OUTPUT_DIR = Path("dataset")
# OUTPUT_DIR.mkdir(exist_ok=True)

# rng = np.random.default_rng(RANDOM_SEED)


# def clip(values, low, high):
#     return np.clip(values, low, high)


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# def generate_dataset():

#     n = N_STUDENTS

#     # --------------------------------------------------------
#     # 1. STUDENT PROFILES
#     # --------------------------------------------------------

#     student_ids = np.array([
#         f"STU{str(i).zfill(5)}"
#         for i in range(1, n + 1)
#     ])

#     departments = rng.choice(
#         ["Computer Science", "Mechanical", "Electrical",
#          "Civil", "Business", "Biotechnology"],
#         n,
#         p=[0.30, 0.16, 0.15, 0.12, 0.15, 0.12]
#     )

#     program_levels = rng.choice(
#         ["Undergraduate", "Postgraduate"],
#         n,
#         p=[0.85, 0.15]
#     )

#     study_modes = rng.choice(
#         ["Full_Time", "Part_Time"],
#         n,
#         p=[0.88, 0.12]
#     )

#     enrollment_year = rng.choice(
#         [2023, 2024, 2025, 2026],
#         n,
#         p=[0.15, 0.25, 0.35, 0.25]
#     )

#     enrollment_year = np.where(
#         enrollment_year == 2026, 2025, enrollment_year
#     )

#     study_year = rng.integers(1, 5, n)

#     # Latent student characteristics.
#     # These create correlations between observations.
#     engagement_trait = rng.normal(0, 1, n)
#     academic_trait = rng.normal(0, 1, n)
#     financial_trait = rng.normal(0, 1, n)
#     satisfaction_trait = rng.normal(0, 1, n)

#     # Financial characteristics
#     tuition_fee = rng.normal(85000, 18000, n)
#     tuition_fee = clip(tuition_fee, 35000, 160000)

#     scholarship_amount = np.where(
#         rng.random(n) < 0.35,
#         rng.uniform(10000, 60000, n),
#         0
#     )

#     financial_aid_status = np.where(
#         scholarship_amount >= 40000, "Full",
#         np.where(scholarship_amount > 0, "Partial", "None")
#     )

#     # Academic baseline
#     base_gpa = clip(
#         6.5 + academic_trait * 0.9 + rng.normal(0, 0.5, n),
#         3.0, 10.0
#     )

#     # --------------------------------------------------------
#     # 2. LONGITUDINAL OBSERVATIONS
#     # --------------------------------------------------------

#     all_records = []

#     observation_dates = pd.date_range(
#         start="2026-01-01",
#         periods=N_MONTHS,
#         freq="MS"
#     )

#     for month_index, observation_date in enumerate(
#         observation_dates
#     ):

#         # Monthly variation
#         monthly_engagement = rng.normal(0, 0.35, n)
#         monthly_academic = rng.normal(0, 0.30, n)
#         monthly_financial = rng.normal(0, 0.25, n)

#         # Gradual trends across observation periods
#         time_factor = month_index / max(N_MONTHS - 1, 1)

#         # ----------------------------------------------------
#         # A. ENGAGEMENT FEATURES
#         # ----------------------------------------------------

#         engagement_level = (
#             0.55 * engagement_trait
#             + monthly_engagement
#             - 0.12 * time_factor
#         )

#         attendance = clip(
#             75 + engagement_level * 17 + rng.normal(0, 7, n),
#             20, 100
#         )

#         lms_logins = clip(
#             35 + engagement_level * 13
#             + rng.normal(0, 9, n),
#             0, 100
#         ).astype(int)

#         weekly_active_hours = clip(
#             10 + engagement_level * 3.5
#             + rng.normal(0, 2.5, n),
#             0, 35
#         )

#         assignment_submission = clip(
#             0.78 + engagement_level * 0.12
#             + rng.normal(0, 0.08, n),
#             0, 1
#         )

#         discussion_participation = clip(
#             5 + engagement_level * 3
#             + rng.normal(0, 2, n),
#             0, 25
#         ).astype(int)

#         learning_material_views = clip(
#             45 + engagement_level * 18
#             + rng.normal(0, 12, n),
#             0, 150
#         ).astype(int)

#         inactive_days = clip(
#             8 - engagement_level * 3
#             + rng.normal(0, 4, n),
#             0, 45
#         ).astype(int)

#         engagement_score = clip(
#             0.25 * attendance
#             + 0.20 * (assignment_submission * 100)
#             + 0.20 * (lms_logins / 100 * 100)
#             + 0.15 * (weekly_active_hours / 35 * 100)
#             + 0.10 * (learning_material_views / 150 * 100)
#             + 0.10 * (discussion_participation / 25 * 100),
#             0, 100
#         )

#         # ----------------------------------------------------
#         # B. ACADEMIC FEATURES
#         # ----------------------------------------------------

#         current_gpa = clip(
#             base_gpa
#             + monthly_academic * 0.6
#             - 0.15 * time_factor
#             + rng.normal(0, 0.25, n),
#             0, 10
#         )

#         previous_gpa = clip(
#             current_gpa
#             - rng.normal(0.1, 0.35, n),
#             0, 10
#         )

#         gpa_change = current_gpa - previous_gpa

#         failed_subject_count = clip(
#             np.round(
#                 1.5 - current_gpa * 0.17
#                 + rng.normal(0, 0.8, n)
#             ),
#             0, 8
#         ).astype(int)

#         pending_assignments = clip(
#             np.round(
#                 3 - engagement_level * 1.2
#                 + rng.normal(0, 1.5, n)
#             ),
#             0, 15
#         ).astype(int)

#         assessment_average = clip(
#             current_gpa * 9.5 + rng.normal(0, 8, n),
#             0, 100
#         )

#         course_completion_rate = clip(
#             0.65 + engagement_level * 0.12
#             + current_gpa * 0.018
#             + rng.normal(0, 0.08, n),
#             0, 1
#         )

#         credit_completion_rate = clip(
#             course_completion_rate
#             + rng.normal(0, 0.06, n),
#             0, 1
#         )

#         academic_risk_score = clip(
#             100 - (
#                 0.35 * (current_gpa / 10 * 100)
#                 + 0.25 * assessment_average
#                 + 0.20 * (course_completion_rate * 100)
#                 + 0.20 * (100 - failed_subject_count / 8 * 100)
#             ),
#             0, 100
#         )

#         academic_progress_score = 100 - academic_risk_score

#         # ----------------------------------------------------
#         # C. FINANCIAL FEATURES
#         # ----------------------------------------------------

#         fee_payment_delay_days = clip(
#             15 - financial_trait * 8
#             + monthly_financial * 5
#             + rng.normal(0, 8, n),
#             0, 120
#         ).astype(int)

#         outstanding_fee = clip(
#             tuition_fee * (
#                 0.15 + fee_payment_delay_days / 150
#                 - scholarship_amount / np.maximum(tuition_fee, 1) * 0.2
#                 + rng.normal(0, 0.08, n)
#             ),
#             0, tuition_fee
#         )

#         payment_completion_rate = clip(
#             1 - outstanding_fee / np.maximum(tuition_fee, 1),
#             0, 1
#         )

#         financial_risk_score = clip(
#             0.45 * (fee_payment_delay_days / 120 * 100)
#             + 0.40 * (outstanding_fee / tuition_fee * 100)
#             + 0.15 * (100 - payment_completion_rate * 100),
#             0, 100
#         )

#         # ----------------------------------------------------
#         # D. AFFECT / FEEDBACK FEATURES
#         # ----------------------------------------------------

#         satisfaction_score = clip(
#             65 + satisfaction_trait * 12
#             + engagement_level * 5
#             + rng.normal(0, 10, n),
#             0, 100
#         )

#         feedback_sentiment = clip(
#             (satisfaction_score - 50) / 50
#             + rng.normal(0, 0.15, n),
#             -1, 1
#         )

#         academic_stress_score = clip(
#             50 + academic_risk_score * 0.25
#             + financial_risk_score * 0.15
#             - satisfaction_score * 0.15
#             + rng.normal(0, 12, n),
#             0, 100
#         )

#         support_request_count = clip(
#             np.round(
#                 1 + academic_risk_score / 35
#                 + financial_risk_score / 60
#                 + rng.normal(0, 1.2, n)
#             ),
#             0, 15
#         ).astype(int)

#         negative_feedback_count = clip(
#             np.round(
#                 (100 - satisfaction_score) / 25
#                 + rng.normal(0, 1, n)
#             ),
#             0, 10
#         ).astype(int)

#         response_engagement_rate = clip(
#             0.65 + engagement_level * 0.10
#             + rng.normal(0, 0.12, n),
#             0, 1
#         )

#         affect_interaction_score = clip(
#             0.35 * satisfaction_score
#             + 0.30 * ((feedback_sentiment + 1) * 50)
#             + 0.20 * (response_engagement_rate * 100)
#             + 0.15 * (100 - academic_stress_score),
#             0, 100
#         )

#         affect_risk_score = 100 - affect_interaction_score

#         # ----------------------------------------------------
#         # E. INSTITUTIONAL SUPPORT FEATURES
#         # ----------------------------------------------------

#         counseling_session_count = clip(
#             np.round(
#                 academic_stress_score / 30
#                 + rng.normal(0, 1, n)
#             ),
#             0, 10
#         ).astype(int)

#         academic_advising_count = clip(
#             np.round(
#                 academic_risk_score / 30
#                 + rng.normal(0, 1, n)
#             ),
#             0, 10
#         ).astype(int)

#         mentoring_participation = (
#             rng.random(n) < sigmoid(
#                 0.3 + engagement_level * 0.5
#             )
#         ).astype(int)

#         support_ticket_count = clip(
#             support_request_count
#             + rng.poisson(1.0, n),
#             0, 20
#         ).astype(int)

#         support_resolution_rate = clip(
#             0.75 + satisfaction_trait * 0.08
#             + rng.normal(0, 0.15, n),
#             0, 1
#         )

#         days_since_last_advisor_contact = clip(
#             35 - academic_risk_score * 0.18
#             - support_ticket_count * 1.5
#             + rng.normal(0, 10, n),
#             0, 180
#         ).astype(int)

#         # ----------------------------------------------------
#         # F. UNIFIED RISK CORRELATION
#         # ----------------------------------------------------

#         engagement_risk_score = 100 - engagement_score

#         # Composite hazard estimate from upstream domains.
#         withdrawal_hazard_score = clip(
#             0.25 * engagement_risk_score
#             + 0.25 * academic_risk_score
#             + 0.25 * financial_risk_score
#             + 0.25 * affect_risk_score
#             + rng.normal(0, 4, n),
#             0, 100
#         )

#         # ----------------------------------------------------
#         # G. TARGET GENERATION
#         # ----------------------------------------------------

#         # Higher risk reduces the probability of persistence.
#         # Support participation can improve the probability.
#         # Additional unobserved variation prevents a
#         # deterministic relationship between features and target.

#         intervention_effect = (
#             mentoring_participation * 0.22
#             + support_resolution_rate * 0.18
#         )

#         latent_risk = (
#             0.25 * engagement_risk_score
#             + 0.27 * academic_risk_score
#             + 0.23 * financial_risk_score
#             + 0.15 * affect_risk_score
#             + 0.10 * withdrawal_hazard_score
#         )

#         logit = (
#             2.0
#             - latent_risk / 23
#             + intervention_effect
#             + rng.normal(0, 0.65, n)
#         )

#         persistence_probability = sigmoid(logit)

#         persistence_next_term = (
#             rng.random(n) < persistence_probability
#         ).astype(int)

#         # ----------------------------------------------------
#         # H. CREATE MONTHLY DATAFRAME
#         # ----------------------------------------------------

#         records = pd.DataFrame({

#             # Student and institutional
#             "Student_ID": student_ids,
#             "Observation_Date": observation_date,
#             "Academic_Year": "2025-2026",
#             "Semester": (
#                 "Semester_1" if observation_date.month <= 6
#                 else "Semester_2"
#             ),
#             "Department": departments,
#             "Program_Level": program_levels,
#             "Study_Mode": study_modes,
#             "Enrollment_Year": enrollment_year,
#             "Study_Year": study_year,

#             # Engagement
#             "LMS_Login_Count": lms_logins,
#             "Weekly_Active_Hours": weekly_active_hours,
#             "Assignment_Submission_Rate": assignment_submission,
#             "Lecture_Attendance_Rate": attendance,
#             "Discussion_Participation": discussion_participation,
#             "Learning_Material_Views": learning_material_views,
#             "Consecutive_Inactive_Days": inactive_days,
#             "Engagement_Score": engagement_score,

#             # Academic
#             "Current_GPA": current_gpa,
#             "Previous_GPA": previous_gpa,
#             "GPA_Change": gpa_change,
#             "Failed_Subject_Count": failed_subject_count,
#             "Pending_Assignments": pending_assignments,
#             "Assessment_Average": assessment_average,
#             "Course_Completion_Rate": course_completion_rate,
#             "Credit_Completion_Rate": credit_completion_rate,
#             "Academic_Risk_Score": academic_risk_score,
#             "Academic_Progress_Score": academic_progress_score,

#             # Financial
#             "Tuition_Fee": tuition_fee,
#             "Outstanding_Fee": outstanding_fee,
#             "Fee_Payment_Delay_Days": fee_payment_delay_days,
#             "Scholarship_Amount": scholarship_amount,
#             "Financial_Aid_Status": financial_aid_status,
#             "Payment_Completion_Rate": payment_completion_rate,
#             "Financial_Risk_Score": financial_risk_score,

#             # Affect and feedback
#             "Feedback_Sentiment_Score": feedback_sentiment,
#             "Student_Satisfaction_Score": satisfaction_score,
#             "Academic_Stress_Survey_Score": academic_stress_score,
#             "Support_Request_Count": support_request_count,
#             "Negative_Feedback_Count": negative_feedback_count,
#             "Response_Engagement_Rate": response_engagement_rate,
#             "Affect_Interaction_Score": affect_interaction_score,
#             "Affect_Risk_Score": affect_risk_score,

#             # Institutional support
#             "Counseling_Session_Count": counseling_session_count,
#             "Academic_Advising_Count": academic_advising_count,
#             "Mentoring_Participation": mentoring_participation,
#             "Support_Ticket_Count": support_ticket_count,
#             "Support_Resolution_Rate": support_resolution_rate,
#             "Days_Since_Last_Advisor_Contact":
#                 days_since_last_advisor_contact,

#             # Correlator output
#             "Engagement_Risk_Score": engagement_risk_score,
#             "Withdrawal_Hazard_Score": withdrawal_hazard_score,

#             # Prediction target
#             "Persistence_Next_Term": persistence_next_term
#         })

#         all_records.append(records)

#     # --------------------------------------------------------
#     # 3. COMBINE ALL MONTHS
#     # --------------------------------------------------------

#     dataset = pd.concat(all_records, ignore_index=True)

#     dataset = dataset.sort_values(
#         ["Student_ID", "Observation_Date"]
#     ).reset_index(drop=True)

#     # --------------------------------------------------------
#     # 4. SAVE DATASET
#     # --------------------------------------------------------

#     output_file = OUTPUT_DIR / "synthetic_student_data.csv"

#     dataset.to_csv(output_file, index=False)

#     # --------------------------------------------------------
#     # 5. GENERATE DATASET SUMMARY
#     # --------------------------------------------------------

#     print("\n" + "=" * 60)
#     print("UNIFIED PERSISTENCE CORRELATOR")
#     print("SYNTHETIC DATA GENERATION COMPLETED")
#     print("=" * 60)

#     print(f"\nUnique Students: {dataset['Student_ID'].nunique():,}")
#     print(f"Observation Periods: {N_MONTHS}")
#     print(f"Total Records: {len(dataset):,}")
#     print(f"Total Features: {len(dataset.columns)}")

#     print("\nTarget Distribution:")
#     print(
#         dataset["Persistence_Next_Term"]
#         .value_counts(normalize=True)
#         .mul(100)
#         .round(2)
#         .rename("Percentage")
#     )

#     print("\nMissing Values:")
#     print(dataset.isnull().sum().sum())

#     print("\nDuplicate Records:")
#     print(dataset.duplicated(
#         ["Student_ID", "Observation_Date"]
#     ).sum())

#     print("\nDataset saved to:", output_file.resolve())

#     print("\nSample Records:")
#     print(dataset.head())

#     print("\nDataset Information:")
#     print(dataset.info())

#     return dataset


# if __name__ == "__main__":
#     df = generate_dataset()
