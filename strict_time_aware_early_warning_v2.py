r"""
Prospective, future-presentation early-warning training and evaluation.

This module trains and evaluates the project's time-aware withdrawal
early-warning models from engineered weekly OULAD records. It constructs
student-course snapshots at weeks 2, 4, and 6; restricts each snapshot to
enrollments still eligible for prediction at that cutoff; and predicts
withdrawal after the cutoff through course end. For each landmark it fits
engagement, affect, and hazard component estimators and a stacking correlator,
then calibrates probabilities and selects the decision threshold using
separate validation subsets.

Input:
  ``engineered_weekly_features_with_affect.csv`` beside this file, or
  ``engineered_data/engineered_weekly_features.csv`` as the fallback.

Temporal evaluation:
  2013B and 2013J are used for training, 2014B for calibration and threshold
  selection, and 2014J is held out as a future-presentation test set. Student
  IDs are purged across partitions. At each landmark, registrations that have
  not started by the cutoff, enrollments already ended, and withdrawals with
  unknown unregistration time are excluded from the risk set.

Outputs:
  ``strict_early_warning_v2/`` contains per-week model artifacts,
  validation/test predictions, threshold search and calibration tables,
  landmark summaries, plus the presentation split and overall summary.
  ``component_models.joblib``, ``learned_correlator.joblib``, and
  ``model_config.json`` are consumed by ``api.py`` (directly or through
  ``combine_models.py``).

Run from the project directory with
``uv run python strict_time_aware_early_warning_v2.py`` after generating the
engineered input. The output scores are model estimates, not definitive
determinations of an individual student's future behavior.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------- CONFIGURATION -----------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "engineered_weekly_features_with_affect.csv"
if not INPUT_FILE.exists():
    INPUT_FILE = BASE_DIR / "engineered_data" / "engineered_weekly_features.csv"
OUTPUT_DIR = BASE_DIR / "strict_early_warning_v2"

LANDMARKS = [2, 4, 6]
RANDOM_STATE = 42
N_SPLITS = 5

FEATURES = [
    "week_number",
    "cumulative_active_weeks",
    "cumulative_active_days",
    "cumulative_interactions",
    "cumulative_clicks",
    "days_since_registration",
    "average_clicks_per_active_week",
    "studied_credits",
    "previous_week_clicks",
    "weekly_total_clicks",
    "click_change_from_previous_week",
    "weekly_interaction_records",
    "weekly_active_days",
    "num_of_prev_attempts",
    "hazard_low_active_days",
]

AFFECT_TARGETS = [
    "motivation",
    "confidence",
    "frustration",
    "stress",
    "disengagement",
    "wellbeing",
]

ENGAGEMENT_TARGET = "engagement_score"
HAZARD_TARGET = "target_withdrawn"

KEYS = ["id_student", "code_module", "code_presentation"]
REQUIRED = KEYS + [
    "week_number", "date_registration", "date_unregistration",
    HAZARD_TARGET, ENGAGEMENT_TARGET,
] + AFFECT_TARGETS + FEATURES


# ----------------------------- UTILITIES ---------------------------------

def safe_auc(y_true, probabilities):
    """Return ROC-AUC when both outcome classes are present.

    Parameters
    ----------
    y_true : array-like
        Binary observed outcomes.
    probabilities : array-like
        Scores or probabilities for the positive class.

    Returns
    -------
    float or None
        ROC-AUC, or ``None`` when the sample contains only one class.
    """
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification and calibration metrics at a fixed threshold.

    Parameters
    ----------
    y_true : array-like
        Binary withdrawal outcomes.
    probabilities : array-like
        Predicted withdrawal probabilities.
    threshold : float
        Probability cutoff used to produce class predictions.

    Returns
    -------
    dict
        Scalar metrics and the confusion matrix for the supplied predictions.
    """
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "records": int(len(y_true)),
        "withdrawal_cases": int(np.sum(y_true)),
        "withdrawal_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "roc_auc": safe_auc(y_true, probabilities),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def expected_calibration_error(y_true, probabilities, bins=10):
    """Estimate calibration error from equal-count probability groups.

    Parameters
    ----------
    y_true : array-like
        Binary observed outcomes.
    probabilities : array-like
        Predicted probabilities, sorted into approximately equal-sized groups.
    bins : int, default=10
        Maximum number of groups.

    Returns
    -------
    float
        Weighted absolute difference between each group's mean prediction and
        observed outcome rate; ``NaN`` for an empty input.
    """
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    order = np.argsort(probabilities)
    chunks = np.array_split(order, min(bins, len(order)))
    return float(sum(
        len(indices) / len(order)
        * abs(float(probabilities[indices].mean()) - float(y_true[indices].mean()))
        for indices in chunks if len(indices)
    ))


def make_regressor():
    """Create the imputed random-forest regressor used for component targets.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Median imputation followed by the configured random-forest regressor.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            max_features=0.85,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def make_classifier():
    """Create the imputed soft-voting classifier for withdrawal risk.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Median imputation followed by the configured weighted classifier
        ensemble.
    """
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        min_samples_leaf=15,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    rf = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", VotingClassifier(
            estimators=[("hgb", hgb), ("rf", rf)],
            voting="soft",
            weights=[0.55, 0.45],
        )),
    ])


def component_prediction_matrix(models, X):
    """Assemble component-model outputs in stacking-model feature order.

    Parameters
    ----------
    models : dict
        Fitted engagement, affect, and hazard component models.
    X : pandas.DataFrame
        Feature rows to score.

    Returns
    -------
    numpy.ndarray
        Columns containing engagement, affect targets, and hazard probability.
    """
    engagement = models["engagement"].predict(X)
    affect = models["affect"].predict(X)
    hazard = models["hazard"].predict_proba(X)[:, 1]

    if affect.ndim == 1:
        affect = affect.reshape(-1, len(AFFECT_TARGETS))

    columns = [engagement.reshape(-1, 1), affect, hazard.reshape(-1, 1)]
    return np.column_stack(columns)


def train_component_models(X, y, affect_y):
    """Fit the engagement, affect, and withdrawal component estimators.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor features for one training partition.
    y : dict
        Engagement and binary hazard targets keyed by target name.
    affect_y : pandas.DataFrame
        Affect-proxy targets in ``AFFECT_TARGETS`` order.

    Returns
    -------
    dict
        Fitted component estimators keyed by ``engagement``, ``affect``, and
        ``hazard``.
    """
    models = {
        "engagement": make_regressor(),
        "affect": make_regressor(),
        "hazard": make_classifier(),
    }
    models["engagement"].fit(X, y["engagement"])
    models["affect"].fit(X, affect_y)
    models["hazard"].fit(X, y["hazard"])
    return models


def create_oof_component_predictions(X, y, affect_y, groups):
    """Generate grouped out-of-fold component predictions for stacking.

    Each student's rows are held out together so the meta-model is trained
    from predictions made without fitting the corresponding component models
    on that student's outcomes.

    Parameters
    ----------
    X : pandas.DataFrame
        Training predictors.
    y : dict
        Engagement and hazard training targets.
    affect_y : pandas.DataFrame
        Affect-proxy training targets.
    groups : array-like
        Student identifiers used to keep groups intact across folds.

    Returns
    -------
    numpy.ndarray
        Out-of-fold predictions in the order expected by the stacking model.

    Raises
    ------
    ValueError
        If fewer than two distinct groups are available.
    RuntimeError
        If any training row is missing an out-of-fold prediction.
    """
    unique_groups = pd.Series(groups).nunique()
    folds = min(N_SPLITS, unique_groups)
    if folds < 2:
        raise ValueError("Not enough distinct training students for grouped OOF predictions.")

    splitter = GroupKFold(n_splits=folds)
    oof = np.full((len(X), 1 + len(AFFECT_TARGETS) + 1), np.nan, dtype=float)

    for fold_number, (fit_idx, hold_idx) in enumerate(splitter.split(X, y["hazard"], groups)):
        fold_models = train_component_models(
            X.iloc[fit_idx],
            {
                "engagement": y["engagement"].iloc[fit_idx],
                "hazard": y["hazard"].iloc[fit_idx],
            },
            affect_y.iloc[fit_idx],
        )
        oof[hold_idx] = component_prediction_matrix(fold_models, X.iloc[hold_idx])
        print(f"      OOF fold {fold_number + 1}/{folds} complete")

    if np.isnan(oof).any():
        raise RuntimeError("Some OOF component predictions were not generated.")
    return oof


def select_threshold(y_true, probabilities):
    """Choose the most accurate threshold subject to a recall constraint.

    Candidate thresholds are evaluated on the supplied validation subset.
    Thresholds achieving at least 50% withdrawal recall are preferred; if
    none qualify, all candidates remain eligible. Accuracy, F1, and precision
    determine the tie-break order.

    Parameters
    ----------
    y_true : array-like
        Binary validation outcomes.
    probabilities : array-like
        Predicted validation probabilities.

    Returns
    -------
    tuple
        Selected threshold and a dataframe containing metrics for all
        candidate thresholds.
    """
    thresholds = np.linspace(0.01, 0.99, 197)
    rows = []
    for threshold in thresholds:
        metrics = calculate_metrics(y_true, probabilities, threshold)
        rows.append(metrics)

    eligible = [m for m in rows if m["recall"] >= 0.50]
    if not eligible:
        eligible = rows

    best = max(
        eligible,
        key=lambda item: (
            item["accuracy"],
            item["f1"],
            item["precision"],
        ),
    )
    return best["threshold"], pd.DataFrame([
        {k: v for k, v in row.items() if k != "confusion_matrix"}
        for row in rows
    ])
def write_json(path, payload):
    """Create parent directories and serialize a JSON payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, default=str)


# ----------------------------- DATA PREPARATION ---------------------------

def load_data():
    """Load, normalize, and validate the engineered weekly feature table.

    Returns
    -------
    pandas.DataFrame
        Sorted weekly records with required targets and numeric predictors.

    Raises
    ------
    FileNotFoundError
        If neither configured input file exists.
    ValueError
        If required columns are absent.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Make sure engineered_weekly_features_with_affect.csv is in engineered_data."
        )

    print("Loading engineered weekly dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    alias_map = {
        "motivation": ["affect_motivation_estimate", "motivation_estimate", "affect_motivation"],
        "confidence": ["affect_confidence_estimate", "confidence_estimate", "affect_confidence"],
        "frustration": ["affect_frustration_estimate", "frustration_estimate", "affect_frustration"],
        "stress": ["affect_stress_estimate", "stress_estimate", "affect_stress"],
        "disengagement": ["affect_disengagement_estimate", "disengagement_estimate", "affect_disengagement"],
        "wellbeing": ["affect_wellbeing_proxy", "wellbeing_proxy", "affect_wellbeing_estimate", "wellbeing_estimate"],
    }
    for target_col, candidates in alias_map.items():
        if target_col not in df.columns:
            for cand in candidates:
                if cand in df.columns:
                    df[target_col] = df[cand]
                    break

    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    df["date_unregistration"] = pd.to_numeric(df["date_unregistration"], errors="coerce")
    df[HAZARD_TARGET] = pd.to_numeric(df[HAZARD_TARGET], errors="coerce")
    df = df.dropna(subset=KEYS + ["week_number", HAZARD_TARGET]).copy()
    df[HAZARD_TARGET] = df[HAZARD_TARGET].astype(int)
    df = df[df[HAZARD_TARGET].isin([0, 1])].copy()

    for column in FEATURES + [ENGAGEMENT_TARGET] + AFFECT_TARGETS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(KEYS + ["week_number"])
    print(f"Dataset rows: {len(df):,}")
    print(f"Unique students: {df['id_student'].nunique():,}")
    print(f"Student-course records: {df[KEYS].drop_duplicates().shape[0]:,}")
    print(f"Withdrawal rate: {df[HAZARD_TARGET].mean():.2%}")
    return df


def presentation_sort_key(value):
    """Sort presentation labels chronologically, placing B before J."""
    text = str(value)
    return (int(text[:4]), 0 if text.endswith("B") else 1)


def create_presentation_split(df):
    """Create chronological train/validation/test partitions without student overlap.

    The latest presentation is reserved for testing and the preceding
    presentation for validation; students assigned to a later partition are
    removed from earlier partitions.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated weekly input records.

    Returns
    -------
    tuple
        A student-presentation-to-partition mapping and split metadata.

    Raises
    ------
    ValueError
        If fewer than three presentations exist or a partition is empty.
    AssertionError
        If student identifiers overlap across partitions.
    """
    presentations = sorted(df["code_presentation"].dropna().unique(), key=presentation_sort_key)
    if len(presentations) < 3:
        raise ValueError("At least three ordered presentations are required for train/validation/test.")

    test_presentation = presentations[-1]
    validation_presentation = presentations[-2]
    train_presentations = presentations[:-2]

    test_rows = df[df["code_presentation"] == test_presentation]
    test_students = set(test_rows["id_student"])
    validation_rows = df[
        (df["code_presentation"] == validation_presentation)
        & ~df["id_student"].isin(test_students)
    ]
    validation_students = set(validation_rows["id_student"])
    training_rows = df[
        df["code_presentation"].isin(train_presentations)
        & ~df["id_student"].isin(test_students | validation_students)
    ]
    training_students = set(training_rows["id_student"])

    if training_students & validation_students or training_students & test_students or validation_students & test_students:
        raise AssertionError("Student IDs overlap across temporal partitions.")
    if min(len(training_rows), len(validation_rows), len(test_rows)) == 0:
        raise ValueError("At least one temporal partition is empty after student purging.")

    split_map = {}
    for frame, partition in (
        (training_rows, "train"),
        (validation_rows, "validation"),
        (test_rows, "test"),
    ):
        for student, presentation in frame[["id_student", "code_presentation"]].drop_duplicates().itertuples(index=False):
            split_map[(student, presentation)] = partition

    split_df = pd.DataFrame([
        {"id_student": student, "code_presentation": presentation, "partition": partition}
        for (student, presentation), partition in split_map.items()
    ])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(OUTPUT_DIR / "presentation_split.csv", index=False)
    split_info = {
        "training_presentations": list(train_presentations),
        "validation_presentation": validation_presentation,
        "test_presentation": test_presentation,
        "train_students": len(training_students),
        "validation_students": len(validation_students),
        "test_students": len(test_students),
        "student_overlap_purged": True,
    }
    write_json(OUTPUT_DIR / "presentation_split.json", split_info)
    print(
        f"Temporal split: train={train_presentations}, validation={validation_presentation}, "
        f"future test={test_presentation}; students train/validation/test="
        f"{len(training_students):,}/{len(validation_students):,}/{len(test_students):,}"
    )
    return split_map, split_info


def create_landmark_snapshot(df, landmark):
    """Select each enrollment's latest available row at or before a landmark.

    Parameters
    ----------
    df : pandas.DataFrame
        Weekly records.
    landmark : int
        Week through which feature information may be observed.

    Returns
    -------
    pandas.DataFrame
        One snapshot row per student-course enrollment with landmark metadata.
    """
    eligible = df[df["week_number"] <= landmark].copy()
    snapshot = (
        eligible.sort_values("week_number")
        .groupby(KEYS, as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    snapshot["observed_week"] = snapshot["week_number"]
    snapshot["landmark_week"] = landmark
    return snapshot.reset_index(drop=True)


def restrict_to_landmark_risk_set(snapshot, landmark):
    """Keep enrollments eligible for prediction at the landmark cutoff.

    Students not yet registered, already unregistered, or withdrawn with no
    known unregistration time are excluded. The returned target marks
    withdrawal strictly after the cutoff through course end.

    Parameters
    ----------
    snapshot : pandas.DataFrame
        One-row-per-enrollment landmark snapshot.
    landmark : int
        Week defining the risk-set cutoff.

    Returns
    -------
    pandas.DataFrame
        Eligible rows with future-withdrawal labels and cutoff metadata.
    """
    cutoff_day = (landmark + 1) * 7 - 1
    registered = snapshot["date_registration"].notna() & (snapshot["date_registration"] <= cutoff_day)
    still_enrolled = snapshot["date_unregistration"].isna() | (snapshot["date_unregistration"] > cutoff_day)
    unknown_withdrawal_time = (
        snapshot[HAZARD_TARGET].eq(1) & snapshot["date_unregistration"].isna()
    )
    eligible = snapshot[registered & still_enrolled & ~unknown_withdrawal_time].copy()
    eligible["target_future_withdrawal"] = (
        eligible[HAZARD_TARGET].eq(1)
        & eligible["date_unregistration"].gt(cutoff_day)
    ).astype(int)
    eligible["risk_set_cutoff_day"] = cutoff_day
    return eligible


# ----------------------------- LANDMARK TRAINING --------------------------

def run_landmark(df, landmark, split_map):
    """Train, calibrate, evaluate, and persist one landmark's model bundle.

    Component models are trained on the training partition, the stacking
    correlator uses grouped out-of-fold predictions, calibration and threshold
    selection use disjoint student subsets of validation, and final metrics
    are measured on the untouched future-presentation test partition.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated weekly input table.
    landmark : int
        Prediction week to evaluate.
    split_map : dict
        Mapping from student-presentation keys to temporal partitions.

    Returns
    -------
    dict
        Summary metrics and record counts for the future test partition.
    """
    print("\n" + "=" * 72)
    print(f"STRICT EARLY-WARNING MODEL: WEEK {landmark}")
    print("=" * 72)

    snapshot = create_landmark_snapshot(df, landmark)
    snapshot["partition"] = [
        split_map.get((student, presentation))
        for student, presentation in zip(
            snapshot["id_student"], snapshot["code_presentation"]
        )
    ]
    snapshot = snapshot.dropna(subset=["partition"]).copy()
    snapshot = restrict_to_landmark_risk_set(snapshot, landmark)

    train = snapshot[snapshot["partition"] == "train"].copy()
    validation = snapshot[snapshot["partition"] == "validation"].copy()
    test = snapshot[snapshot["partition"] == "test"].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(f"Empty partition at landmark week {landmark}.")

    X_train = train[FEATURES]
    X_val = validation[FEATURES]
    X_test = test[FEATURES]

    y_train = {
        "engagement": train[ENGAGEMENT_TARGET].fillna(train[ENGAGEMENT_TARGET].median()),
        "hazard": train["target_future_withdrawal"].astype(int),
    }
    affect_train = train[AFFECT_TARGETS].copy()
    affect_train = affect_train.fillna(affect_train.median()).fillna(0)

    y_val = validation["target_future_withdrawal"].astype(int).to_numpy()
    y_test = test["target_future_withdrawal"].astype(int).to_numpy()

    print(
        f"Snapshot rows: {len(snapshot):,} | train: {len(train):,} | "
        f"validation: {len(validation):,} | test: {len(test):,}"
    )
    print(
        f"Future-withdrawal rates: train={y_train['hazard'].mean():.2%}, "
        f"validation={y_val.mean():.2%}, test={y_test.mean():.2%}"
    )

    landmark_dir = OUTPUT_DIR / f"week_{landmark}"
    landmark_dir.mkdir(parents=True, exist_ok=True)

    print("\n1/5 Creating grouped out-of-fold component predictions...")
    oof_components = create_oof_component_predictions(
        X_train, y_train, affect_train, train["id_student"].to_numpy()
    )

    print("\n2/5 Training stacking correlator...")
    meta_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    meta_model.fit(oof_components, y_train["hazard"])

    print("\n3/5 Training final component models on training partition...")
    final_models = train_component_models(X_train, y_train, affect_train)

    print("\n4/6 Calibrating on one validation subset and selecting threshold on another...")
    val_components = component_prediction_matrix(final_models, X_val)
    calibration_splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE + landmark)
    calibration_idx, threshold_idx = next(
        calibration_splitter.split(validation, y_val, groups=validation["id_student"])
    )
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(meta_model),
        method="sigmoid",
    )
    calibrator.fit(val_components[calibration_idx], y_val[calibration_idx])
    val_probabilities = calibrator.predict_proba(val_components)[:, 1]
    best_threshold, threshold_table = select_threshold(
        y_val[threshold_idx], val_probabilities[threshold_idx]
    )
    threshold_table.to_csv(landmark_dir / "validation_threshold_search.csv", index=False)

    validation_metrics = calculate_metrics(
        y_val[threshold_idx], val_probabilities[threshold_idx], best_threshold
    )
    print(f"Validation-selected threshold: {best_threshold:.3f}")
    print(
        f"Validation F1={validation_metrics['f1']:.4f}, "
        f"Precision={validation_metrics['precision']:.4f}, "
        f"Recall={validation_metrics['recall']:.4f}"
    )

    print("\n5/6 Evaluating untouched future-presentation test partition...")
    test_components = component_prediction_matrix(final_models, X_test)
    raw_test_probabilities = meta_model.predict_proba(test_components)[:, 1]
    test_probabilities = calibrator.predict_proba(test_components)[:, 1]
    test_metrics = calculate_metrics(y_test, test_probabilities, best_threshold)
    raw_test_metrics = calculate_metrics(y_test, raw_test_probabilities, best_threshold)

    test_metrics_050 = calculate_metrics(y_test, test_probabilities, 0.50)
    training_prevalence = float(y_train["hazard"].mean())
    baseline_brier = float(brier_score_loss(y_test, np.full(len(y_test), training_prevalence)))
    test_metrics["brier_skill_vs_train_prevalence"] = (
        float(1 - test_metrics["brier"] / baseline_brier) if baseline_brier > 0 else None
    )
    baseline_rows = []
    majority_class = int(training_prevalence >= 0.5)
    for baseline_name, probability in (
        ("majority_class", np.full(len(y_test), majority_class, dtype=float)),
        ("training_prevalence_probability", np.full(len(y_test), training_prevalence)),
    ):
        baseline_rows.append({
            "model": baseline_name,
            **calculate_metrics(y_test, probability, 0.50),
        })
    pd.DataFrame(baseline_rows).to_csv(landmark_dir / "test_baselines.csv", index=False)

    calibration_table = pd.DataFrame({
        "actual_future_withdrawal": y_test,
        "predicted_probability": test_probabilities,
    })
    calibration_table["probability_decile"] = pd.qcut(
        calibration_table["predicted_probability"],
        q=min(10, calibration_table["predicted_probability"].nunique()),
        duplicates="drop",
    )
    calibration_table.groupby("probability_decile", observed=True).agg(
        records=("actual_future_withdrawal", "size"),
        mean_predicted_probability=("predicted_probability", "mean"),
        observed_withdrawal_rate=("actual_future_withdrawal", "mean"),
    ).reset_index().to_csv(landmark_dir / "test_calibration.csv", index=False)

    prediction_output = test[KEYS + ["week_number", "observed_week", "target_future_withdrawal"]].copy()
    prediction_output = prediction_output.rename(columns={"target_future_withdrawal": "actual_future_withdrawal"})
    prediction_output["withdrawal_probability"] = test_probabilities
    prediction_output["predicted_withdrawn"] = (
        test_probabilities >= best_threshold
    ).astype(int)
    prediction_output["risk_category"] = pd.cut(
        test_probabilities,
        bins=[-np.inf, best_threshold, np.inf],
        labels=["No Alert", "Alert"],
    ).astype(str)
    prediction_output.to_csv(landmark_dir / "test_predictions.csv", index=False)

    validation_output = validation[KEYS + ["week_number", "target_future_withdrawal"]].copy()
    validation_output = validation_output.rename(columns={"target_future_withdrawal": "actual_future_withdrawal"})
    validation_output["withdrawal_probability"] = val_probabilities
    validation_output["calibration_fit_subset"] = False
    validation_output.loc[validation_output.index.isin(validation.iloc[calibration_idx].index), "calibration_fit_subset"] = True
    validation_output["threshold_selection_subset"] = validation_output["calibration_fit_subset"].eq(False)
    validation_output.to_csv(landmark_dir / "validation_predictions.csv", index=False)

    joblib.dump(final_models, landmark_dir / "component_models.joblib")
    joblib.dump(meta_model, landmark_dir / "stacking_correlator_raw.joblib")
    joblib.dump(calibrator, landmark_dir / "calibrated_correlator.joblib")
    joblib.dump(calibrator, landmark_dir / "learned_correlator.joblib")
    summary = {
        "landmark_week": landmark,
        "snapshot_records": int(len(snapshot)),
        "unique_students": int(snapshot["id_student"].nunique()),
        "future_withdrawal_target": "withdrawal after the end of the landmark week through course end",
        "risk_set_cutoff_day": int((landmark + 1) * 7 - 1),
        "presentation_split": {
            "training": sorted(train["code_presentation"].unique().tolist()),
            "validation": sorted(validation["code_presentation"].unique().tolist()),
            "test": sorted(test["code_presentation"].unique().tolist()),
        },
        "calibration_method": "sigmoid (Platt scaling)",
        "calibration_fit_students": int(validation.iloc[calibration_idx]["id_student"].nunique()),
        "threshold_selection_students": int(validation.iloc[threshold_idx]["id_student"].nunique()),
        "train_records": int(len(train)),
        "validation_records": int(len(validation)),
        "test_records": int(len(test)),
        "train_students": int(train["id_student"].nunique()),
        "validation_students": int(validation["id_student"].nunique()),
        "test_students": int(test["id_student"].nunique()),
        "validation_selected_threshold": float(best_threshold),
        "validation_metrics_at_selected_threshold": validation_metrics,
        "final_test_metrics_at_validation_threshold": test_metrics,
        "final_test_metrics_at_threshold_0_50": test_metrics_050,
        "raw_uncalibrated_test_metrics": raw_test_metrics,
        "test_baselines": baseline_rows,
        "training_withdrawal_prevalence": training_prevalence,
        "test_risk_category_distribution": {
            str(k): int(v)
            for k, v in prediction_output["risk_category"].value_counts().items()
        },
        "notes": [
            "Train, validation, and test are split chronologically by presentation; overlapping student IDs are purged from earlier partitions.",
            "Every enrollment has a complete calendar-week panel, including zero-activity weeks.",
            "Students not registered by the landmark or already unregistered are excluded from its risk set.",
            "The positive label means withdrawal after the landmark and by course end.",
            "Sigmoid calibration fits on one validation subset; threshold selection uses a disjoint student subset.",
            "The stacking correlator is trained using out-of-fold component predictions.",
            "Affect values are synthetic activity-derived proxies, not measured emotions.",
            "The calibrator and threshold are evaluated on the latest presentation held out from all fitting and tuning.",
            "Synthetic affect proxies remain formula-derived activity indicators, not measured emotions.",
        ],
    }
    write_json(landmark_dir / "landmark_summary.json", summary)
    write_json(landmark_dir / "model_config.json", summary)

    print("\nFINAL TEST RESULTS")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1:        {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {test_metrics['pr_auc']:.4f}")
    print(f"Brier:     {test_metrics['brier']:.4f}")
    print(f"Confusion: {test_metrics['confusion_matrix']}")

    return {
        "landmark_week": landmark,
        "threshold": best_threshold,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_brier": test_metrics["brier"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_expected_calibration_error": test_metrics["expected_calibration_error"],
        "test_brier_skill_vs_train_prevalence": test_metrics["brier_skill_vs_train_prevalence"],
        "test_majority_accuracy": baseline_rows[0]["accuracy"],
        "test_prevalence_baseline_brier": baseline_rows[1]["brier"],
        "test_records": len(test),
        "test_students": test["id_student"].nunique(),
    }


def main():
    """Run the full prospective evaluation for every configured landmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    split_map, split_info = create_presentation_split(df)

    results = []
    for landmark in LANDMARKS:
        results.append(run_landmark(df, landmark, split_map))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_DIR / "overall_summary.csv", index=False)
    write_json(
        OUTPUT_DIR / "overall_summary.json",
        {"presentation_split": split_info, "landmarks": summary_df.to_dict(orient="records")},
    )

    print("\n" + "=" * 72)
    print("ALL LANDMARKS COMPLETED")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
