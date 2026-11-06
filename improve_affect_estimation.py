"""Add synthetic affect proxies to an engineered weekly feature dataset.

The script searches the configured candidate paths for
``engineered_weekly_features.csv`` and derives six 0-to-100 proxy columns
(motivation, confidence, frustration, stress, disengagement, and wellbeing)
from available weekly activity and, when present, assessment features. It
also marks the values as synthetic, produces a distribution/correlation
diagnostic CSV, and writes metadata describing the output. The source file is
read as input and is not overwritten.

Run ``python improve_affect_estimation.py`` from the project directory after
feature engineering, or update ``INPUT_CANDIDATES`` if the weekly table is in
another location. The enriched dataset is written to
``engineered_weekly_features_with_affect.csv`` and diagnostics to
``trained_models/affect_diagnostics/``. These formula-derived values are
activity proxies, not measured emotions or validated psychological
assessments.
"""

# improve_affect_estimation.py

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_CANDIDATES = [
    Path("engineered_data/engineered_weekly_features.csv"),
    Path("data/processed/engineered_weekly_features.csv"),
    Path("processed/engineered_weekly_features.csv"),
]

OUTPUT_FILE = Path("engineered_weekly_features_with_affect.csv")
REPORT_DIR = Path("trained_models") / "affect_diagnostics"

AFFECT_COLUMNS = [
    "affect_motivation_estimate",
    "affect_confidence_estimate",
    "affect_frustration_estimate",
    "affect_stress_estimate",
    "affect_disengagement_estimate",
    "affect_wellbeing_proxy",
]


# ---------------------------------------------------------
# FIND INPUT DATASET
# ---------------------------------------------------------

def find_input_file():
    """Return the first existing weekly feature dataset candidate.

    Returns
    -------
    pathlib.Path
        First existing path from ``INPUT_CANDIDATES``.

    Raises
    ------
    FileNotFoundError
        If none of the configured input paths exists.
    """
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find engineered_weekly_features.csv.\n"
        "Place this script in your project root or update INPUT_CANDIDATES."
    )


# ---------------------------------------------------------
# FEATURE UTILITIES
# ---------------------------------------------------------

def get_numeric_column(df, possible_names):
    """Return the first matching column converted to numeric values.

    Unparseable and missing entries are filled with zero. If no candidate
    column exists, return ``None``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table.
    possible_names : iterable of str
        Candidate column names in priority order.

    Returns
    -------
    pandas.Series or None
        Numeric values aligned with ``df``, or ``None`` when unmatched.
    """

    for name in possible_names:
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce")
            return values.fillna(0)

    return None


def bounded_score(series, inverse=False):
    """Map feature values to a bounded score using feature-specific units.

    Assessment scores, active days, interactions, clicks, and week counts use
    fixed scales; other values use a centered hyperbolic-tangent transform.
    Optional inversion maps larger source values to smaller scores.

    Parameters
    ----------
    series : pandas.Series
        Values to normalize; its name selects the fixed-scale rule.
    inverse : bool, default=False
        Whether to return the complement of the normalized score.

    Returns
    -------
    pandas.Series
        Scores clipped to the inclusive interval from zero to one.
    """
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(0)
    name = str(getattr(series, "name", "")).lower()
    if "assessment_score" in name:
        result = values.clip(0, 100) / 100
    elif "active_day" in name:
        result = values.clip(0, 7) / 7
    elif "interaction" in name:
        positive = values.clip(lower=0)
        result = positive / (positive + 20)
    elif "cumulative" in name and "click" in name:
        positive = values.clip(lower=0)
        result = positive / (positive + 1000)
    elif "click" in name or "activity" in name:
        positive = values.clip(lower=0)
        result = positive / (positive + 100)
    elif "week" in name:
        positive = values.clip(lower=0)
        result = positive / (positive + 10)
    else:
        result = 0.5 + 0.5 * np.tanh(values / 100)

    if inverse:
        result = 1 - result

    return result.clip(0, 1)


def blend_features(df, feature_definitions):
    """
    Combine multiple available features using weighted fixed-scale scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Source rows containing zero or more candidate features.
    feature_definitions : iterable of tuple
        Each tuple contains candidate column names, a weight, and an inversion
        flag: ``(possible_column_names, weight, inverse)``. Definitions with
        no available column are skipped and remaining weights are normalized.

    Returns
    -------
    pandas.Series
        Weighted scores clipped to ``[0, 1]``. If no feature is available,
        returns a neutral score of 0.5 for every row.
    """

    scores = []
    weights = []

    for possible_names, weight, inverse in feature_definitions:
        series = get_numeric_column(df, possible_names)

        if series is None:
            continue

        scores.append(bounded_score(series, inverse=inverse))
        weights.append(weight)

    if not scores:
        return pd.Series(0.5, index=df.index)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    combined = sum(
        score * weight
        for score, weight in zip(scores, weights)
    )

    return combined.clip(0, 1)


def calculate_activity_decline(df):
    """Estimate a bounded decline signal from current and prior activity.

    Current-versus-previous values take precedence when both are available;
    otherwise a signed change feature is used. If neither source is present,
    return a neutral midpoint signal.

    Parameters
    ----------
    df : pandas.DataFrame
        Weekly records containing one or more recognized activity columns.

    Returns
    -------
    pandas.Series
        Per-row decline signal in the inclusive interval from zero to one.
    """

    current = get_numeric_column(
        df,
        ["weekly_total_clicks", "total_clicks", "weekly_clicks"]
    )

    previous = get_numeric_column(
        df,
        ["previous_week_clicks", "previous_week_total_clicks"]
    )

    change = get_numeric_column(
        df,
        ["click_change_from_previous_week", "weekly_click_change"]
    )

    if current is not None and previous is not None:
        denominator = previous.abs().replace(0, 1)
        decline = ((previous - current) / denominator).clip(0, 1)
        return decline

    if change is not None:
        return (change < 0).astype(float) * bounded_score(
            change.abs()
        )

    return pd.Series(0.5, index=df.index)


# ---------------------------------------------------------
# GENERATE SYNTHETIC AFFECT PROXIES
# ---------------------------------------------------------

def generate_affect_proxies(df):
    """Add six formula-derived affect proxy columns to a weekly dataset.

    Activity intensity, consistency, recent stability, decline, and optional
    assessment performance are combined using fixed weights. The input frame
    is updated in place and returned; proxy values are stored on a 0-100 scale.

    Parameters
    ----------
    df : pandas.DataFrame
        Weekly feature records.

    Returns
    -------
    pandas.DataFrame
        The input table with synthetic proxy columns and a data-type marker.
    """

    # Activity intensity
    activity_intensity = blend_features(
        df,
        [
            (["weekly_total_clicks", "total_clicks", "weekly_clicks"], 0.35, False),
            (["weekly_interaction_records", "weekly_interactions"], 0.25, False),
            (["weekly_active_days", "active_days"], 0.20, False),
            (["average_clicks_per_active_week"], 0.20, False),
        ],
    )

    # Consistency across observed weeks
    consistency = blend_features(
        df,
        [
            (["cumulative_active_weeks"], 0.35, False),
            (["cumulative_active_days"], 0.25, False),
            (["cumulative_interactions"], 0.20, False),
            (["average_clicks_per_active_week"], 0.20, False),
        ],
    )

    # Recent activity stability
    recent_stability = blend_features(
        df,
        [
            (["previous_week_clicks"], 0.35, False),
            (["weekly_total_clicks", "total_clicks"], 0.35, False),
            (["weekly_interaction_records"], 0.15, False),
            (["weekly_active_days"], 0.15, False),
        ],
    )

    # Higher value means more evidence of a recent activity decline
    decline = calculate_activity_decline(df).clip(0, 1)

    # Low activity signal
    low_activity = blend_features(
        df,
        [
            (["weekly_total_clicks", "total_clicks"], 0.35, True),
            (["weekly_active_days", "active_days"], 0.35, True),
            (["weekly_interaction_records"], 0.30, True),
        ],
    )

    # Optional academic performance signal
    performance = get_numeric_column(
        df,
        [
            "average_assessment_score",
            "mean_assessment_score",
            "maximum_assessment_score",
        ],
    )

    if performance is not None:
        performance_score = bounded_score(performance)
    else:
        # Do not fabricate assessment performance if it is unavailable.
        performance_score = consistency

    # Motivation proxy: activity, consistency and recent stability
    motivation = (
        0.40 * activity_intensity
        + 0.35 * consistency
        + 0.25 * recent_stability
    )

    # Confidence proxy: consistency and available academic performance
    confidence = (
        0.45 * consistency
        + 0.35 * performance_score
        + 0.20 * recent_stability
    )

    # Frustration proxy: low activity and declining activity
    frustration = (
        0.40 * low_activity
        + 0.40 * decline
        + 0.20 * (1 - performance_score)
    )

    # Stress proxy: declining activity, low activity and instability
    stress = (
        0.40 * decline
        + 0.35 * low_activity
        + 0.25 * (1 - recent_stability)
    )

    # Disengagement proxy: low activity and weak consistency
    disengagement = (
        0.45 * low_activity
        + 0.35 * (1 - consistency)
        + 0.20 * (1 - activity_intensity)
    )

    # Wellbeing proxy: consistency, activity and inverse stress
    wellbeing = (
        0.35 * consistency
        + 0.30 * activity_intensity
        + 0.35 * (1 - stress)
    )

    proxy_scores = {
        "affect_motivation_estimate": motivation,
        "affect_confidence_estimate": confidence,
        "affect_frustration_estimate": frustration,
        "affect_stress_estimate": stress,
        "affect_disengagement_estimate": disengagement,
        "affect_wellbeing_proxy": wellbeing,
    }

    for column, values in proxy_scores.items():
        # Store values on a 0-100 scale.
        df[column] = (values.clip(0, 1) * 100).round(4)

    df["affect_data_type"] = "synthetic_activity_derived_proxy"

    return df


# ---------------------------------------------------------
# VALIDATE GENERATED PROXIES
# ---------------------------------------------------------

def create_diagnostic_report(df):
    """Summarize distributions and optional activity correlations for proxies.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing every column listed in ``AFFECT_COLUMNS``.

    Returns
    -------
    pandas.DataFrame
        One row per proxy with descriptive statistics and, when available,
        correlation with weekly active days.
    """

    rows = []

    for column in AFFECT_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")

        rows.append({
            "feature": column,
            "count": int(values.count()),
            "unique_values": int(values.nunique()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
        })

    report = pd.DataFrame(rows)

    # Correlation with weekly active days is a diagnostic only.
    if "weekly_active_days" in df.columns:
        report["correlation_with_weekly_active_days"] = [
            df[column].corr(
                pd.to_numeric(
                    df["weekly_active_days"],
                    errors="coerce"
                )
            )
            for column in AFFECT_COLUMNS
        ]

    return report


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    """Load weekly features, regenerate proxies, validate, and save outputs.

    Writes the enriched feature table, a distribution report, and metadata
    describing the synthetic nature and scale of the generated columns.
    """

    input_file = find_input_file()

    print("=" * 65)
    print("SYNTHETIC AFFECT PROXY REGENERATION")
    print("=" * 65)

    print(f"\nReading: {input_file}")

    df = pd.read_csv(input_file, low_memory=False)

    print(f"Input shape: {df.shape}")

    # Preserve the original file and replace only the affect columns
    # in the new output dataset.
    df = generate_affect_proxies(df)

    # Validate that every generated target has meaningful variation.
    for column in AFFECT_COLUMNS:
        if df[column].nunique() <= 1:
            raise ValueError(
                f"{column} is still constant. "
                "Check whether the source activity columns contain variation."
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report = create_diagnostic_report(df)

    report_path = REPORT_DIR / "affect_proxy_distribution_report.csv"
    report.to_csv(report_path, index=False)

    df.to_csv(OUTPUT_FILE, index=False)

    metadata = {
        "input_file": str(input_file),
        "output_file": str(OUTPUT_FILE),
        "rows": int(len(df)),
        "affect_columns": AFFECT_COLUMNS,
        "scale": "0-100",
        "data_type": "synthetic_activity_derived_proxy",
        "interpretation": (
            "These are simulated activity-derived indicators. "
            "They are not measured emotions or clinically validated "
            "psychological states."
        ),
        "important_note": (
            "Do not interpret a model's ability to reproduce these "
            "formula-generated targets as validated affect prediction."
        ),
    }

    metadata_path = REPORT_DIR / "affect_proxy_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nAFFECT PROXY DISTRIBUTION")
    print(report.to_string(index=False))

    print("\nGenerated files:")
    print(f"1. {OUTPUT_FILE}")
    print(f"2. {report_path}")
    print(f"3. {metadata_path}")

    print("\nValidation:")
    print("All six affect proxy columns have non-constant values.")
    print("Original dataset was not overwritten.")

    print("\nIMPORTANT:")
    print(
        "These are synthetic activity-derived proxies, "
        "not measured student emotions."
    )

    print("\nCompleted successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)
