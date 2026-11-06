"""Serve landmark-specific OULAD withdrawal-risk predictions over HTTP.

Purpose and workflow
--------------------
This module defines a FastAPI application for the time-aware early-warning
models trained by :mod:`strict_time_aware_early_warning_v2`. At startup it
loads the serialized model bundle and prepares artifacts for landmark weeks
2, 4, and 6. For a request, it validates one feature snapshot, runs the
engagement, affect, and hazard component models, combines those predictions
with the saved stacking correlator, and compares the resulting withdrawal
probability with the threshold selected during validation. The response
includes the probability-derived persistence score and component estimates.

Input and model artifacts
-------------------------
``POST /predict`` accepts one snapshot and a matching ``landmark_week`` (2,
4, or 6). The snapshot must provide every field in ``FEATURE_COLUMNS`` and its
``week_number`` must equal the selected landmark. Optional student and course
identifiers are carried through to the response; they are not predictors.

By default, the application reads ``unified_model.joblib`` beside this file.
Set ``PERSISTENCE_MODEL_FILE`` to use a different model bundle path. The bundle
must contain the component models, learned correlator, and model configuration
for all three supported landmark weeks; ``combine_models.py`` can package the
per-week artifacts produced by the training script into the unified file.

Routes and execution
--------------------
``GET /`` reports service information, ``GET /health`` reports whether all
three landmark bundles loaded, and ``POST /predict`` returns the prediction
for the requested week. Invalid snapshots or incompatible model artifacts are
reported as client errors; predictions are unavailable until model loading
succeeds.

Run from the project directory with
``uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload`` and open
``/docs`` for the interactive API schema. This service produces model
estimates from the supplied snapshot; it does not fetch student records or
compute upstream features itself.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = Path(
    os.getenv("PERSISTENCE_MODEL_FILE", BASE_DIR / "unified_model.joblib")
)

LANDMARKS = (2, 4, 6)

FEATURE_COLUMNS = [
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

AFFECT_COLUMNS = [
    "motivation",
    "confidence",
    "frustration",
    "stress",
    "disengagement",
    "wellbeing",
]

# The trained model_config.json contains the validation-selected threshold.
DEFAULT_THRESHOLDS = {
    2: 0.50,
    4: 0.50,
    6: 0.50,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("persistence_api")


# ---------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------

class FeatureSnapshot(BaseModel):
    """Model inputs observed through one landmark week.

    The fields correspond to the model's fixed predictor schema and must be
    computed only from information available at the selected landmark.
    """

    model_config = ConfigDict(extra="forbid")

    week_number: float
    cumulative_active_weeks: float
    cumulative_active_days: float
    cumulative_interactions: float
    cumulative_clicks: float
    days_since_registration: float
    average_clicks_per_active_week: float
    studied_credits: float
    previous_week_clicks: float
    weekly_total_clicks: float
    click_change_from_previous_week: float
    weekly_interaction_records: float
    weekly_active_days: float
    num_of_prev_attempts: float
    hazard_low_active_days: float


class PersistencePredictionRequest(BaseModel):
    """Request one withdrawal-risk prediction from a single current snapshot.

    Optional identifiers are copied to the response for record association;
    they are not model features.
    """

    model_config = ConfigDict(extra="forbid")

    student_id: str | None = None
    code_module: str | None = None
    code_presentation: str | None = None

    landmark_week: Literal[2, 4, 6]
    snapshot: FeatureSnapshot


class LandmarkPrediction(BaseModel):
    """Prediction scores, threshold decision, and component estimates."""

    landmark_week: int
    withdrawal_risk_score: float
    persistence_score: float
    decision_threshold: float
    withdrawal_alert: bool
    risk_category: str
    component_estimates: dict[str, float]


class PersistencePredictionResponse(BaseModel):
    """Response envelope for a landmark-specific prediction."""

    student_id: str | None
    code_module: str | None
    code_presentation: str | None
    predictions: dict[str, LandmarkPrediction]
    note: str


# ---------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty mapping when the file is absent.

    Parameters
    ----------
    path : pathlib.Path
        JSON file to read.

    Returns
    -------
    dict
        Parsed JSON contents, or an empty dictionary if ``path`` is missing.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_component_model(component_models: Any, aliases: tuple[str, ...]) -> Any:
    """Find a fitted component model using normalized alias keys.

    Parameters
    ----------
    component_models : Any
        Expected to be the saved dictionary of fitted component models.
    aliases : tuple of str
        Candidate names accepted for the requested component.

    Returns
    -------
    Any
        The first model whose normalized key matches an alias.

    Raises
    ------
    TypeError
        If ``component_models`` is not a dictionary.
    KeyError
        If none of the aliases is present.
    """

    if not isinstance(component_models, dict):
        raise TypeError(
            "component_models.joblib must contain a dictionary of fitted models. "
            f"Received: {type(component_models).__name__}"
        )

    normalized = {
        str(key).lower().replace("-", "_").replace(" ", "_"): value
        for key, value in component_models.items()
    }

    for alias in aliases:
        key = alias.lower().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return normalized[key]

    available = ", ".join(map(str, component_models.keys()))
    raise KeyError(
        f"Could not find model using aliases {aliases}. "
        f"Available keys: [{available}]"
    )


def load_landmark_artifacts(week: int, bundle: dict[str, Any]) -> dict[str, Any]:
    """Extract component models, correlator, and threshold for one landmark.

    Parameters
    ----------
    week : int
        Supported landmark week.
    bundle : dict
        Serialized artifacts for that landmark.

    Returns
    -------
    dict
        Normalized model references, decision threshold, and configuration.
    """

    # Support the standard v2 bundle and common key naming variations.
    component_bundle = bundle["component_models"]
    correlator = bundle["correlator"]
    config = bundle.get("config", {})

    engagement_model = find_component_model(
        component_bundle,
        ("engagement_model", "engagement", "engagement_regressor"),
    )
    affect_model = find_component_model(
        component_bundle,
        ("affect_model", "affect", "affect_regressor"),
    )
    hazard_model = find_component_model(
        component_bundle,
        ("hazard_model", "hazard", "withdrawal_model", "hazard_classifier"),
    )

    threshold = DEFAULT_THRESHOLDS[week]
    for key in (
        "validation_selected_threshold",
        "selected_threshold",
        "optimal_threshold",
        "decision_threshold",
        "threshold",
    ):
        value = config.get(key)
        if isinstance(value, (int, float)) and 0 < float(value) < 1:
            threshold = float(value)
            break

    logger.info("Loaded Week %s artifacts from unified model", week)

    return {
        "engagement": engagement_model,
        "affect": affect_model,
        "hazard": hazard_model,
        "correlator": correlator,
        "threshold": threshold,
        "config": config,
    }


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------

def positive_class_probability(model: Any, features: pd.DataFrame | np.ndarray) -> float:
    """Return one row's probability for the positive class.

    The model's class labels are consulted when available; otherwise column
    index one is treated as positive. DataFrame inputs are converted to arrays
    only when the estimator does not advertise feature-name support.

    Parameters
    ----------
    model : Any
        Fitted classifier exposing ``predict_proba``.
    features : pandas.DataFrame or numpy.ndarray
        Feature row to score.

    Returns
    -------
    float
        Probability associated with the positive class.
    """
    if isinstance(features, pd.DataFrame):
        has_feature_names = getattr(model, "feature_names_in_", None) is not None
        if not has_feature_names and hasattr(model, "named_steps"):
            first_step = next(iter(model.named_steps.values()), None)
            has_feature_names = getattr(first_step, "feature_names_in_", None) is not None
        if not has_feature_names:
            features = features.to_numpy()

    probabilities = model.predict_proba(features)
    classes = getattr(model, "classes_", None)

    if classes is not None:
        class_values = list(classes)
        if 1 in class_values:
            positive_index = class_values.index(1)
        else:
            positive_index = len(class_values) - 1
    else:
        positive_index = 1

    return float(np.asarray(probabilities)[0, positive_index])


def build_meta_features(
    correlator: Any,
    engagement_value: float,
    affect_values: np.ndarray,
    hazard_probability: float,
) -> pd.DataFrame:
    """
    Build the stacking input in the saved correlator's expected column order.

    When the correlator exposes ``feature_names_in_``, common naming aliases
    are mapped to their component predictions. Otherwise the training order
    is used: engagement, six affect outputs, then hazard probability.

    Parameters
    ----------
    correlator : Any
        Fitted stacking estimator.
    engagement_value : float
        Predicted engagement score.
    affect_values : numpy.ndarray
        Affect proxy predictions in ``AFFECT_COLUMNS`` order.
    hazard_probability : float
        Component classifier's positive-class probability.

    Returns
    -------
    pandas.DataFrame
        Single-row meta-feature table with the correlator's expected columns.

    Raises
    ------
    ValueError
        If named correlator features cannot be constructed from the supplied
        component predictions.
    """

    values_by_name: dict[str, float] = {
        "engagement": engagement_value,
        "engagement_prediction": engagement_value,
        "engagement_pred": engagement_value,
        "predicted_engagement": engagement_value,
        "predicted_engagement_score": engagement_value,
        "engagement_score": engagement_value,
        "hazard": hazard_probability,
        "hazard_probability": hazard_probability,
        "hazard_prediction": hazard_probability,
        "hazard_pred": hazard_probability,
        "predicted_hazard": hazard_probability,
        "predicted_hazard_probability": hazard_probability,
        "withdrawal_risk_probability": hazard_probability,
    }

    for name, value in zip(AFFECT_COLUMNS, affect_values):
        numeric = float(value)
        values_by_name[name] = numeric
        values_by_name[f"predicted_{name}"] = numeric
        values_by_name[f"affect_{name}"] = numeric
        values_by_name[f"{name}_prediction"] = numeric
        values_by_name[f"{name}_pred"] = numeric

    model_feature_names = getattr(correlator, "feature_names_in_", None)

    if model_feature_names is not None:
        ordered_names = [str(name) for name in model_feature_names]
        row: dict[str, float] = {}

        for name in ordered_names:
            normalized = name.lower().replace("-", "_").replace(" ", "_")
            if normalized not in values_by_name:
                raise ValueError(
                    "The correlator expects a meta-feature that the API "
                    f"cannot construct: '{name}'. Check the training script's "
                    "stacking feature names and update build_meta_features()."
                )
            row[name] = values_by_name[normalized]

        return pd.DataFrame([row], columns=ordered_names)

    # Fallback order must match the order used to train the correlator.
    ordered_values = [
        engagement_value,
        *[float(value) for value in affect_values],
        hazard_probability,
    ]
    return pd.DataFrame(
        [ordered_values],
        columns=[
            "engagement_prediction",
            "motivation_prediction",
            "confidence_prediction",
            "frustration_prediction",
            "stress_prediction",
            "disengagement_prediction",
            "wellbeing_prediction",
            "hazard_probability",
        ],
    )


def predict_one_landmark(
    week: int,
    snapshot: FeatureSnapshot,
    artifacts: dict[str, Any],
) -> LandmarkPrediction:
    """Score one feature snapshot using the specified landmark artifacts.

    Component outputs are assembled into the stacking model's meta-feature
    row, then the withdrawal probability is clipped to the valid probability
    interval and compared with the validation-selected threshold.

    Parameters
    ----------
    week : int
        Landmark week associated with the loaded artifacts.
    snapshot : FeatureSnapshot
        Validated feature values for that week.
    artifacts : dict
        Fitted component estimators, correlator, and decision threshold.

    Returns
    -------
    LandmarkPrediction
        Withdrawal and persistence scores, alert decision, and component
        estimates.

    Raises
    ------
    ValueError
        If the affect estimator returns a different number of outputs than
        expected or its outputs cannot be mapped to the correlator.
    """
    input_row = snapshot.model_dump()
    feature_frame = pd.DataFrame(
        [[input_row[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    # Component predictions: one engagement estimate, six affect proxies,
    # and one hazard probability.
    engagement_value = float(
        np.asarray(artifacts["engagement"].predict(feature_frame)).reshape(-1)[0]
    )

    affect_values = np.asarray(
        artifacts["affect"].predict(feature_frame)
    ).reshape(-1)

    if len(affect_values) != len(AFFECT_COLUMNS):
        raise ValueError(
            "Affect model returned "
            f"{len(affect_values)} outputs; expected {len(AFFECT_COLUMNS)}. "
            "Check AFFECT_COLUMNS against the training target order."
        )

    hazard_probability = positive_class_probability(
        artifacts["hazard"], feature_frame
    )

    meta_frame = build_meta_features(
        artifacts["correlator"],
        engagement_value,
        affect_values,
        hazard_probability,
    )

    # Stacking correlator's positive class is eventual withdrawal.
    withdrawal_probability = positive_class_probability(
        artifacts["correlator"], meta_frame
    )
    withdrawal_probability = float(
        np.clip(withdrawal_probability, 0.0, 1.0)
    )

    threshold = float(artifacts["threshold"])
    withdrawal_alert = withdrawal_probability >= threshold

    # Use the validation-selected operating threshold rather than inventing
    # additional risk bands that were not established during model validation.
    category = "Withdrawal Alert" if withdrawal_alert else "No Withdrawal Alert"

    component_estimates = {
        "engagement_score_estimate": engagement_value,
        **{
            f"{name}_proxy_estimate": float(value)
            for name, value in zip(AFFECT_COLUMNS, affect_values)
        },
        "base_hazard_probability": hazard_probability,
    }

    return LandmarkPrediction(
        landmark_week=week,
        withdrawal_risk_score=withdrawal_probability,
        persistence_score=1.0 - withdrawal_probability,
        decision_threshold=threshold,
        withdrawal_alert=withdrawal_alert,
        risk_category=category,
        component_estimates=component_estimates,
    )


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Unified Persistence Correlator API",
    description=(
        "Inference for one currently available Week 2, Week 4 or Week 6 "
        "time-aware withdrawal-risk snapshot."
    ),
    version="1.0.0",
)

MODEL_REGISTRY: dict[int, dict[str, Any]] = {}
MODEL_LOAD_ERROR: str | None = None


@app.on_event("startup")
def load_all_models() -> None:
    """Load and register all landmark models during application startup.

    Any loading failure is retained in ``MODEL_LOAD_ERROR`` and leaves the
    registry empty so health and prediction endpoints can report readiness.
    """

    global MODEL_REGISTRY, MODEL_LOAD_ERROR

    try:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Unified model file not found: {MODEL_FILE}"
            )

        logger.info("Loading unified model from %s ...", MODEL_FILE)
        unified_bundle = joblib.load(MODEL_FILE)

        loaded: dict[int, dict[str, Any]] = {}
        for week in LANDMARKS:
            if week not in unified_bundle:
                raise KeyError(f"Week {week} not found in unified model bundle")
            loaded[week] = load_landmark_artifacts(week, unified_bundle[week])

        MODEL_REGISTRY = loaded
        MODEL_LOAD_ERROR = None
        logger.info("All three landmark model bundles loaded successfully.")

    except Exception as exc:
        MODEL_REGISTRY = {}
        MODEL_LOAD_ERROR = str(exc)
        logger.exception("Model loading failed.")


@app.get("/")
def root() -> dict[str, str]:
    """Return service status and the primary prediction route."""
    return {
        "service": "Unified Persistence Correlator API",
        "status": "running",
        "prediction_endpoint": "POST /predict",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Report model-loading readiness and the configured artifact path."""
    healthy = len(MODEL_REGISTRY) == len(LANDMARKS)
    return {
        "status": "healthy" if healthy else "model_loading_error",
        "loaded_landmarks": sorted(MODEL_REGISTRY.keys()),
        "expected_landmarks": list(LANDMARKS),
        "model_file": str(MODEL_FILE),
        "error": MODEL_LOAD_ERROR,
    }


@app.post(
    "/predict",
    response_model=PersistencePredictionResponse,
)
def predict(
    request: PersistencePredictionRequest,
) -> PersistencePredictionResponse:
    """Predict withdrawal risk for the request's single landmark snapshot.

    Returns HTTP 503 until all landmark bundles are loaded, HTTP 422 for
    invalid snapshots or incompatible model artifacts, and HTTP 500 for
    unexpected inference failures.
    """
    if len(MODEL_REGISTRY) != len(LANDMARKS):
        raise HTTPException(
            status_code=503,
            detail=(
                "The three landmark models are not ready. "
                f"Model loading error: {MODEL_LOAD_ERROR}"
            ),
        )

    try:
        results: dict[str, LandmarkPrediction] = {}

        week = request.landmark_week
        snapshot_week = float(request.snapshot.week_number)
        if not np.isclose(snapshot_week, week, atol=1e-6):
            raise ValueError(
                f"The snapshot week_number must equal landmark_week ({week}); got {snapshot_week:g}."
            )
        results[f"week_{week}"] = predict_one_landmark(
            week,
            request.snapshot,
            MODEL_REGISTRY[week],
        )

        return PersistencePredictionResponse(
            student_id=request.student_id,
            code_module=request.code_module,
            code_presentation=request.code_presentation,
            predictions=results,
            note=f"Prediction generated for landmark week {week}.",
        )

    except (ValueError, KeyError, TypeError) as exc:
        logger.exception("Prediction input or model compatibility error.")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected inference error.")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc
