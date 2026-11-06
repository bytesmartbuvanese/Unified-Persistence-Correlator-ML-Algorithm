"""Serve persistence estimates from the compact synthetic-data model.

This is the inference API for the separate ``drop`` training workflow, not
the OULAD landmark service in the parent directory's ``api.py``. It validates
one row of institutional, engagement, academic, financial, and feedback
features, loads the serialized model artifact, calculates component risk
scores using the saved standardization values and weights, and combines them
with a logistic classifier. The response reports estimated persistence
probability and score together with the component risks.

The default model path is ``models/unified_persistence_model.pkl`` relative
to the process working directory; set ``MODEL_PATH`` to override it. The
artifact is produced by ``drop/train.py`` from
``dataset/unified_persistence_dataset.csv``. Run the server from the ``drop``
directory with ``uvicorn api:app --reload`` after training. ``GET /health``
reports model availability and ``POST /predict`` accepts the validated
``StudentFeatures`` schema. This module performs inference only; it does not
generate a dataset or fit model weights.
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------

class StudentFeatures(BaseModel):
    """Validated institutional, engagement, and interaction signals.

    All values describe a student's current observation. Constraints enforce
    the ranges expected by the training pipeline before inference begins.

    Attributes
    ----------
    LMS_Login_Count : float
        Number of learning-management-system logins.
    Weekly_Active_Hours : float
        Weekly hours of active participation.
    Assignment_Submission_Rate : float
        Submitted-assignment proportion, from 0 to 1.
    Lecture_Attendance_Rate : float
        Lecture attendance percentage, from 0 to 100.
    Feedback_Sentiment_Score : float
        Feedback sentiment, from -1 to 1.
    Academic_Stress_Survey_Score : float
        Survey-derived stress score, from 0 to 100.
    Support_Request_Count : float
        Number of support requests.
    Counseling_Session_Count : float
        Number of counseling sessions.
    Current_GPA : float
        Current GPA, from 0 to 10.
    Failed_Subject_Count : float
        Number of failed subjects.
    Outstanding_Fee : float
        Outstanding tuition balance.
    Tuition_Fee : float
        Tuition amount; must be greater than zero.
    Scholarship_Amount : float
        Scholarship amount applied to the student's tuition.
    """

    LMS_Login_Count: float = Field(ge=0)
    Weekly_Active_Hours: float = Field(ge=0)
    Assignment_Submission_Rate: float = Field(ge=0, le=1)
    Lecture_Attendance_Rate: float = Field(ge=0, le=100)
    Feedback_Sentiment_Score: float = Field(ge=-1, le=1)
    Academic_Stress_Survey_Score: float = Field(ge=0, le=100)
    Support_Request_Count: float = Field(ge=0)
    Counseling_Session_Count: float = Field(ge=0)
    Current_GPA: float = Field(ge=0, le=10)
    Failed_Subject_Count: float = Field(ge=0)
    Outstanding_Fee: float = Field(ge=0)
    Tuition_Fee: float = Field(gt=0)
    Scholarship_Amount: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Model output for an individual persistence estimate.

    Attributes
    ----------
    persistence_probability : float
        Estimated probability of continued enrollment, between 0 and 1.
    persistence_score : float
        The probability expressed on a 0-to-100 scale.
    component_risk_scores : dict[str, float]
        Risk estimates for the model's component domains, each on a
        0-to-100 scale.
    interpretation : str
        Brief explanation of the persistence score's direction.
    """

    persistence_probability: float
    persistence_score: float
    component_risk_scores: dict[str, float]
    interpretation: str


app = FastAPI(
    title="Unified Persistence Correlator API",
    description="Predicts the likelihood of continued enrollment from institutional signals.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model() -> dict:
    """Load and cache the trained model artifact.

    The artifact path is read from ``MODEL_PATH`` or defaults to
    ``models/unified_persistence_model.pkl``.

    Returns
    -------
    dict
        Serialized component regressors, scaling parameters, and classifier
        weights required by :func:`predict`.

    Raises
    ------
    FileNotFoundError
        If the configured model artifact does not exist.
    """
    model_path = Path(os.environ.get("MODEL_PATH", "models/unified_persistence_model.pkl"))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python train.py` first or set MODEL_PATH."
        )
    with model_path.open("rb") as file:
        return pickle.load(file)


def predict(features: StudentFeatures) -> PredictionResponse:
    """Estimate continued-enrollment probability from validated features.

    Component inputs are standardized with the statistics saved during
    training, transformed into risk scores, then combined by the classifier.

    Parameters
    ----------
    features : StudentFeatures
        Validated current-observation values for one student.

    Returns
    -------
    PredictionResponse
        Persistence probability, 0-to-100 score, component risks, and
        interpretation.

    Raises
    ------
    HTTPException
        With status 503 when the trained model artifact is unavailable.
    """
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    values = features.model_dump()
    component_risks: dict[str, float] = {}
    for name in model["component_order"]:
        component = model["components"][name]
        raw = np.asarray([[values[column] for column in component["features"]]], dtype=float)
        scaled = (raw - component["mean"]) / component["scale"]
        design = np.column_stack((np.ones(len(raw)), scaled))
        risk = float(np.clip(design @ component["weights"], 0, 100)[0])
        component_risks[name] = risk

    component_vector = np.asarray([[component_risks[name] for name in model["component_order"]]])
    standardized = (component_vector - model["component_mean"]) / model["component_scale"]
    design = np.column_stack((np.ones(len(standardized)), standardized))
    logit = float((design @ model["classifier_weights"])[0])
    probability = float(1 / (1 + np.exp(-np.clip(logit, -35, 35))))
    score = 100 * probability
    return PredictionResponse(
        persistence_probability=probability,
        persistence_score=score,
        component_risk_scores=component_risks,
        interpretation="Higher score indicates a greater model-estimated likelihood of continued enrollment.",
    )


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the model artifact can be loaded.

    Returns
    -------
    dict[str, str]
        ``{"status": "ok"}`` when the model is available, otherwise
        ``{"status": "model_not_loaded"}``.
    """
    try:
        load_model()
    except FileNotFoundError:
        return {"status": "model_not_loaded"}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_persistence(features: StudentFeatures) -> PredictionResponse:
    """Handle a prediction request through the registered API endpoint.

    Parameters
    ----------
    features : StudentFeatures
        Request body validated against the inference feature schema.

    Returns
    -------
    PredictionResponse
        The model's persistence estimate.
    """
    return predict(features)
