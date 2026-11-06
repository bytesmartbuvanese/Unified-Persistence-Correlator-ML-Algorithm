
# ---------------------------------------------------------------------------
# Purpose and usage
# ---------------------------------------------------------------------------
# Fit the compact persistence model consumed by ``drop/api.py``. The active
# workflow reads synthetic student observations, validates the feature and
# target columns, splits whole students into training and validation sets,
# fits standardized component-risk regressors and a logistic classifier, and
# writes the model weights and scaling data as a pickle artifact.
#
# From this directory, generate the input with ``python generate_synthetic_data.py``
# and train with ``python train.py``. Defaults are
# ``dataset/unified_persistence_dataset.csv`` and
# ``models/unified_persistence_model.pkl``; ``--data``, ``--output``,
# ``--validation-fraction``, and ``--seed`` customize the run. The older
# commented training draft below this overview is inactive.

# import warnings
# warnings.filterwarnings("ignore")

# import json
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from pathlib import Path

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     average_precision_score,
#     confusion_matrix,
#     classification_report
# )

# # Optional advanced algorithms
# try:
#     from xgboost import XGBClassifier
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False

# try:
#     from catboost import CatBoostClassifier
#     CATBOOST_AVAILABLE = True
# except ImportError:
#     CATBOOST_AVAILABLE = False


# # ==========================================================
# # 1. CONFIGURATION
# # ==========================================================

# DATA_PATH = Path(
#     "dataset/unified_persistence_dataset.csv"
# )

# OUTPUT_DIR = Path("training_results")
# MODEL_DIR = OUTPUT_DIR / "models"
# PLOT_DIR = OUTPUT_DIR / "plots"

# MODEL_DIR.mkdir(parents=True, exist_ok=True)
# PLOT_DIR.mkdir(parents=True, exist_ok=True)

# TARGET = "Persistence_Next_Term"
# GROUP_COLUMN = "Student_ID"

# RANDOM_STATE = 42
# TEST_SIZE = 0.20


# # ==========================================================
# # 2. DEFINE INPUT FEATURES
# # ==========================================================

# FEATURES = [
#     "LMS_Login_Count",
#     "Weekly_Active_Hours",
#     "Assignment_Submission_Rate",
#     "Lecture_Attendance_Rate",

#     "Current_GPA",
#     "Failed_Subject_Count",

#     "Tuition_Fee",
#     "Outstanding_Fee",
#     "Scholarship_Amount",

#     "Feedback_Sentiment_Score",
#     "Academic_Stress_Survey_Score",

#     "Support_Request_Count",
#     "Counseling_Session_Count",

#     "Engagement_Risk_Score",
#     "Withdrawal_Hazard_Score"
# ]


# # ==========================================================
# # 3. LOAD DATASET
# # ==========================================================

# print("\n" + "=" * 65)
# print("UNIFIED PERSISTENCE CORRELATOR")
# print("MACHINE LEARNING TRAINING PIPELINE")
# print("=" * 65)

# if not DATA_PATH.exists():
#     raise FileNotFoundError(
#         f"Dataset not found: {DATA_PATH}"
#     )

# df = pd.read_csv(DATA_PATH)

# print("\nDataset loaded successfully")
# print("Shape:", df.shape)

# required_columns = (
#     FEATURES + [TARGET, GROUP_COLUMN]
# )

# missing_columns = [
#     col for col in required_columns
#     if col not in df.columns
# ]

# if missing_columns:
#     raise ValueError(
#         f"Missing columns: {missing_columns}"
#     )

# if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
#     raise ValueError(
#         "Target must contain only binary values 0 and 1."
#     )

# if df[TARGET].isna().any():
#     raise ValueError("Target contains missing values.")

# if df[GROUP_COLUMN].isna().any():
#     raise ValueError("Student_ID contains missing values.")

# if df[FEATURES].select_dtypes(
#     exclude=np.number
# ).shape[1] > 0:
#     raise ValueError("All model input features must be numeric.")

# if not df[FEATURES].apply(
#     lambda col: np.isfinite(col.dropna()).all()
# ).all():
#     raise ValueError("Input features contain infinite values.")


# # ==========================================================
# # 4. PREPARE INPUTS AND TARGET
# # ==========================================================

# X = df[FEATURES].copy()
# y = df[TARGET].astype(int).copy()
# groups = df[GROUP_COLUMN].copy()

# print("\nInput features:", len(FEATURES))
# print("Target:", TARGET)
# print("Unique students:", groups.nunique())

# print("\nTarget distribution:")
# print(y.value_counts(normalize=True).round(4))


# # ==========================================================
# # 5. GROUP-BASED TRAIN / TEST SPLIT
# # ==========================================================

# # All observations from one student remain in
# # either training or testing, never both.

# splitter = GroupShuffleSplit(
#     n_splits=1,
#     test_size=TEST_SIZE,
#     random_state=RANDOM_STATE
# )

# train_idx, test_idx = next(
#     splitter.split(X, y, groups=groups)
# )

# X_train = X.iloc[train_idx]
# X_test = X.iloc[test_idx]

# y_train = y.iloc[train_idx]
# y_test = y.iloc[test_idx]

# train_groups = set(groups.iloc[train_idx])
# test_groups = set(groups.iloc[test_idx])

# assert train_groups.isdisjoint(test_groups)

# print("\nTraining records:", len(X_train))
# print("Testing records:", len(X_test))
# print("Training students:", len(train_groups))
# print("Testing students:", len(test_groups))


# # ==========================================================
# # 6. DEFINE MODELS
# # ==========================================================

# models = {}

# # Logistic Regression
# models["Logistic Regression"] = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
#     ("classifier", LogisticRegression(
#         max_iter=2000,
#         class_weight="balanced",
#         random_state=RANDOM_STATE
#     ))
# ])


# # Random Forest
# models["Random Forest"] = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("classifier", RandomForestClassifier(
#         n_estimators=400,
#         max_depth=12,
#         min_samples_split=10,
#         min_samples_leaf=4,
#         max_features="sqrt",
#         class_weight="balanced",
#         n_jobs=-1,
#         random_state=RANDOM_STATE
#     ))
# ])


# # XGBoost
# if XGBOOST_AVAILABLE:

#     models["XGBoost"] = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("classifier", XGBClassifier(
#             n_estimators=400,
#             max_depth=4,
#             learning_rate=0.04,
#             subsample=0.85,
#             colsample_bytree=0.85,
#             reg_lambda=3,
#             eval_metric="logloss",
#             n_jobs=-1,
#             random_state=RANDOM_STATE
#         ))
#     ])


# # CatBoost
# if CATBOOST_AVAILABLE:

#     models["CatBoost"] = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("classifier", CatBoostClassifier(
#             iterations=600,
#             depth=5,
#             learning_rate=0.04,
#             loss_function="Logloss",
#             verbose=False,
#             random_seed=RANDOM_STATE,
#             l2_leaf_reg=5,
#             allow_writing_files=False
#         ))
#     ])


# print("\nModels selected:")

# for name in models:
#     print("-", name)


# # ==========================================================
# # 7. TRAIN AND EVALUATE
# # ==========================================================

# results = []
# trained_models = {}
# predictions = {}

# for name, model in models.items():

#     print("\n" + "-" * 60)
#     print("Training:", name)
#     print("-" * 60)

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     accuracy = accuracy_score(y_test, y_pred)

#     precision = precision_score(
#         y_test, y_pred, zero_division=0
#     )

#     recall = recall_score(
#         y_test, y_pred, zero_division=0
#     )

#     f1 = f1_score(
#         y_test, y_pred, zero_division=0
#     )

#     roc_auc = roc_auc_score(y_test, y_prob)

#     pr_auc = average_precision_score(y_test, y_prob)

#     results.append({
#         "Model": name,
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1_Score": f1,
#         "ROC_AUC": roc_auc,
#         "PR_AUC": pr_auc
#     })

#     trained_models[name] = model

#     predictions[name] = {
#         "predicted": y_pred,
#         "probability": y_prob
#     }

#     print(f"Accuracy  : {accuracy:.4f}")
#     print(f"Precision : {precision:.4f}")
#     print(f"Recall    : {recall:.4f}")
#     print(f"F1 Score  : {f1:.4f}")
#     print(f"ROC-AUC   : {roc_auc:.4f}")
#     print(f"PR-AUC    : {pr_auc:.4f}")


# # ==========================================================
# # 8. MODEL COMPARISON
# # ==========================================================

# comparison = pd.DataFrame(results)

# comparison = comparison.sort_values(
#     by="ROC_AUC",
#     ascending=False
# ).reset_index(drop=True)

# print("\n" + "=" * 65)
# print("MODEL PERFORMANCE COMPARISON")
# print("=" * 65)

# print(comparison.to_string(index=False))

# comparison.to_csv(
#     OUTPUT_DIR / "model_comparison.csv",
#     index=False
# )


# # ==========================================================
# # 9. SELECT BEST MODEL
# # ==========================================================

# # ROC-AUC is used for model ranking.
# # This does not guarantee the best operational threshold.

# best_model_name = comparison.iloc[0]["Model"]

# best_model = trained_models[best_model_name]

# best_pred = predictions[best_model_name]["predicted"]
# best_prob = predictions[best_model_name]["probability"]

# print("\nBest model:", best_model_name)


# # ==========================================================
# # 10. CLASSIFICATION REPORT
# # ==========================================================

# report = classification_report(
#     y_test,
#     best_pred,
#     target_names=[
#         "Discontinued",
#         "Continued"
#     ],
#     zero_division=0
# )

# print("\nClassification Report:")
# print(report)

# with open(
#     OUTPUT_DIR / "classification_report.txt",
#     "w"
# ) as file:
#     file.write(report)


# # ==========================================================
# # 11. CONFUSION MATRIX
# # ==========================================================

# cm = confusion_matrix(y_test, best_pred)

# plt.figure(figsize=(7, 5))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Discontinued", "Continued"],
#     yticklabels=["Discontinued", "Continued"]
# )

# plt.title(f"Confusion Matrix - {best_model_name}")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()

# plt.savefig(
#     PLOT_DIR / "confusion_matrix.png",
#     dpi=300
# )

# plt.close()


# # ==========================================================
# # 12. FEATURE IMPORTANCE
# # ==========================================================

# classifier = best_model.named_steps["classifier"]

# if hasattr(classifier, "feature_importances_"):

#     importance_values = classifier.feature_importances_

# elif hasattr(classifier, "coef_"):

#     importance_values = np.abs(
#         classifier.coef_[0]
#     )

# else:
#     importance_values = None


# if importance_values is not None:

#     importance_df = pd.DataFrame({
#         "Feature": FEATURES,
#         "Importance": importance_values
#     }).sort_values(
#         "Importance",
#         ascending=False
#     )

#     importance_df.to_csv(
#         OUTPUT_DIR / "feature_importance.csv",
#         index=False
#     )

#     plt.figure(figsize=(10, 7))

#     sns.barplot(
#         data=importance_df,
#         x="Importance",
#         y="Feature"
#     )

#     plt.title(
#         f"Feature Importance - {best_model_name}"
#     )

#     plt.tight_layout()

#     plt.savefig(
#         PLOT_DIR / "feature_importance.png",
#         dpi=300
#     )

#     plt.close()


# # ==========================================================
# # 13. SAVE BEST MODEL
# # ==========================================================

# model_artifact = {
#     "model": best_model,
#     "features": FEATURES,
#     "target": TARGET,
#     "model_name": best_model_name,
#     "random_state": RANDOM_STATE
# }

# model_path = MODEL_DIR / "best_persistence_model.pkl"

# joblib.dump(
#     model_artifact,
#     model_path
# )

# print("\nBest model saved:", model_path)


# # ==========================================================
# # 14. SAVE TEST PREDICTIONS
# # ==========================================================

# prediction_output = df.iloc[test_idx][
#     [GROUP_COLUMN, "Observation_Date"]
# ].copy()

# prediction_output["Actual_Persistence"] = y_test.values

# prediction_output["Predicted_Persistence"] = best_pred

# prediction_output["Persistence_Probability"] = best_prob

# prediction_output["Persistence_Score"] = (
#     best_prob * 100
# ).round(2)

# prediction_output["Predicted_Risk"] = (
#     1 - best_prob
# ).round(4)

# prediction_output.to_csv(
#     OUTPUT_DIR / "test_predictions.csv",
#     index=False
# )


# # ==========================================================
# # 15. SAVE TRAINING SUMMARY
# # ==========================================================

# summary = {
#     "best_model": best_model_name,
#     "training_records": int(len(X_train)),
#     "testing_records": int(len(X_test)),
#     "training_students": int(len(train_groups)),
#     "testing_students": int(len(test_groups)),
#     "number_of_features": len(FEATURES),
#     "metrics": comparison.iloc[0].drop(
#         "Model"
#     ).to_dict()
# }

# with open(
#     OUTPUT_DIR / "training_summary.json",
#     "w"
# ) as file:
#     json.dump(summary, file, indent=4)


# # ==========================================================
# # 16. FINAL OUTPUT
# # ==========================================================

# print("\n" + "=" * 65)
# print("TRAINING COMPLETED SUCCESSFULLY")
# print("=" * 65)

# print("Selected model:", best_model_name)

# print(
#     "Test accuracy:",
#     f"{accuracy_score(y_test, best_pred):.4f}"
# )

# print(
#     "Test ROC-AUC:",
#     f"{roc_auc_score(y_test, best_prob):.4f}"
# )

# print("\nGenerated files:")

# print("1. Model:", model_path)
# print("2. Comparison:", OUTPUT_DIR / "model_comparison.csv")
# print("3. Predictions:", OUTPUT_DIR / "test_predictions.csv")
# print("4. Feature importance:", OUTPUT_DIR / "feature_importance.csv")
# print("5. Confusion matrix:", PLOT_DIR / "confusion_matrix.png")
# print("6. Training summary:", OUTPUT_DIR / "training_summary.json")

# print("\nAll operations completed.")

"""Train the ``drop`` workflow's persistence estimator from synthetic data.

The script loads and validates the student-level identifier, persistence
target, component targets, and predictor columns. It partitions students into
training and validation groups so observations from one student do not cross
the split, fits standardized linear regressors for engagement, affect, and
hazard risk, and trains a logistic persistence classifier on those component
scores. Validation accuracy, F1, and log loss are printed.

Usage from this directory::

    python train.py
    python train.py --data dataset/unified_persistence_dataset.csv

The default output is ``models/unified_persistence_model.pkl``. The pickle
stores feature definitions, fitted component weights, scaling values, and
classifier weights for use by ``api.py``. The input is generated by
``generate_synthetic_data.py``; these synthetic-data results are not a
substitute for evaluation on real student records.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = {
    "engagement": [
        "LMS_Login_Count",
        "Weekly_Active_Hours",
        "Assignment_Submission_Rate",
        "Lecture_Attendance_Rate",
    ],
    "affect": [
        "Feedback_Sentiment_Score",
        "Academic_Stress_Survey_Score",
        "Support_Request_Count",
        "Counseling_Session_Count",
    ],
    "hazard": [
        "Current_GPA",
        "Failed_Subject_Count",
        "Outstanding_Fee",
        "Tuition_Fee",
        "Scholarship_Amount",
    ],
}
COMPONENT_TARGETS = {
    "engagement": "Engagement_Risk_Score",
    "affect": "Affect_Risk_Score",
    "hazard": "Withdrawal_Hazard_Score",
}
TARGET = "Persistence_Next_Term"


# ---------------------------------------------------------------------------
# Numerical and model-fitting helpers
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities with numerically bounded inputs.

    Parameters
    ----------
    z : numpy.ndarray
        Unnormalized binary-classification scores.

    Returns
    -------
    numpy.ndarray
        Elementwise probabilities between zero and one.
    """
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Fit ridge regression with an unpenalized intercept.

    Parameters
    ----------
    x : numpy.ndarray
        Two-dimensional matrix of standardized input features.
    y : numpy.ndarray
        Regression targets corresponding to the rows of ``x``.
    ridge : float, default=1e-3
        L2 penalty applied to feature weights, but not the intercept.

    Returns
    -------
    numpy.ndarray
        Coefficients with the intercept in the first position.
    """
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def fit_logistic(x: np.ndarray, y: np.ndarray, learning_rate: float = 0.1,
                 epochs: int = 2500, l2: float = 1e-3) -> np.ndarray:
    """Fit binary logistic regression using batch gradient descent.

    Parameters
    ----------
    x : numpy.ndarray
        Two-dimensional input-feature matrix.
    y : numpy.ndarray
        Binary target values aligned with the rows of ``x``.
    learning_rate : float, default=0.1
        Step size used for each gradient update.
    epochs : int, default=2500
        Number of full-data gradient updates.
    l2 : float, default=1e-3
        L2 penalty applied to feature weights, but not the intercept.

    Returns
    -------
    numpy.ndarray
        Fitted coefficients with the intercept in the first position.
    """
    design = np.column_stack((np.ones(len(x)), x))
    weights = np.zeros(design.shape[1])
    for _ in range(epochs):
        error = sigmoid(design @ weights) - y
        gradient = design.T @ error / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate thresholded classification metrics and probability loss.

    Predictions use a fixed 0.5 probability threshold. Probabilities are
    clipped only for the logarithms in log loss.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels.
    probabilities : numpy.ndarray
        Predicted positive-class probabilities aligned with ``y_true``.

    Returns
    -------
    dict[str, float]
        Accuracy, F1 score, and binary log loss.
    """
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float(np.mean(predictions == y_true))
    denominator = np.sum(predictions == 1) + np.sum(y_true == 1)
    f1 = float(2 * np.sum((predictions == 1) & (y_true == 1)) / denominator) if denominator else 0.0
    clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
    log_loss = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))
    return {"accuracy": accuracy, "f1": f1, "log_loss": log_loss}


# ---------------------------------------------------------------------------
# Training command and artifact generation
# ---------------------------------------------------------------------------

def main() -> None:
    """Train the component regressors and persistence classifier.

    Parses the dataset and output paths, partitions whole students into
    training and validation sets, evaluates the classifier, and serializes
    the model artifact consumed by the inference API.

    Raises
    ------
    SystemExit
        If command-line arguments are invalid or the dataset file is missing.
    ValueError
        If required columns are absent or there are too few distinct students
        for a non-empty training and validation split.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/unified_persistence_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/unified_persistence_model.pkl"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")

    frame = pd.read_csv(args.data)
    if "Affect_Risk_Score" not in frame.columns:
        if "Affect_Interaction_Score" in frame.columns:
            frame["Affect_Risk_Score"] = 100 - frame["Affect_Interaction_Score"]
        elif {"Feedback_Sentiment_Score", "Academic_Stress_Survey_Score"}.issubset(frame.columns):
            # The unified generator keeps these source signals but omits the
            # intermediate affect-risk label. Reconstruct its documented proxy.
            satisfaction_proxy = np.clip(50 + 50 * frame["Feedback_Sentiment_Score"], 0, 100)
            frame["Affect_Risk_Score"] = (
                0.6 * (100 - satisfaction_proxy)
                + 0.4 * frame["Academic_Stress_Survey_Score"]
            ).clip(0, 100)
        else:
            raise ValueError("Dataset needs Affect_Risk_Score or Affect_Interaction_Score")
    required = {TARGET, "Student_ID", *COMPONENT_TARGETS.values()}
    required.update(column for columns in FEATURES.values() for column in columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    columns = sorted(required - {"Student_ID"})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=columns).copy()
    frame[TARGET] = frame[TARGET].astype(int)
    students = frame["Student_ID"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(students)
    split_at = max(1, min(len(students) - 1, int(len(students) * (1 - args.validation_fraction))))
    train_ids, valid_ids = set(students[:split_at]), set(students[split_at:])
    train = frame[frame["Student_ID"].isin(train_ids)]
    valid = frame[frame["Student_ID"].isin(valid_ids)]
    if train.empty or valid.empty:
        raise ValueError("Not enough distinct students for the requested split")

    # Fit component regressors on training observations and reuse their
    # training statistics when transforming validation observations.
    component_models: dict[str, dict[str, object]] = {}

    def component_matrix(part: pd.DataFrame, fit: bool) -> np.ndarray:
        """Transform rows into the ordered component-risk feature matrix.

        Parameters
        ----------
        part : pandas.DataFrame
            Dataset partition to transform.
        fit : bool
            Whether to estimate and store component scalers and regressors
            from the training partition.

        Returns
        -------
        numpy.ndarray
            One clipped risk-score column per configured component, in
            ``FEATURES`` order.
        """
        outputs = []
        for name, feature_names in FEATURES.items():
            raw = part[feature_names].to_numpy(dtype=float)
            if fit:
                mean = raw.mean(axis=0)
                scale = raw.std(axis=0)
                scale[scale == 0] = 1
                standardized = (raw - mean) / scale
                weights = fit_linear(standardized, train[COMPONENT_TARGETS[name]].to_numpy(dtype=float))
                component_models[name] = {"features": feature_names, "mean": mean, "scale": scale, "weights": weights}
            model = component_models[name]
            standardized = (raw - model["mean"]) / model["scale"]
            risk = np.column_stack((np.ones(len(raw)), standardized)) @ model["weights"]
            outputs.append(np.clip(risk, 0, 100))
        return np.column_stack(outputs)

    # Train the final classifier on component risks; validation rows are
    # transformed using training-only component and classifier scalers.
    train_components = component_matrix(train, fit=True)
    valid_components = component_matrix(valid, fit=False)
    component_mean = train_components.mean(axis=0)
    component_scale = train_components.std(axis=0)
    component_scale[component_scale == 0] = 1
    train_x = (train_components - component_mean) / component_scale
    valid_x = (valid_components - component_mean) / component_scale
    classifier = fit_logistic(train_x, train[TARGET].to_numpy(dtype=float))
    probabilities = sigmoid(np.column_stack((np.ones(len(valid_x)), valid_x)) @ classifier)
    scores = metrics(valid[TARGET].to_numpy(dtype=int), probabilities)

    # Persist all transforms with the weights so inference uses the same
    # feature ordering and standardization as training.
    artifact = {
        "version": 1,
        "components": component_models,
        "component_order": list(FEATURES),
        "component_mean": component_mean,
        "component_scale": component_scale,
        "classifier_weights": classifier,
        "score_scale": "0-100; higher means greater likelihood of continued enrollment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    print(f"Rows: {len(frame):,} ({len(train):,} train, {len(valid):,} validation)")
    print(f"Students: {len(train_ids):,} train, {len(valid_ids):,} validation")
    print(f"Validation accuracy: {scores['accuracy']:.3f}")
    print(f"Validation F1:       {scores['f1']:.3f}")
    print(f"Validation log loss: {scores['log_loss']:.3f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
