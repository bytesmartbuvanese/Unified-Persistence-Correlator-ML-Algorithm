"""Compare candidate persistence classifiers on synthetic student data.

This command-line utility reads the dataset used by the ``drop`` workflow,
keeps each student's latest observation when ``Observation_Date`` is
available, validates the configured feature and target columns, and splits
students—not individual rows—into training and validation groups. It
standardizes features using training-set statistics, fits logistic regression,
Gaussian Naive Bayes, 15-neighbor KNN, a shallow decision tree, and a
25-tree randomized forest, then ranks them by validation balanced accuracy
and log loss. Training/validation gaps and a majority-class baseline are
printed as basic model-fit diagnostics; this script does not save a selected
model.

Run from the ``drop`` directory with ``python compare.py`` or pass an input
path with ``python compare.py --data dataset/unified_persistence_dataset.csv``.
The default data file is created by ``generate_synthetic_data.py``. Feature
groups and the target are imported from ``train.py``, so the files must remain
available together. Results are evaluation output only and do not establish
performance on real student populations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train import FEATURES, TARGET


# ---------------------------------------------------------------------------
# Probability and preprocessing helpers
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities after bounding extreme values.

    Parameters
    ----------
    z : numpy.ndarray
        Unnormalized binary-classification scores.

    Returns
    -------
    numpy.ndarray
        Elementwise probabilities between zero and one.
    """
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features using statistics calculated from training data.

    Constant training columns use a scale of one to avoid division by zero.

    Parameters
    ----------
    train_x : numpy.ndarray
        Training feature matrix used to estimate means and scales.
    other_x : numpy.ndarray
        Feature matrix transformed with the training statistics.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Standardized training and other feature matrices.
    """
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1
    return (train_x - mean) / scale, (other_x - mean) / scale


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 0.01) -> np.ndarray:
    """Fit binary logistic regression with batch gradient descent.

    Parameters
    ----------
    x : numpy.ndarray
        Standardized input-feature matrix.
    y : numpy.ndarray
        Binary labels aligned with the rows of ``x``.
    l2 : float, default=0.01
        L2 regularization strength applied to feature weights, not the
        intercept.

    Returns
    -------
    numpy.ndarray
        Fitted coefficients with the intercept first.
    """
    design = np.column_stack((np.ones(len(x)), x))
    w = np.zeros(design.shape[1])
    for _ in range(1800):
        grad = design.T @ (sigmoid(design @ w) - y) / len(y)
        grad[1:] += l2 * w[1:]
        w -= 0.08 * grad
    return w


def fit_gaussian_nb(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate class priors and per-feature Gaussian parameters.

    A small variance offset limits numerical problems for constant features.

    Parameters
    ----------
    x : numpy.ndarray
        Training feature matrix.
    y : numpy.ndarray
        Binary labels aligned with the rows of ``x``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Log class priors, class-conditional means, and class-conditional
        variances, each ordered by class 0 then class 1.
    """
    priors, means, variances = [], [], []
    for cls in (0, 1):
        subset = x[y == cls]
        priors.append(max(len(subset), 1) / len(y))
        means.append(subset.mean(axis=0))
        variances.append(subset.var(axis=0) + 1e-6)
    return np.log(priors), np.asarray(means), np.asarray(variances)


def predict_gaussian_nb(model, x: np.ndarray) -> np.ndarray:
    """Predict positive-class probabilities from Gaussian Naive Bayes.

    Parameters
    ----------
    model : tuple
        Log priors, means, and variances returned by
        :func:`fit_gaussian_nb`.
    x : numpy.ndarray
        Feature matrix in the same representation used to fit ``model``.

    Returns
    -------
    numpy.ndarray
        Positive-class probabilities, one per input row.
    """
    log_prior, means, variances = model
    log_prob = []
    for cls in (0, 1):
        lp = log_prior[cls] - 0.5 * np.sum(np.log(2 * np.pi * variances[cls]))
        lp -= 0.5 * np.sum((x - means[cls]) ** 2 / variances[cls], axis=1)
        log_prob.append(lp)
    return sigmoid(log_prob[1] - log_prob[0])


def predict_knn(train_x: np.ndarray, train_y: np.ndarray, x: np.ndarray, k: int = 15) -> np.ndarray:
    """Estimate positive-class probabilities by averaging nearest labels.

    Distances are squared Euclidean distances, computed in blocks to bound
    temporary memory use. The neighborhood size is capped by the training
    sample count.

    Parameters
    ----------
    train_x : numpy.ndarray
        Training feature matrix.
    train_y : numpy.ndarray
        Binary training labels aligned with ``train_x``.
    x : numpy.ndarray
        Feature matrix to score.
    k : int, default=15
        Number of nearest training rows used for each estimate.

    Returns
    -------
    numpy.ndarray
        Mean neighbor label for each row in ``x``.
    """
    k = min(k, len(train_x))
    result = np.empty(len(x))
    for start in range(0, len(x), 256):
        block = x[start:start + 256]
        distances = ((block[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        result[start:start + len(block)] = train_y[nearest].mean(axis=1)
    return result


# ---------------------------------------------------------------------------
# Decision-tree and ensemble models
# ---------------------------------------------------------------------------

def gini(y: np.ndarray) -> float:
    """Compute binary Gini impurity for a set of labels.

    Parameters
    ----------
    y : numpy.ndarray
        Binary labels for a node.

    Returns
    -------
    float
        Gini impurity, or zero for an empty label array.
    """
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 2 * p * (1 - p)


def build_tree(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
               depth: int, max_depth: int, feature_count: int) -> dict:
    """Build a randomized, depth-limited binary classification tree.

    Candidate split points are selected from quantiles of a random subset of
    features. Nodes stop splitting when they are pure, too small, at maximum
    depth, or have no positive impurity reduction.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix for the current node's samples.
    y : numpy.ndarray
        Binary labels aligned with ``x``.
    rng : numpy.random.Generator
        Random generator used to select candidate features.
    depth : int
        Current depth, with the root at zero.
    max_depth : int
        Maximum permitted tree depth.
    feature_count : int
        Number of candidate features considered at each split.

    Returns
    -------
    dict
        Nested node representation containing a leaf probability and, for
        split nodes, a feature, threshold, and left/right children.
    """
    probability = float(y.mean()) if len(y) else 0.5
    node = {"probability": probability}
    if depth >= max_depth or len(y) < 20 or probability in (0.0, 1.0):
        return node
    features = rng.choice(x.shape[1], min(feature_count, x.shape[1]), replace=False)
    best = None
    parent_impurity = gini(y)
    for feature in features:
        values = x[:, feature]
        thresholds = np.unique(np.quantile(values, [0.2, 0.35, 0.5, 0.65, 0.8]))
        for threshold in thresholds:
            left = values <= threshold
            n_left, n_right = int(left.sum()), int((~left).sum())
            if min(n_left, n_right) < 10:
                continue
            gain = parent_impurity - (n_left * gini(y[left]) + n_right * gini(y[~left])) / len(y)
            if best is None or gain > best[0]:
                best = (gain, int(feature), float(threshold), left)
    if best is None or best[0] <= 1e-8:
        return node
    _, feature, threshold, left = best
    node.update({"feature": feature, "threshold": threshold,
                 "left": build_tree(x[left], y[left], rng, depth + 1, max_depth, feature_count),
                 "right": build_tree(x[~left], y[~left], rng, depth + 1, max_depth, feature_count)})
    return node


def predict_tree(tree: dict, x: np.ndarray) -> np.ndarray:
    """Predict positive-class probabilities by traversing a tree.

    Parameters
    ----------
    tree : dict
        Root node produced by :func:`build_tree`.
    x : numpy.ndarray
        Feature matrix whose columns use the tree's feature indices.

    Returns
    -------
    numpy.ndarray
        Leaf probabilities, one per input row.
    """
    result = np.empty(len(x))
    def visit(node: dict, indices: np.ndarray) -> None:
        if "feature" not in node:
            result[indices] = node["probability"]
            return
        left = x[indices, node["feature"]] <= node["threshold"]
        visit(node["left"], indices[left])
        visit(node["right"], indices[~left])
    visit(tree, np.arange(len(x)))
    return result


def fit_forest(x: np.ndarray, y: np.ndarray, seed: int, trees: int = 25) -> list[dict]:
    """Fit a bagged forest of randomized depth-limited trees.

    Each tree receives a bootstrap sample and an independently seeded feature
    sampler derived from ``seed``.

    Parameters
    ----------
    x : numpy.ndarray
        Training feature matrix.
    y : numpy.ndarray
        Binary labels aligned with ``x``.
    seed : int
        Seed used to make bootstrap and tree sampling reproducible.
    trees : int, default=25
        Number of trees to fit.

    Returns
    -------
    list[dict]
        Fitted tree roots.
    """
    rng = np.random.default_rng(seed)
    forest = []
    for _ in range(trees):
        indices = rng.integers(0, len(y), size=len(y))
        tree_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        forest.append(build_tree(x[indices], y[indices], tree_rng, 0, 6,
                                 max(1, int(np.sqrt(x.shape[1])))))
    return forest


def predict_forest(forest: list[dict], x: np.ndarray) -> np.ndarray:
    """Average probability estimates across a fitted forest.

    Parameters
    ----------
    forest : list[dict]
        Trees returned by :func:`fit_forest`.
    x : numpy.ndarray
        Feature matrix to score.

    Returns
    -------
    numpy.ndarray
        Mean positive-class probability across trees for each row.
    """
    return np.mean([predict_tree(tree, x) for tree in forest], axis=0)


# ---------------------------------------------------------------------------
# Evaluation and command-line reporting
# ---------------------------------------------------------------------------

def score(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate classification metrics at a 0.5 probability threshold.

    Parameters
    ----------
    y : numpy.ndarray
        Binary ground-truth labels.
    probabilities : numpy.ndarray
        Predicted positive-class probabilities aligned with ``y``.

    Returns
    -------
    dict[str, float]
        Accuracy, balanced accuracy, F1 score, and binary log loss.
    """
    pred = (probabilities >= 0.5).astype(int)
    accuracy = np.mean(pred == y)
    recall_pos = np.sum((pred == 1) & (y == 1)) / max(np.sum(y == 1), 1)
    recall_neg = np.sum((pred == 0) & (y == 0)) / max(np.sum(y == 0), 1)
    precision = np.sum((pred == 1) & (y == 1)) / max(np.sum(pred == 1), 1)
    f1 = 2 * precision * recall_pos / max(precision + recall_pos, 1e-12)
    p = np.clip(probabilities, 1e-8, 1 - 1e-8)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"accuracy": float(accuracy), "balanced_accuracy": float((recall_pos + recall_neg) / 2),
            "f1": float(f1), "log_loss": float(loss)}


def diagnosis(train: dict[str, float], valid: dict[str, float]) -> tuple[float, str]:
    """Apply simple heuristic fit checks to training and validation scores.

    The returned label is a diagnostic cue, not a statistical test or a
    substitute for evaluation across repeated splits.

    Parameters
    ----------
    train : dict[str, float]
        Training metrics produced by :func:`score`.
    valid : dict[str, float]
        Validation metrics produced by :func:`score`.

    Returns
    -------
    tuple[float, str]
        Training-minus-validation accuracy gap and a brief fit warning.
    """
    gap = train["accuracy"] - valid["accuracy"]
    if valid["balanced_accuracy"] < 0.55:
        label = "weak class discrimination; possible underfitting"
    elif gap > 0.08:
        label = "likely overfitting"
    elif gap < -0.08:
        label = "validation unusually higher; check split variance"
    elif train["accuracy"] < 0.62 and valid["accuracy"] < 0.62:
        label = "possible underfitting"
    else:
        label = "no strong fit warning"
    return gap, label


def main() -> None:
    """Compare classifiers using a student-level train/validation split.

    Reads the configured dataset, retains each student's latest observation
    when dates are available, trains the candidate models, and prints ranked
    validation metrics with heuristic fit checks.

    Raises
    ------
    SystemExit
        If command-line arguments are invalid.
    ValueError
        If the dataset is missing required columns or cannot yield non-empty
        student-level training and validation partitions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/unified_persistence_dataset.csv"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    frame = pd.read_csv(args.data)
    # Use each student's latest observation. This matches a one-score-per-
    # student use case and avoids giving students with more rows extra weight.
    if "Observation_Date" in frame.columns:
        frame["Observation_Date"] = pd.to_datetime(frame["Observation_Date"], errors="coerce")
        frame = frame.sort_values("Observation_Date")
    frame = frame.drop_duplicates("Student_ID", keep="last")
    feature_names = [c for group in FEATURES.values() for c in group]
    required = feature_names + ["Student_ID", TARGET]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    y_all = frame[TARGET].to_numpy(dtype=int)
    student_ids = frame["Student_ID"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(student_ids)
    split = max(1, min(len(student_ids) - 1,
                       int(len(student_ids) * (1 - args.validation_fraction))))
    train_ids, valid_ids = set(student_ids[:split]), set(student_ids[split:])
    is_train = frame["Student_ID"].isin(train_ids).to_numpy()
    x_all = frame[feature_names].to_numpy(dtype=float)
    x_train, x_valid = standardize(x_all[is_train], x_all[~is_train])
    y_train, y_valid = y_all[is_train], y_all[~is_train]
    if len(y_train) == 0 or len(y_valid) == 0:
        raise ValueError("Not enough distinct students for this split")

    logistic = fit_logistic(x_train, y_train)
    nb = fit_gaussian_nb(x_train, y_train)
    shallow_tree = build_tree(x_train, y_train, np.random.default_rng(args.seed), 0, 3,
                              x_train.shape[1])
    forest = fit_forest(x_train, y_train, args.seed)
    models = {
        "Logistic regression": (sigmoid(np.column_stack((np.ones(len(x_train)), x_train)) @ logistic),
                               sigmoid(np.column_stack((np.ones(len(x_valid)), x_valid)) @ logistic)),
        "Gaussian Naive Bayes": (predict_gaussian_nb(nb, x_train), predict_gaussian_nb(nb, x_valid)),
        "15-neighbor KNN": (predict_knn(x_train, y_train, x_train, 15),
                            predict_knn(x_train, y_train, x_valid, 15)),
        "Shallow decision tree": (predict_tree(shallow_tree, x_train), predict_tree(shallow_tree, x_valid)),
        "Random forest (25 trees)": (predict_forest(forest, x_train), predict_forest(forest, x_valid)),
    }
    results = []
    for name, (train_probs, valid_probs) in models.items():
        train_scores, valid_scores = score(y_train, train_probs), score(y_valid, valid_probs)
        gap, fit_note = diagnosis(train_scores, valid_scores)
        results.append((name, valid_scores, train_scores, gap, fit_note))

    results.sort(key=lambda row: (row[1]["balanced_accuracy"], -row[1]["log_loss"]), reverse=True)
    print(f"Rows: {len(frame):,} | train: {len(y_train):,} | validation: {len(y_valid):,}")
    print(f"Students: {len(train_ids):,} train | {len(valid_ids):,} validation")
    print(f"Persistence class balance: train={y_train.mean():.1%}, validation={y_valid.mean():.1%} positive")
    print("Features: " + ", ".join(feature_names))
    print("\nRanked by validation balanced accuracy (then log loss):")
    print(f"{'Model':<27} {'Val Acc':>8} {'Train Acc':>10} {'Gap':>7} {'Val Bal Acc':>12} {'Val F1':>8} {'Val LogLoss':>12}  Fit check")
    for name, val, train, gap, note in results:
        print(f"{name:<27} {val['accuracy']:>8.3f} {train['accuracy']:>10.3f} {gap:>7.3f} "
              f"{val['balanced_accuracy']:>12.3f} {val['f1']:>8.3f} {val['log_loss']:>12.3f}  {note}")
    majority_accuracy = max(float(np.mean(y_valid == 0)), float(np.mean(y_valid == 1)))
    print(f"\nValidation majority-class baseline accuracy: {majority_accuracy:.3f} (balanced accuracy: 0.500)")
    print("\nFit checks are heuristics based on train/validation accuracy; a low gap alone does not prove generalization.")


if __name__ == "__main__":
    main()
