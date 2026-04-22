from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(model_name: str, params: dict) -> Pipeline:
    if model_name == "linear_svm":
        clf = SVC(
            kernel="linear",
            C=params["C"],
            probability=True,
            random_state=42,
        )
    elif model_name == "rbf_svm":
        clf = SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            probability=True,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )
    return pipe


def get_param_grid() -> dict[str, list[dict]]:
    return {
        "linear_svm": [
            {"C": 0.01},
            {"C": 0.1},
            {"C": 1.0},
            {"C": 10.0},
        ],
        "rbf_svm": [
            {"C": 0.1, "gamma": "scale"},
            {"C": 1.0, "gamma": "scale"},
            {"C": 10.0, "gamma": "scale"},
            {"C": 1.0, "gamma": 0.01},
            {"C": 1.0, "gamma": 0.1},
            {"C": 10.0, "gamma": 0.01},
        ],
    }


def run_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> list[dict]:
    results: list[dict] = []
    grids = get_param_grid()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for model_name, param_list in grids.items():
        for params in param_list:
            fold_scores = []

            for train_idx, valid_idx in kf.split(X):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_valid = X.iloc[valid_idx]
                y_valid = y.iloc[valid_idx]

                model = build_model(model_name, params)
                model.fit(X_train, y_train)
                pred = model.predict(X_valid)
                acc = accuracy_score(y_valid, pred)
                fold_scores.append(acc)

            results.append(
                {
                    "model_name": model_name,
                    "params": params,
                    "cv_scores": fold_scores,
                    "cv_mean_accuracy": float(np.mean(fold_scores)),
                    "cv_std_accuracy": float(np.std(fold_scores)),
                }
            )

    results = sorted(results, key=lambda d: d["cv_mean_accuracy"], reverse=True)
    return results


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, pred)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": cm.tolist(),
        "predictions": pred.tolist(),
        "probabilities": proba.tolist(),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pickle(obj, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
