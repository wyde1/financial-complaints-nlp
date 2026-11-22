"""Leakage-free TF-IDF modeling pipelines, validation benchmarking, test evaluation, and plots."""

from __future__ import annotations

import pickle
import struct
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from complaints_nlp.preprocessing import ComplaintTextCleaner, DatasetStats
from complaints_nlp.taxonomy import SHORT_PRODUCT_LABELS


@dataclass(frozen=True)
class ModelBenchmarkResult:
    """Validation metrics and runtime for one candidate pipeline."""

    name: str
    display_name: str
    val_macro_f1: float
    val_weighted_f1: float
    val_accuracy: float
    fit_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "val_macro_f1": round(self.val_macro_f1, 4),
            "val_weighted_f1": round(self.val_weighted_f1, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "fit_seconds": round(self.fit_seconds, 2),
        }


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "multinomial_nb": "Multinomial NB",
    "complement_nb": "Complement NB",
    "logreg_default": "Logistic Regression",
    "logreg_balanced": "Logistic Regression (Balanced)",
    "linear_svc_default": "Linear SVC",
    "linear_svc_balanced": "Linear SVC (Balanced)",
}


def build_vectorizer(
    max_features: int = 50_000,
    min_df: int = 5,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Construct a TF-IDF vectorizer configured for English financial complaints."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )


def build_candidate_estimators(random_state: int = 42) -> dict[str, Any]:
    """Return fast, reproducible sparse-text classifiers (default + class-balanced)."""
    return {
        "multinomial_nb": MultinomialNB(alpha=0.2),
        "complement_nb": ComplementNB(alpha=0.2),
        "logreg_default": LogisticRegression(
            C=2.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        ),
        "logreg_balanced": LogisticRegression(
            C=2.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        ),
        "linear_svc_default": LinearSVC(
            C=0.5,
            dual="auto",
            max_iter=2000,
            random_state=random_state,
        ),
        "linear_svc_balanced": LinearSVC(
            C=0.5,
            class_weight="balanced",
            dual="auto",
            max_iter=2000,
            random_state=random_state,
        ),
    }


def build_pipeline(
    estimator: Any,
    max_features: int = 50_000,
    min_df: int = 5,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
) -> Pipeline:
    """Wrap text cleaning, TF-IDF vectorization, and estimator in one leak-free Pipeline."""
    return Pipeline(
        steps=[
            ("cleaner", ComplaintTextCleaner()),
            (
                "tfidf",
                build_vectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=ngram_range,
                ),
            ),
            ("clf", estimator),
        ]
    )


def split_train_val_test(
    texts: Sequence[str],
    labels: Sequence[str],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Perform a leak-free stratified 3-way split (Train / Validation / Test)."""
    if not (0.0 < val_size < 1.0 and 0.0 < test_size < 1.0 and (val_size + test_size) < 1.0):
        raise ValueError("val_size and test_size must be positive and sum to less than 1.0.")

    texts_list = list(texts)
    labels_list = list(labels)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts_list,
        labels_list,
        test_size=test_size,
        stratify=labels_list,
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def extract_top_features(pipeline: Pipeline, top_n: int = 6) -> dict[str, list[tuple[str, float]]]:
    """Extract top discriminative TF-IDF n-grams per class from a fitted pipeline."""
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    classes = list(clf.classes_)

    weights: np.ndarray | None = None
    if hasattr(clf, "coef_"):
        weights = np.asarray(clf.coef_)
    elif isinstance(clf, CalibratedClassifierCV) and getattr(clf, "calibrated_classifiers_", None):
        fold_coefs = [
            np.asarray(cal.estimator.coef_)
            for cal in clf.calibrated_classifiers_
            if hasattr(cal.estimator, "coef_")
        ]
        if fold_coefs:
            weights = np.mean(fold_coefs, axis=0)
    elif hasattr(clf, "feature_log_prob_"):
        weights = np.asarray(clf.feature_log_prob_)

    if weights is None:
        return {}

    # Binary classification returns a 1D/single-row coef_; expand cleanly for symmetry.
    if weights.ndim == 2 and weights.shape[0] == 1 and len(classes) == 2:
        weights = np.vstack([-weights[0], weights[0]])

    top_per_class: dict[str, list[tuple[str, float]]] = {}
    for idx, cls_name in enumerate(classes):
        row = weights[idx]
        top_indices = np.argsort(row)[-top_n:][::-1]
        top_per_class[str(cls_name)] = [
            (str(feature_names[i]), round(float(row[i]), 4)) for i in top_indices
        ]
    return top_per_class


def train_and_evaluate(
    texts: Sequence[str],
    labels: Sequence[str],
    dataset_stats: DatasetStats | None = None,
    max_features: int = 50_000,
    min_df: int = 5,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    calibrate_linear_svc: bool = True,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Run leak-free 3-way split, benchmark candidate models on Validation, and test the winner once.

    If a ``LinearSVC`` variant wins on Validation (by Macro-F1) and ``calibrate_linear_svc`` is True,
    ``CalibratedClassifierCV`` is fitted strictly on ``X_train`` via internal cross-validation —
    never using the Validation or Test splits.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        texts=texts,
        labels=labels,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    candidates = build_candidate_estimators(random_state=random_state)
    benchmarks: list[ModelBenchmarkResult] = []
    best_bench: ModelBenchmarkResult | None = None
    best_pipeline: Pipeline | None = None

    for name, estimator in candidates.items():
        pipe = build_pipeline(
            estimator=estimator,
            max_features=max_features,
            min_df=min_df,
        )
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_secs = time.perf_counter() - t0

        val_pred = pipe.predict(X_val)
        scores = _evaluate_predictions(y_val, val_pred)
        bench = ModelBenchmarkResult(
            name=name,
            display_name=MODEL_DISPLAY_NAMES.get(name, name),
            val_macro_f1=scores["macro_f1"],
            val_weighted_f1=scores["weighted_f1"],
            val_accuracy=scores["accuracy"],
            fit_seconds=fit_secs,
        )
        benchmarks.append(bench)

        # Keep only the current winning pipeline in memory to minimize peak RAM usage.
        if best_bench is None or (bench.val_macro_f1, bench.val_weighted_f1) > (
            best_bench.val_macro_f1,
            best_bench.val_weighted_f1,
        ):
            best_bench = bench
            best_pipeline = pipe

    assert best_bench is not None and best_pipeline is not None
    best_name = best_bench.name
    calibrated = False
    calibration_val_metrics: dict[str, float] | None = None

    if calibrate_linear_svc and best_name.startswith("linear_svc"):
        # Fit CalibratedClassifierCV strictly on X_train (internal CV only; never touch val/test).
        _, class_counts = np.unique(y_train, return_counts=True)
        cv_folds = min(3, int(class_counts.min()))
        if cv_folds >= 2:
            base_svc = build_candidate_estimators(random_state=random_state)[best_name]
            cal_estimator = CalibratedClassifierCV(
                estimator=base_svc,
                method="sigmoid",
                cv=cv_folds,
            )
            cal_pipeline = build_pipeline(
                estimator=cal_estimator,
                max_features=max_features,
                min_df=min_df,
            )
            cal_pipeline.fit(X_train, y_train)
            cal_val_pred = cal_pipeline.predict(X_val)
            calibration_val_metrics = {
                k: round(v, 4) for k, v in _evaluate_predictions(y_val, cal_val_pred).items()
            }
            best_pipeline = cal_pipeline
            calibrated = True

    # Single unbiased evaluation on the held-out Test split.
    test_pred = best_pipeline.predict(X_test)
    test_scores = _evaluate_predictions(y_test, test_pred)
    classes = list(best_pipeline.named_steps["clf"].classes_)
    per_class_report = classification_report(
        y_test,
        test_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, test_pred, labels=classes)
    top_features = extract_top_features(best_pipeline, top_n=6)

    report = {
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "dataset_stats": dataset_stats.to_dict() if dataset_stats else None,
        "vectorizer": {
            "ngram_range": [1, 2],
            "max_features": max_features,
            "min_df": min_df,
            "sublinear_tf": True,
            "vocab_size": int(len(best_pipeline.named_steps["tfidf"].vocabulary_)),
        },
        "validation_benchmarks": [b.to_dict() for b in benchmarks],
        "selected_model": {
            "name": best_name,
            "display_name": MODEL_DISPLAY_NAMES.get(best_name, best_name),
            "calibrated_on_train_cv": calibrated,
            "calibrated_val_metrics": calibration_val_metrics,
        },
        "test_metrics": {
            "macro_f1": round(test_scores["macro_f1"], 4),
            "weighted_f1": round(test_scores["weighted_f1"], 4),
            "accuracy": round(test_scores["accuracy"], 4),
        },
        "test_per_class": {
            cls: {
                "precision": round(float(per_class_report[cls]["precision"]), 4),
                "recall": round(float(per_class_report[cls]["recall"]), 4),
                "f1": round(float(per_class_report[cls]["f1-score"]), 4),
                "support": int(per_class_report[cls]["support"]),
            }
            for cls in classes
        },
        "top_features_per_class": top_features,
    }
    plot_payload = {
        "classes": classes,
        "confusion_matrix": cm,
        "benchmarks": benchmarks,
        "top_features": top_features,
        "class_counts": dataset_stats.class_counts if dataset_stats else None,
    }
    return best_pipeline, report, plot_payload


def predict_complaints(
    pipeline: Pipeline,
    texts: Sequence[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Predict financial product categories with calibrated probabilities / confidence scores."""
    text_list = list(texts)
    if not text_list:
        return []

    clf = pipeline.named_steps["clf"]
    classes = [str(c) for c in clf.classes_]
    k = max(1, min(int(top_k), len(classes)))

    if hasattr(pipeline, "predict_proba"):
        probs = np.asarray(pipeline.predict_proba(text_list), dtype=float)
    elif hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(text_list), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        probs = softmax(scores, axis=1)
    else:
        preds = pipeline.predict(text_list)
        probs = np.zeros((len(text_list), len(classes)), dtype=float)
        for i, p in enumerate(preds):
            probs[i, classes.index(str(p))] = 1.0

    outputs: list[dict[str, Any]] = []
    for row_probs in probs:
        top_idx = np.argsort(row_probs)[::-1][:k]
        best_idx = int(top_idx[0])
        outputs.append(
            {
                "predicted_product": classes[best_idx],
                "confidence": round(float(row_probs[best_idx]), 4),
                "top_predictions": [
                    {
                        "product": classes[int(idx)],
                        "probability": round(float(row_probs[int(idx)]), 4),
                    }
                    for idx in top_idx
                ],
            }
        )
    return outputs


def save_model_bundle(pipeline: Pipeline, metadata: dict[str, Any], output_path: Path) -> Path:
    """Persist the fitted scikit-learn Pipeline and evaluation metadata via joblib."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, output_path, compress=3)
    return output_path


def load_model_bundle(model_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load a saved joblib bundle containing the fitted Pipeline and metadata."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'.\n"
            "Train a model locally first:\n"
            "  python -m complaints_nlp.cli train --data-path data/rows.csv\n"
            "Or point to an existing .joblib artifact via --model-path or MODEL_PATH."
        )
    try:
        bundle = joblib.load(model_path)
    except (
        EOFError,
        OSError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        ValueError,
        pickle.UnpicklingError,
        struct.error,
        zlib.error,
    ) as exc:
        raise ValueError(f"Model artifact at '{model_path}' is corrupted or unreadable.") from exc
    if isinstance(bundle, dict) and "pipeline" in bundle:
        return bundle["pipeline"], bundle.get("metadata", {})
    if isinstance(bundle, Pipeline):
        return bundle, {}
    raise ValueError(f"Unrecognized model artifact format in '{model_path}'.")


def save_evaluation_plots(
    plots_dir: Path,
    report: dict[str, Any],
    plot_payload: dict[str, Any],
) -> list[Path]:
    """Generate clean, publication-ready charts in ``plots_dir``."""
    # Lazy import avoids matplotlib/seaborn overhead during `cli predict` and preserves
    # Jupyter's active inline backend when called from notebooks.
    import matplotlib

    if "inline" not in matplotlib.get_backend().lower():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Class distribution plot
    class_counts = plot_payload.get("class_counts")
    if class_counts:
        sorted_items = sorted(class_counts.items(), key=lambda kv: kv[1])
        labels = [SHORT_PRODUCT_LABELS.get(k, k) for k, _ in sorted_items]
        counts = [v for _, v in sorted_items]
        total = sum(counts) or 1

        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=140)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8FAFC")
        bars = ax.barh(labels, counts, color="#0F62FE", alpha=0.9, edgecolor="white")
        for bar, cnt in zip(bars, counts):
            pct = (cnt / total) * 100.0
            ax.text(
                bar.get_width() + max(counts) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt:,} ({pct:.1f}%)",
                va="center",
                fontsize=8.5,
                color="#111827",
            )
        ax.set_xlim(0, max(counts) * 1.24)
        ax.set_xlabel("Unique Deduplicated Complaints", fontsize=10, fontweight="bold")
        ax.set_title(
            f"Canonical CFPB Product Distribution ({total:,} Unique Narratives)",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="x", linestyle="--", alpha=0.35, color="#94A3B8")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        p1 = plots_dir / "class_distribution.png"
        plt.savefig(p1, dpi=140)
        plt.close(fig)
        saved.append(p1)

    # 2. Model comparison plot (Validation split)
    benchmarks: list[ModelBenchmarkResult] = plot_payload["benchmarks"]
    if benchmarks:
        ordered = sorted(benchmarks, key=lambda b: b.val_macro_f1, reverse=True)
        names = [b.display_name for b in ordered]
        macro_vals = [b.val_macro_f1 for b in ordered]
        weighted_vals = [b.val_weighted_f1 for b in ordered]
        acc_vals = [b.val_accuracy for b in ordered]

        x = np.arange(len(names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=140)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8FAFC")

        for offset, vals, label, color in [
            (-width, macro_vals, "Macro-F1 (selection)", "#0F62FE"),
            (0.0, weighted_vals, "Weighted-F1", "#0E8A5F"),
            (width, acc_vals, "Accuracy", "#64748B"),
        ]:
            bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="white")
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    color="#111827",
                )

        ax.set_ylabel("Validation Score", fontsize=10, fontweight="bold")
        ax.set_title(
            "Validation Benchmark — Leak-Free TF-IDF Pipelines (Text Only)",
            fontsize=11.5,
            fontweight="bold",
            pad=28,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=16, ha="right", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35, color="#94A3B8")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=3,
            fontsize=9,
        )
        plt.tight_layout()
        p2 = plots_dir / "model_comparison.png"
        plt.savefig(p2, dpi=140)
        plt.close(fig)
        saved.append(p2)

    # 3. Row-normalized confusion matrix (Held-Out Test split)
    classes: list[str] = plot_payload["classes"]
    cm: np.ndarray = plot_payload["confusion_matrix"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    short_labels = [SHORT_PRODUCT_LABELS.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(9.5, 7.6), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar_kws={"label": "Recall (Row-Normalized)"},
        ax=ax,
    )
    sel_name = report["selected_model"]["display_name"]
    test_mf1 = report["test_metrics"]["macro_f1"]
    ax.set_xlabel("Predicted Product", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Product", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Held-Out Test Confusion Matrix — {sel_name} (Macro-F1 = {test_mf1:.3f})",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    plt.xticks(rotation=30, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    plt.tight_layout()
    p3 = plots_dir / "confusion_matrix_norm.png"
    plt.savefig(p3, dpi=140)
    plt.close(fig)
    saved.append(p3)

    # 4. Top discriminative TF-IDF n-grams per class
    top_features: dict[str, list[tuple[str, float]]] = plot_payload.get("top_features", {})
    if top_features:
        n_cls = len(top_features)
        cols = 3
        rows = (n_cls + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(13.5, 3.2 * rows), dpi=140)
        fig.patch.set_facecolor("#FFFFFF")
        axes_flat = np.atleast_1d(axes).ravel()

        for idx, (cls_name, terms) in enumerate(sorted(top_features.items())):
            ax = axes_flat[idx]
            ax.set_facecolor("#F8FAFC")
            term_names = [t for t, _ in reversed(terms)]
            term_weights = [w for _, w in reversed(terms)]
            ax.barh(term_names, term_weights, color="#0F62FE", alpha=0.88)
            ax.set_title(SHORT_PRODUCT_LABELS.get(cls_name, cls_name), fontsize=9.5, fontweight="bold")
            ax.tick_params(axis="both", labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

        for idx in range(n_cls, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(
            f"Top Discriminative TF-IDF Features per Class ({sel_name})",
            fontsize=12.5,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        p4 = plots_dir / "top_tfidf_features.png"
        plt.savefig(p4, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(p4)

    return saved
