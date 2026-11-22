"""Synthetic-data unit & integration tests for the NLP pipelines and CLI (no rows.csv required)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from complaints_nlp.cli import run_cli
from complaints_nlp.config import NARRATIVE_COLUMN, PRODUCT_COLUMN, Settings
from complaints_nlp.modeling import (
    build_pipeline,
    extract_top_features,
    load_model_bundle,
    predict_complaints,
    save_evaluation_plots,
    save_model_bundle,
    train_and_evaluate,
)
from complaints_nlp.preprocessing import load_complaints_csv


def _make_synthetic_csv(csv_path: Path, n_per_class: int = 18) -> Path:
    """Create a realistic synthetic multi-class CFPB CSV for fast, offline testing."""
    templates = {
        "Mortgage": [
            "mortgage escrow payment home loan foreclosure refinance property closing statement",
            "servicer misapplied mortgage principal interest escrow tax insurance account",
            "home loan modification forbearance lender mortgage billing error appraisal",
        ],
        "Debt collection": [
            "debt collector harassed calls workplace validation letter collection agency dispute",
            "third party collector threatened lawsuit unpaid medical debt collection notice",
            "debt collection agency failed verify debt written dispute phone calls",
        ],
        "Credit card": [
            "credit card unauthorized fraudulent charge billing statement apr interest dispute",
            "credit card issuer lowered credit limit annual fee rewards dispute chargeback",
            "revolving credit card balance late fee autopay billing cycle statement",
        ],
        "Student loan": [
            "student loan servicer federal repayment income driven forgiveness deferment forbearance",
            "private student loan tuition lender interest capitalization repayment plan",
            "college student loan consolidation autopay billing statement servicer error",
        ],
    }

    rows = []
    for product, base_phrases in templates.items():
        for i in range(n_per_class):
            phrase = base_phrases[i % len(base_phrases)]
            # Use alphabetic identifiers (digits are stripped during cleaning) plus CFPB masks.
            tag = f"{chr(97 + (i // 26) % 26)}{chr(97 + i % 26)}"
            narrative = f"On XX/XX/2018 XXXX reported case{tag} ref{tag}: {phrase}."
            rows.append(
                {
                    NARRATIVE_COLUMN: narrative,
                    PRODUCT_COLUMN: product,
                    "Sub-product": "leakage_col_must_be_ignored",
                    "Issue": "leakage_issue_must_be_ignored",
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_train_evaluate_save_load_predict_and_plots(tmp_path: Path) -> None:
    csv_path = _make_synthetic_csv(tmp_path / "synthetic.csv", n_per_class=20)
    df, stats = load_complaints_csv(csv_path)
    assert stats.final_rows == 80

    pipeline, report, plot_payload = train_and_evaluate(
        texts=df["cleaned_text"].tolist(),
        labels=df["product"].tolist(),
        dataset_stats=stats,
        max_features=2000,
        min_df=1,
        random_state=42,
    )

    assert len(report["validation_benchmarks"]) == 6
    assert report["test_metrics"]["macro_f1"] >= 0.85
    assert report["test_metrics"]["accuracy"] >= 0.85
    assert report["top_features_per_class"]

    model_path = tmp_path / "artifacts" / "model.joblib"
    save_model_bundle(pipeline, metadata=report, output_path=model_path)
    loaded_pipe, loaded_meta = load_model_bundle(model_path)
    assert loaded_meta["selected_model"]["name"] == report["selected_model"]["name"]

    preds = predict_complaints(
        loaded_pipe,
        [
            "On XX/XX/2018 my mortgage escrow servicer charged an incorrect foreclosure fee.",
            "A debt collector keeps calling my office without providing debt validation.",
        ],
        top_k=3,
    )
    assert preds[0]["predicted_product"] == "Mortgage"
    assert preds[1]["predicted_product"] == "Debt collection"
    assert 0.0 < preds[0]["confidence"] <= 1.0
    assert len(preds[0]["top_predictions"]) == 3

    plots_dir = tmp_path / "plots"
    saved_plots = save_evaluation_plots(plots_dir, report, plot_payload)
    assert {p.name for p in saved_plots} == {
        "class_distribution.png",
        "model_comparison.png",
        "confusion_matrix_norm.png",
        "top_tfidf_features.png",
    }
    assert all(p.stat().st_size > 1000 for p in saved_plots)


def test_cli_train_and_predict_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = _make_synthetic_csv(tmp_path / "rows.csv", n_per_class=18)
    model_path = tmp_path / "artifacts" / "clf.joblib"
    metrics_path = tmp_path / "artifacts" / "metrics.json"
    plots_dir = tmp_path / "plots"

    rc_train = run_cli(
        [
            "train",
            "--data-path",
            str(csv_path),
            "--model-out",
            str(model_path),
            "--metrics-out",
            str(metrics_path),
            "--plots-dir",
            str(plots_dir),
            "--max-features",
            "1500",
            "--min-df",
            "1",
        ]
    )
    assert rc_train == 0
    assert model_path.is_file()
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["split_sizes"]["train"] + metrics["split_sizes"]["val"] + metrics["split_sizes"]["test"] == 72

    # Single-text JSON prediction
    capsys.readouterr()
    rc_pred = run_cli(
        [
            "predict",
            "--model-path",
            str(model_path),
            "--text",
            "My federal student loan servicer miscalculated my income driven repayment plan.",
            "--top-k",
            "2",
            "--json",
        ]
    )
    assert rc_pred == 0
    pred_json = json.loads(capsys.readouterr().out)
    assert pred_json["predicted_product"] == "Student loan"
    assert len(pred_json["top_predictions"]) == 2

    # Batch CSV prediction
    batch_in = tmp_path / "batch_in.csv"
    batch_out = tmp_path / "batch_out.csv"
    pd.DataFrame(
        {
            NARRATIVE_COLUMN: [
                "Unauthorized credit card billing statement charge dispute.",
                "Mortgage servicer failed to pay property tax from escrow.",
            ]
        }
    ).to_csv(batch_in, index=False)

    rc_batch = run_cli(
        [
            "predict",
            "--model-path",
            str(model_path),
            "--input-csv",
            str(batch_in),
            "--output-csv",
            str(batch_out),
        ]
    )
    assert rc_batch == 0
    out_df = pd.read_csv(batch_out)
    assert list(out_df["predicted_product"]) == ["Credit card or prepaid card", "Mortgage"]


def test_cli_predict_missing_model_gives_clear_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_model = tmp_path / "missing.joblib"
    rc = run_cli(
        [
            "predict",
            "--model-path",
            str(missing_model),
            "--text",
            "Debt collector calling repeatedly.",
        ]
    )
    assert rc == 1
    stderr = capsys.readouterr().err
    assert "Trained model artifact not found" in stderr
    assert "complaints_nlp.cli train" in stderr


def test_cli_predict_blank_text_rejected_before_loading_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_cli(
        [
            "predict",
            "--model-path",
            str(tmp_path / "does_not_exist.joblib"),
            "--text",
            "XXXX XX/XX/2018 !!!",
        ]
    )
    assert rc == 1
    stderr = capsys.readouterr().err
    assert "empty after removing redaction masks" in stderr
    assert "Trained model artifact not found" not in stderr


def test_cli_predict_bad_csv_column_rejected_before_loading_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_column": ["hello"]}).to_csv(bad_csv, index=False)
    rc = run_cli(
        [
            "predict",
            "--model-path",
            str(tmp_path / "does_not_exist.joblib"),
            "--input-csv",
            str(bad_csv),
        ]
    )
    assert rc == 1
    stderr = capsys.readouterr().err
    assert "not found in" in stderr
    assert "Trained model artifact not found" not in stderr


def test_calibrated_and_uncalibrated_linear_svc_paths(tmp_path: Path) -> None:
    csv_path = _make_synthetic_csv(tmp_path / "svc.csv", n_per_class=10)
    df, _ = load_complaints_csv(csv_path)
    X = df["cleaned_text"].tolist()
    y = df["product"].tolist()

    # 1. Calibrated LinearSVC fitted via internal train-only CV
    cal_pipe = build_pipeline(
        estimator=CalibratedClassifierCV(
            estimator=LinearSVC(C=0.5, dual="auto", max_iter=2000, random_state=42),
            method="sigmoid",
            cv=2,
        ),
        max_features=1000,
        min_df=1,
    )
    cal_pipe.fit(X, y)
    cal_features = extract_top_features(cal_pipe, top_n=4)
    assert len(cal_features) == 4
    assert all(len(terms) == 4 for terms in cal_features.values())
    cal_preds = predict_complaints(cal_pipe, ["My mortgage escrow servicer charged foreclosure fees."], top_k=2)
    assert cal_preds[0]["predicted_product"] == "Mortgage"
    assert 0.0 < cal_preds[0]["confidence"] <= 1.0

    # 2. Uncalibrated LinearSVC exercises decision_function softmax fallback
    raw_pipe = build_pipeline(
        estimator=LinearSVC(C=0.5, dual="auto", max_iter=2000, random_state=42),
        max_features=1000,
        min_df=1,
    )
    raw_pipe.fit(X, y)
    raw_preds = predict_complaints(raw_pipe, ["A debt collector called my workplace about a bill."], top_k=2)
    assert raw_preds[0]["predicted_product"] == "Debt collection"
    assert 0.0 < raw_preds[0]["confidence"] <= 1.0


def test_load_corrupted_joblib_raises_value_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.joblib"
    corrupt.write_bytes(b"not-a-valid-joblib-pickle")
    with pytest.raises(ValueError, match="corrupted or unreadable"):
        load_model_bundle(corrupt)


def test_settings_loads_dotenv_without_overriding_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DATA_PATH=custom/rows.csv\nMODEL_PATH=custom/model.joblib\n", encoding="utf-8")
    monkeypatch.setenv("DATA_PATH", "")
    monkeypatch.delenv("DATA_PATH")
    monkeypatch.setenv("MODEL_PATH", "override.joblib")

    cfg = Settings.from_env(env_file=env_file)
    assert cfg.data_path.as_posix().endswith("custom/rows.csv")
    assert cfg.model_path.name == "override.joblib"
