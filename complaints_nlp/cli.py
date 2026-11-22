"""Command-line interface for training and inference on financial consumer complaints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from complaints_nlp.config import DEFAULT_METRICS_PATH, NARRATIVE_COLUMN, Settings
from complaints_nlp.modeling import (
    load_model_bundle,
    predict_complaints,
    save_evaluation_plots,
    save_model_bundle,
    train_and_evaluate,
)
from complaints_nlp.preprocessing import clean_narrative, load_complaints_csv


def _positive_int(raw: str) -> int:
    try:
        val = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {raw!r}.") from exc
    if val < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return val


def build_parser(defaults: Settings | None = None) -> argparse.ArgumentParser:
    cfg = defaults or Settings.from_env()
    parser = argparse.ArgumentParser(
        description="Leakage-free CFPB financial consumer complaint classification CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand ---
    train_p = subparsers.add_parser(
        "train",
        help="Train and benchmark leak-free TF-IDF pipelines on a CFPB CSV.",
    )
    train_p.add_argument(
        "--data-path",
        type=Path,
        default=cfg.data_path,
        help=f"Path to CFPB rows.csv (default: {cfg.data_path}).",
    )
    train_p.add_argument(
        "--model-out",
        type=Path,
        default=cfg.model_path,
        help=f"Output path for saved .joblib pipeline (default: {cfg.model_path}).",
    )
    train_p.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Output path for JSON evaluation report (default: {DEFAULT_METRICS_PATH}).",
    )
    train_p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional directory to save evaluation PNG charts (e.g. docs/assets).",
    )
    train_p.add_argument(
        "--max-rows",
        type=_positive_int,
        default=None,
        help="Optional stratified row cap after deduplication (useful for quick local experiments).",
    )
    train_p.add_argument(
        "--max-features",
        type=_positive_int,
        default=50_000,
        help="Maximum TF-IDF vocabulary size (default: 50000).",
    )
    train_p.add_argument(
        "--min-df",
        type=_positive_int,
        default=5,
        help="Minimum document frequency for TF-IDF terms (default: 5).",
    )
    train_p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits and models (default: 42).",
    )

    # --- predict subcommand ---
    pred_p = subparsers.add_parser(
        "predict",
        help="Classify a single complaint text or batch CSV using a trained .joblib model.",
    )
    src_group = pred_p.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--text", type=str, help="Raw consumer complaint narrative text.")
    src_group.add_argument(
        "--input-csv",
        type=Path,
        help=f"CSV file containing a '{NARRATIVE_COLUMN}' (or --text-col) column.",
    )
    pred_p.add_argument(
        "--text-col",
        type=str,
        default=NARRATIVE_COLUMN,
        help=f"Text column name for --input-csv (default: '{NARRATIVE_COLUMN}').",
    )
    pred_p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path when using --input-csv.",
    )
    pred_p.add_argument(
        "--model-path",
        type=Path,
        default=cfg.model_path,
        help=f"Path to saved .joblib model (default: {cfg.model_path}).",
    )
    pred_p.add_argument(
        "--top-k",
        type=_positive_int,
        default=3,
        help="Number of top candidate classes to display (default: 3).",
    )
    pred_p.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print structured JSON prediction output.",
    )
    return parser


def _handle_train(args: argparse.Namespace) -> int:
    df, stats = load_complaints_csv(
        csv_path=args.data_path,
        consolidate=True,
        drop_deprecated=True,
        deduplicate=True,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )
    print(
        f"Loaded {stats.raw_rows:,} raw rows -> {stats.non_empty_narrative_rows:,} non-empty narratives -> "
        f"{stats.final_rows:,} unique canonical complaints "
        f"(removed {stats.dropped_deprecated_rows:,} deprecated, "
        f"{stats.conflicting_label_rows_removed:,} cross-label collisions, "
        f"{stats.exact_duplicate_rows_removed:,} exact duplicates)."
    )

    pipeline, report, plot_payload = train_and_evaluate(
        texts=df["cleaned_text"].tolist(),
        labels=df["product"].tolist(),
        dataset_stats=stats,
        max_features=args.max_features,
        min_df=args.min_df,
        random_state=args.random_state,
    )

    save_model_bundle(pipeline, metadata=report, output_path=args.model_out)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.plots_dir is not None:
        saved_plots = save_evaluation_plots(args.plots_dir, report, plot_payload)
        print(f"Saved {len(saved_plots)} evaluation plots to '{args.plots_dir}'.")

    print("\nValidation Benchmark (sorted by Macro-F1):")
    for row in sorted(report["validation_benchmarks"], key=lambda r: r["val_macro_f1"], reverse=True):
        print(
            f"  {row['display_name']:32} | Macro-F1={row['val_macro_f1']:.4f} | "
            f"Weighted-F1={row['val_weighted_f1']:.4f} | Acc={row['val_accuracy']:.4f} | "
            f"{row['fit_seconds']:.1f}s"
        )

    sel = report["selected_model"]
    tm = report["test_metrics"]
    cal_note = " (calibrated via train-only CV)" if sel["calibrated_on_train_cv"] else ""
    print(
        f"\nSelected model: {sel['display_name']}{cal_note}\n"
        f"Held-out Test  : Macro-F1={tm['macro_f1']:.4f} | "
        f"Weighted-F1={tm['weighted_f1']:.4f} | Accuracy={tm['accuracy']:.4f}\n"
        f"Saved pipeline : '{args.model_out}'\n"
        f"Saved metrics  : '{args.metrics_out}'"
    )
    return 0


def _handle_predict(args: argparse.Namespace) -> int:
    # Validate text/CSV inputs and column headers before loading the joblib model artifact.
    if args.text is not None:
        if not clean_narrative(args.text):
            raise ValueError("Input --text is empty after removing redaction masks and punctuation.")
    else:
        if not args.input_csv.is_file():
            raise FileNotFoundError(f"Input CSV not found: '{args.input_csv}'")
        header = pd.read_csv(args.input_csv, nrows=0)
        if args.text_col not in header.columns:
            raise ValueError(
                f"Column '{args.text_col}' not found in '{args.input_csv}'. "
                f"Available columns: {list(header.columns)}"
            )

    pipeline, _meta = load_model_bundle(args.model_path)

    if args.text is not None:
        pred = predict_complaints(pipeline, [args.text], top_k=args.top_k)[0]
        if args.as_json:
            print(json.dumps(pred, indent=2))
        else:
            print(
                f"Predicted Product: {pred['predicted_product']} "
                f"(confidence: {pred['confidence']:.2%})"
            )
            print("Top candidates:")
            for cand in pred["top_predictions"]:
                print(f"  - {cand['product']:38} {cand['probability']:.2%}")
        return 0

    df_in = pd.read_csv(args.input_csv, dtype=str, low_memory=False)
    texts = df_in[args.text_col].fillna("").tolist()
    preds = predict_complaints(pipeline, texts, top_k=args.top_k)
    df_out = df_in.copy()
    df_out["predicted_product"] = [p["predicted_product"] for p in preds]
    df_out["confidence"] = [p["confidence"] for p in preds]

    out_csv = args.output_csv or args.input_csv.with_name(f"{args.input_csv.stem}_predictions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Classified {len(df_out):,} complaint(s) -> '{out_csv}'.")
    return 0


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return _handle_train(args)
        if args.command == "predict":
            return _handle_predict(args)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
