"""Leakage-free text cleaning, CFPB mask stripping, and dataset deduplication."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from complaints_nlp.config import NARRATIVE_COLUMN, PRODUCT_COLUMN
from complaints_nlp.taxonomy import consolidate_product

# Compiled regex patterns for deterministic, fast cleaning of CFPB narratives.
_REDACTED_DATE_RE = re.compile(
    r"\b[xX]{2,4}[/-][xX]{2,4}(?:[/-](?:[xX]{2,4}|\d{2,4}))?\b"
)
_CURLY_AMOUNT_RE = re.compile(r"\{\$?[\d,.\sxX]+\}")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]+")
_REDACTED_X_RUN_RE = re.compile(r"\bx{2,}\b")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_narrative(text: object) -> str:
    """Normalize a CFPB consumer complaint narrative for TF-IDF modeling.

    Specifically removes CFPB redaction masks (``XXXX``, ``XX/XX/2018``, ``1234XXXX``,
    ``acct_XXXX``), curly-braced dollar tokens, digits (preventing calendar-year
    temporal leakage), and punctuation, returning lowercased whitespace-normalized text.

    This function is strictly idempotent: ``clean_narrative(clean_narrative(x)) == clean_narrative(x)``.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    s = _REDACTED_DATE_RE.sub(" ", s)
    s = _CURLY_AMOUNT_RE.sub(" ", s)
    s = s.lower()
    s = _NON_ALPHA_RE.sub(" ", s)
    s = _REDACTED_X_RUN_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


class ComplaintTextCleaner(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer applying ``clean_narrative`` inside a Pipeline."""

    def fit(self, X: Iterable[object], y: object = None) -> ComplaintTextCleaner:
        return self

    def transform(self, X: Iterable[object]) -> list[str]:
        return [clean_narrative(item) for item in X]


@dataclass(frozen=True)
class DatasetStats:
    """Audit statistics recorded during dataset loading and deduplication."""

    raw_rows: int
    non_empty_narrative_rows: int
    dropped_deprecated_rows: int
    empty_after_cleaning_rows: int
    conflicting_label_rows_removed: int
    exact_duplicate_rows_removed: int
    final_rows: int
    class_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "raw_rows": self.raw_rows,
            "non_empty_narrative_rows": self.non_empty_narrative_rows,
            "dropped_deprecated_rows": self.dropped_deprecated_rows,
            "empty_after_cleaning_rows": self.empty_after_cleaning_rows,
            "conflicting_label_rows_removed": self.conflicting_label_rows_removed,
            "exact_duplicate_rows_removed": self.exact_duplicate_rows_removed,
            "final_rows": self.final_rows,
            "class_counts": dict(self.class_counts),
        }


def prepare_complaints_dataframe(
    raw_df: pd.DataFrame,
    consolidate: bool = True,
    drop_deprecated: bool = True,
    deduplicate: bool = True,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DatasetStats]:
    """Filter, clean, consolidate taxonomy, and deduplicate complaint narratives.

    Uses ONLY ``Consumer complaint narrative`` and ``Product`` to prevent target leakage
    from correlated metadata columns (``Sub-product``, ``Issue``, ``Sub-issue``).
    Deduplication runs on the normalized/cleaned narrative (with CFPB ``XXXX`` and
    redacted dates stripped) before splitting, and drops any normalized text whose
    exact duplicates carry conflicting product labels.
    """
    missing_cols = {NARRATIVE_COLUMN, PRODUCT_COLUMN} - set(raw_df.columns)
    if missing_cols:
        raise ValueError(f"Input dataset is missing required column(s): {sorted(missing_cols)}")

    raw_rows = len(raw_df)
    df = raw_df[[NARRATIVE_COLUMN, PRODUCT_COLUMN]].copy()

    # Keep only rows with non-empty narrative and non-null product label.
    valid_mask = (
        df[NARRATIVE_COLUMN].notna()
        & df[PRODUCT_COLUMN].notna()
        & (df[NARRATIVE_COLUMN].astype(str).str.strip() != "")
        & (df[PRODUCT_COLUMN].astype(str).str.strip() != "")
    )
    df = df.loc[valid_mask].copy()
    non_empty_rows = len(df)

    # Consolidate pre/post April 2017 CFPB product labels.
    if consolidate:
        df["product"] = df[PRODUCT_COLUMN].map(
            lambda p: consolidate_product(p, drop_deprecated=drop_deprecated)
        )
    else:
        df["product"] = df[PRODUCT_COLUMN].astype(str).str.strip()

    before_drop_deprecated = len(df)
    df = df[df["product"].notna()].copy()
    dropped_deprecated_rows = before_drop_deprecated - len(df)

    # Normalize text and drop narratives that become empty once masks/punctuation are stripped.
    df["cleaned_text"] = [clean_narrative(t) for t in df[NARRATIVE_COLUMN].tolist()]
    before_empty_clean = len(df)
    df = df[df["cleaned_text"] != ""].copy()
    empty_after_cleaning_rows = before_empty_clean - len(df)

    conflicting_label_rows_removed = 0
    exact_duplicate_rows_removed = 0

    if deduplicate and len(df) > 0:
        # Remove normalized narratives that appear under conflicting product labels.
        label_nunique = df.groupby("cleaned_text")["product"].transform("nunique")
        conflict_mask = label_nunique > 1
        conflicting_label_rows_removed = int(conflict_mask.sum())
        if conflicting_label_rows_removed > 0:
            df = df.loc[~conflict_mask].copy()

        # Deduplicate exact normalized narratives so templates cannot leak across splits.
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["cleaned_text"], keep="first").copy()
        exact_duplicate_rows_removed = before_dedup - len(df)

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        n_classes = int(df["product"].nunique())
        if max_rows < n_classes:
            raise ValueError(
                f"max_rows ({max_rows}) must be at least the number of classes ({n_classes})."
            )
        df, _ = train_test_split(
            df,
            train_size=max_rows,
            stratify=df["product"],
            random_state=random_state,
        )
        df = df.copy()

    df = df.rename(columns={NARRATIVE_COLUMN: "raw_text"})[["raw_text", "cleaned_text", "product"]]
    df = df.reset_index(drop=True)

    counts = {str(k): int(v) for k, v in df["product"].value_counts().items()}
    stats = DatasetStats(
        raw_rows=raw_rows,
        non_empty_narrative_rows=non_empty_rows,
        dropped_deprecated_rows=dropped_deprecated_rows,
        empty_after_cleaning_rows=empty_after_cleaning_rows,
        conflicting_label_rows_removed=conflicting_label_rows_removed,
        exact_duplicate_rows_removed=exact_duplicate_rows_removed,
        final_rows=len(df),
        class_counts=counts,
    )
    return df, stats


def load_complaints_csv(
    csv_path: Path,
    consolidate: bool = True,
    drop_deprecated: bool = True,
    deduplicate: bool = True,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DatasetStats]:
    """Load only the narrative and product columns from a CFPB CSV and prepare the dataset."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found at '{csv_path}'. "
            "Download 'rows.csv' from Kaggle/CFPB into data/rows.csv or pass --data-path."
        )

    header = pd.read_csv(csv_path, nrows=0)
    missing = {NARRATIVE_COLUMN, PRODUCT_COLUMN} - set(header.columns)
    if missing:
        raise ValueError(f"CSV '{csv_path}' is missing required column(s): {sorted(missing)}")

    raw_df = pd.read_csv(
        csv_path,
        usecols=[NARRATIVE_COLUMN, PRODUCT_COLUMN],
        dtype=str,
        low_memory=False,
    )
    return prepare_complaints_dataframe(
        raw_df=raw_df,
        consolidate=consolidate,
        drop_deprecated=drop_deprecated,
        deduplicate=deduplicate,
        max_rows=max_rows,
        random_state=random_state,
    )
