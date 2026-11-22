"""Unit tests for CFPB narrative cleaning, taxonomy consolidation, and deduplication."""

from __future__ import annotations

import pandas as pd
import pytest

from complaints_nlp.config import NARRATIVE_COLUMN, PRODUCT_COLUMN
from complaints_nlp.preprocessing import (
    ComplaintTextCleaner,
    clean_narrative,
    prepare_complaints_dataframe,
)
from complaints_nlp.taxonomy import (
    CANONICAL_PRODUCTS,
    PRODUCT_CONSOLIDATION_MAP,
    consolidate_product,
)


def test_clean_narrative_strips_cfpb_masks_dates_and_is_idempotent() -> None:
    raw = (
        "On XX/XX/2018, XXXX Bank charged me {$250.00} on card 1234XXXX / acct_XXXX / XXXX-XXXX-1234! "
        "Next tax year I paid $500 via express transfer."
    )
    cleaned = clean_narrative(raw)
    assert "xxxx" not in cleaned
    assert "2018" not in cleaned
    assert "250" not in cleaned
    assert "bank charged me" in cleaned
    assert "acct" in cleaned
    # Legitimate words containing 'x' must be preserved.
    assert "next" in cleaned
    assert "tax" in cleaned
    assert "express" in cleaned
    # Strict idempotency check including digit/underscore-adjacent masks
    assert clean_narrative("1234XXXX") == ""
    assert clean_narrative("acct_XXXX") == "acct"
    assert clean_narrative(cleaned) == cleaned


def test_consolidate_product_handles_nulls_and_live_cfpb_alias() -> None:
    assert consolidate_product(None) is None
    assert consolidate_product(float("nan")) is None
    assert consolidate_product("   ") is None
    assert (
        consolidate_product("Credit reporting or other personal consumer reports")
        == "Credit reporting & repair"
    )


def test_cleaner_transformer_handles_nulls_and_lists() -> None:
    cleaner = ComplaintTextCleaner()
    out = cleaner.fit_transform(["XXXX XX/XX/2017", None, "Mortgage escrow issue!"])
    assert out == ["", "", "mortgage escrow issue"]


def test_taxonomy_consolidation_maps_all_17_non_deprecated_to_9_canonical() -> None:
    assert len(CANONICAL_PRODUCTS) == 9
    assert set(PRODUCT_CONSOLIDATION_MAP.values()) == set(CANONICAL_PRODUCTS)
    assert consolidate_product("Credit reporting") == "Credit reporting & repair"
    assert consolidate_product("Virtual currency") == "Money transfer or virtual currency"
    assert consolidate_product("Other financial service", drop_deprecated=True) is None
    assert consolidate_product("Other financial service", drop_deprecated=False) == "Other financial service"


def test_deduplication_uses_cleaned_text_and_drops_conflicting_labels() -> None:
    raw_df = pd.DataFrame(
        {
            NARRATIVE_COLUMN: [
                # Two template duplicates differing only by redacted date & casing -> collapse to 1
                "On XX/XX/2016 XXXX reported an inaccurate collection on my credit file.",
                "on XX/XX/2018 XXXX reported an inaccurate collection on my credit file!",
                # Exact normalized text with conflicting labels -> drop BOTH rows
                "I called XXXX about a late payment fee on my account.",
                "I called XXXX about a late payment fee on my account!",
                # Deprecated product -> dropped
                "General complaint about a check cashing store.",
                # Empty after stripping redaction masks -> dropped
                "XXXX XX/XX/2018 {$100.00}",
                # Distinct valid row
                "My mortgage escrow disbursement was delayed by two months.",
            ],
            PRODUCT_COLUMN: [
                "Credit reporting",
                "Credit reporting, credit repair services, or other personal consumer reports",
                "Credit card",
                "Debt collection",
                "Other financial service",
                "Mortgage",
                "Mortgage",
            ],
            "Sub-product": ["ignored"] * 7,
            "Issue": ["ignored_leakage"] * 7,
        }
    )

    df, stats = prepare_complaints_dataframe(raw_df)

    assert stats.raw_rows == 7
    assert stats.dropped_deprecated_rows == 1
    assert stats.empty_after_cleaning_rows == 1
    assert stats.conflicting_label_rows_removed == 2
    assert stats.exact_duplicate_rows_removed == 1
    assert stats.final_rows == 2
    assert set(df["product"]) == {"Credit reporting & repair", "Mortgage"}
    assert "Sub-product" not in df.columns
    assert "Issue" not in df.columns


def test_prepare_dataframe_missing_required_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        prepare_complaints_dataframe(pd.DataFrame({"foo": ["bar"]}))
