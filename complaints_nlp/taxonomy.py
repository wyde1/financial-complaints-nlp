"""CFPB historical product taxonomy consolidation (18 raw labels -> 9 canonical classes).

In April 2017 the CFPB revised its complaint product taxonomy, replacing several
pre-2017 product names with consolidated labels (e.g., ``Credit reporting`` ->
``Credit reporting, credit repair services, or other personal consumer reports``).
Leaving pre-2017 and post-2017 aliases as separate classes forces text-only models
to guess submission dates rather than financial product semantics.
"""

from __future__ import annotations

DEPRECATED_PRODUCTS: frozenset[str] = frozenset({"Other financial service"})

PRODUCT_CONSOLIDATION_MAP: dict[str, str] = {
    # Credit reporting (pre/post April 2017 & post-2023 CFPB portal wording)
    "Credit reporting": "Credit reporting & repair",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting & repair",
    "Credit reporting or other personal consumer reports": "Credit reporting & repair",
    # Debt collection
    "Debt collection": "Debt collection",
    # Mortgage
    "Mortgage": "Mortgage",
    # Credit & prepaid cards (pre/post April 2017)
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Credit card or prepaid card": "Credit card or prepaid card",
    # Checking & savings / bank accounts (pre/post April 2017)
    "Bank account or service": "Checking or savings account",
    "Checking or savings account": "Checking or savings account",
    # Student loans
    "Student loan": "Student loan",
    # Vehicle & consumer loans (pre/post April 2017)
    "Consumer Loan": "Vehicle or consumer loan",
    "Vehicle loan or lease": "Vehicle or consumer loan",
    # Money transfers & virtual currency (pre/post April 2017)
    "Money transfers": "Money transfer or virtual currency",
    "Virtual currency": "Money transfer or virtual currency",
    "Money transfer, virtual currency, or money service": "Money transfer or virtual currency",
    # Payday, title & personal loans (pre/post April 2017)
    "Payday loan": "Payday, title, or personal loan",
    "Payday loan, title loan, or personal loan": "Payday, title, or personal loan",
}

CANONICAL_PRODUCTS: tuple[str, ...] = (
    "Checking or savings account",
    "Credit card or prepaid card",
    "Credit reporting & repair",
    "Debt collection",
    "Money transfer or virtual currency",
    "Mortgage",
    "Payday, title, or personal loan",
    "Student loan",
    "Vehicle or consumer loan",
)

SHORT_PRODUCT_LABELS: dict[str, str] = {
    "Checking or savings account": "Checking / Savings",
    "Credit card or prepaid card": "Credit / Prepaid Card",
    "Credit reporting & repair": "Credit Reporting",
    "Debt collection": "Debt Collection",
    "Money transfer or virtual currency": "Money Transfer / Crypto",
    "Mortgage": "Mortgage",
    "Payday, title, or personal loan": "Payday / Personal Loan",
    "Student loan": "Student Loan",
    "Vehicle or consumer loan": "Vehicle / Consumer Loan",
}


def consolidate_product(
    raw_product: object,
    drop_deprecated: bool = True,
) -> str | None:
    """Map a raw CFPB ``Product`` label to its canonical financial category.

    Returns ``None`` when ``raw_product`` is null/empty or deprecated (when ``drop_deprecated`` is True).
    """
    if raw_product is None:
        return None
    if isinstance(raw_product, float) and raw_product != raw_product:  # NaN check without pandas import
        return None
    cleaned = str(raw_product).strip()
    if not cleaned:
        return None
    if drop_deprecated and cleaned in DEPRECATED_PRODUCTS:
        return None
    return PRODUCT_CONSOLIDATION_MAP.get(cleaned, cleaned)
