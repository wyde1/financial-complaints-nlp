"""Leakage-free multi-class NLP classification for CFPB financial consumer complaints."""

from complaints_nlp.config import Settings
from complaints_nlp.preprocessing import ComplaintTextCleaner, clean_narrative
from complaints_nlp.taxonomy import CANONICAL_PRODUCTS, PRODUCT_CONSOLIDATION_MAP, consolidate_product

__all__ = [
    "CANONICAL_PRODUCTS",
    "ComplaintTextCleaner",
    "PRODUCT_CONSOLIDATION_MAP",
    "Settings",
    "clean_narrative",
    "consolidate_product",
]
