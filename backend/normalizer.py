"""Placeholder for normalization logic.

This module provides a small Normalizer class with a `normalize` method
that can be extended later with rule-based or learned normalization.
"""
from typing import Optional


class Normalizer:
    """Placeholder normalizer — identity by default."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def normalize(self, text: str) -> str:
        """Normalize text. Currently returns text unchanged."""
        return text
