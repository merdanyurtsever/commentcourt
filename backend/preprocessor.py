"""Preprocessor wrapper to centralize cleaning and normalization.

This module provides `Preprocessor` that combines the existing
`TextCleaner` with a `Normalizer` instance to produce final
preprocessed text for models.
"""
from typing import List, Optional
from backend.cleaner import TextCleaner, CleanerConfig
from backend.normalizer import Normalizer


class Preprocessor:
    """High-level preprocessor used by model training code."""

    def __init__(self, cleaner_config: Optional[dict] = None,
                 normalizer: Optional[Normalizer] = None):
        self.cleaner = TextCleaner(CleanerConfig(**(cleaner_config or {})))
        self.normalizer = normalizer or Normalizer()

    def preprocess(self, text: str) -> str:
        """Clean then normalize a single text string."""
        cleaned = self.cleaner.clean(text)
        return self.normalizer.normalize(cleaned)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Process multiple texts."""
        cleaned = self.cleaner.clean_batch(texts)
        return [self.normalizer.normalize(t) for t in cleaned]
