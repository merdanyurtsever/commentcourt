"""Very small, explicit preprocessor wrapper.

The goal is to make the flow clear: clean then normalize. No type hints
and simple loops/steps so a beginner can read it easily.
"""

from backend.cleaner import TextCleaner, CleanerConfig
from backend.normalizer import Normalizer


class Preprocessor:
    def __init__(self, cleaner_config=None, normalizer=None):
        self.cleaner = TextCleaner(CleanerConfig(**(cleaner_config or {})))
        self.normalizer = normalizer or Normalizer()

    def preprocess(self, text):
        # Clean the text first
        cleaned = self.cleaner.clean(text)
        # Then normalize the cleaned text
        return self.normalizer.normalize(cleaned)

    def preprocess_batch(self, texts):
        cleaned = self.cleaner.clean_batch(texts)
        out = []
        for t in cleaned:
            out.append(self.normalizer.normalize(t))
        return out