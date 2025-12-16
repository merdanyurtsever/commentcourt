"""Inference helpers for transformer-based sentiment models.

Provides a small wrapper around Hugging Face Transformers for loading
an AutoModelForSequenceClassification checkpoint and computing a
continuous sentiment score in [0, 1]. If no transformer checkpoint
is provided, falls back to a registered BaseMLModel via ModelRegistry.
"""
from typing import List, Optional
import logging

import numpy as np

from model.registry import ModelRegistry
from model.base import PredictionResult

logger = logging.getLogger(__name__)


class TransformerInference:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 positive_class_index: int = -1):
        """Initialize the inference helper.

        Args:
            model_path: Path or HF id of a seq-class checkpoint. If None, no transformer is loaded.
            device: 'cpu' or 'cuda' or None to auto-select. If None, tries to use torch if available.
            positive_class_index: index of the positive class in multi-class logits. Default -1 (last class).
        """
        self.model_path = model_path
        self.device = device
        self.positive_class_index = positive_class_index

        self.tokenizer = None
        self.model = None
        self._torch = None

        if model_path:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                self._torch = torch
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()

                logger.info(f"Loaded transformer model from {model_path} on {self.device}")
            except Exception as e:
                logger.error(f"Could not load transformer model: {e}")
                self.tokenizer = None
                self.model = None
                self._torch = None

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-logits))

    def predict_scores(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """Return a continuous sentiment score in [0,1] for each text.

        If a transformer model is loaded, runs batched inference.
        Otherwise, falls back to the best registered BaseMLModel and
        uses the model's positive class probability when available.
        """
        if self.model is None or self.tokenizer is None:
            # Fallback to a registered rule-based model
            logger.info("No transformer loaded, falling back to registered model (rule_based)")
            try:
                model = ModelRegistry.get('rule_based')
            except Exception:
                model = None

            if model is None:
                raise RuntimeError("No model available for inference")

            preds = model.predict(texts)
            scores = []
            for p in preds:
                # Prefer an explicit 'positive' score in scores dict
                s = None
                if p.scores and 'positive' in p.scores:
                    s = p.scores['positive']
                else:
                    # Map categorical sentiment to numeric
                    if p.sentiment == 'positive':
                        s = 1.0
                    elif p.sentiment == 'negative':
                        s = 0.0
                    else:
                        s = 0.5
                scores.append(float(s))
            return scores

        # Transformer path
        torch = self._torch
        device = torch.device(self.device)

        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model(**enc)
                logits = outputs.logits.detach().cpu().numpy()

                if logits.shape[1] == 1:
                    # Single-output regression / sigmoid
                    prob = self._sigmoid(logits).reshape(-1)
                else:
                    probs = self._softmax(logits)
                    idx = self.positive_class_index
                    if idx < 0:
                        idx = probs.shape[1] - 1
                    prob = probs[:, idx]

                scores.extend([float(x) for x in prob.tolist()])

        return scores

    def predict_binary(self, texts: List[str], threshold: float = 0.5, batch_size: int = 16) -> List[int]:
        s = self.predict_scores(texts, batch_size=batch_size)
        return [1 if x > threshold else 0 for x in s]
