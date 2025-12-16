"""Small, explicit transformer inference helper.

This uses torch when available and otherwise falls back to a registered
rule-based model. The implementation avoids numpy and keeps the logic
simple so it's easy to read and follow.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

from model.registry import ModelRegistry
from model.base import PredictionResult


class TransformerInference:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 positive_class_index: int = -1):
        # Simple attributes
        self.model_path = model_path
        self.device = device
        self.positive_class_index = positive_class_index

        self.tokenizer = None
        self.model = None
        self._torch = None

        if model_path:
            try:
                # Import torch and transformers lazily to avoid hard deps
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                self._torch = torch
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                # If anything fails, we keep tokenizer/model as None and fall back
                self.tokenizer = None
                self.model = None
                self._torch = None

    def _sigmoid(self, x):
        # x can be a torch tensor or a float
        try:
            import math
            return 1.0 / (1.0 + math.exp(-float(x)))
        except Exception:
            return 0.5

    def predict_scores(self, texts: List[str], batch_size: int = 16) -> List[float]:
        # If we don't have a transformer model, use rule-based fallback
        if not self.model or not self.tokenizer:
            try:
                model = ModelRegistry.get('rule_based')
            except Exception:
                raise RuntimeError('No model available for inference')

            preds = model.predict(texts)
            scores = []
            for p in preds:
                s = None
                if p.scores and 'positive' in p.scores:
                    s = p.scores['positive']
                else:
                    if p.sentiment == 'positive':
                        s = 1.0
                    elif p.sentiment == 'negative':
                        s = 0.0
                    else:
                        s = 0.5
                scores.append(float(s))
            return scores

        # Transformer path using torch tensors and simple operations
        torch = self._torch
        device = torch.device(self.device)

        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model(**enc)
                logits = outputs.logits

                # If single logit per example, use sigmoid
                if logits.shape[-1] == 1:
                    for i in range(logits.shape[0]):
                        val = logits[i].item()
                        prob = self._sigmoid(val)
                        results.append(float(prob))
                else:
                    # multi-class: get positive class probability
                    for i in range(logits.shape[0]):
                        row = logits[i].tolist()
                        # softmax manually (small lists)
                        maxv = max(row)
                        exps = [float((v - maxv)) for v in row]
                        exps = [math.exp(v) for v in exps]
                        s = sum(exps) if sum(exps) != 0 else 1.0
                        probs = [e / s for e in exps]

                        idx = self.positive_class_index
                        if idx < 0:
                            idx = len(probs) - 1
                        results.append(float(probs[idx]))

        return results

    def predict_binary(self, texts: List[str], threshold: float = 0.5, batch_size: int = 16) -> List[int]:
        scores = self.predict_scores(texts, batch_size=batch_size)
        out = []
        for s in scores:
            out.append(1 if s > threshold else 0)
        return out