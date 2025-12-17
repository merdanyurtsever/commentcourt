from model.base import BaseMLModel, PredictionResult, ModelMetrics
from model.registry import ModelRegistry
from pathlib import Path
import json


@ModelRegistry.register("merdan_classic", "Merdan's classic placeholder model")
class MerdanClassic(BaseMLModel):
    def __init__(self, name: str = "merdan_classic", version: str = "0.0.0"):
        super().__init__(name, version)
        # Try to load naive bias weights if present
        self.bias = {}
        bias_path = Path('model/weights/merdan_bias.json')
        try:
            if bias_path.exists():
                with open(bias_path, 'r', encoding='utf-8') as f:
                    self.bias = json.load(f)
        except Exception:
            self.bias = {}

    def train(self, texts, labels, validation_split: float = 0.2, **kwargs) -> ModelMetrics:
        self.is_trained = True
        self.metrics = ModelMetrics(f1_score=0.0, dataset_size=len(texts))
        return self.metrics

    def predict_single(self, text: str) -> PredictionResult:
        # Use rule-based model if available, then apply a tiny naive bias.
        try:
            rule = ModelRegistry.get('rule_based')
            pred = rule.predict_single(text)
        except Exception:
            pred = PredictionResult(text=text, sentiment='neutral', confidence=0.5, scores={})

        # Naive bias: simple keyword heuristics to nudge sentiment/confidence
        txt = (text or '').lower()
        positive_keywords = ['iyi', 'güzel', 'harika', 'teşekkür', 'mükemmel', 'indirim', 'ücretsiz']
        negative_keywords = ['berbat', 'kötü', 'çöp', 'iade', 'şikayet', 'rezalet']

        # If positive keywords appear, boost toward positive
        if any(k in txt for k in positive_keywords):
            pred.sentiment = 'positive'
            pred.confidence = max(pred.confidence or 0.0, 0.6)
            # add/adjust scores dict
            scores = dict(pred.scores or {})
            scores['positive'] = max(scores.get('positive', 0.0), pred.confidence)
            pred.scores = scores
        elif any(k in txt for k in negative_keywords):
            pred.sentiment = 'negative'
            pred.confidence = max(pred.confidence or 0.0, 0.6)
            scores = dict(pred.scores or {})
            scores['negative'] = max(scores.get('negative', 0.0), pred.confidence)
            pred.scores = scores

        # Apply generated bias weights (if any)
        try:
            kw = self.bias.get('keyword_weights', {}) or {}
            mismatch_pen = float(self.bias.get('mismatch_penalty', 1.0))

            words = set(txt.split())
            # Adjust scores according to keyword weights
            scores = dict(pred.scores or {})
            for w in words:
                if w in kw:
                    weight = float(kw[w])
                    # If word is present, nudge positive/negative scores
                    # Use a heuristic: if it looks positive/negative, adjust that side
                    if w in positive_keywords:
                        scores['positive'] = scores.get('positive', 0.0) * weight
                    if w in negative_keywords:
                        scores['negative'] = scores.get('negative', 0.0) * weight

            # Re-normalize scores and set final sentiment/confidence
            total = sum(scores.values()) if scores else 0.0
            if total > 0:
                for k in scores:
                    scores[k] = scores[k] / total

                # set sentiment to argmax
                best = max(scores.items(), key=lambda x: x[1])
                pred.sentiment = best[0]
                # reduce confidence a bit if ambiguous (both pos and neg present)
                if scores.get('positive', 0.0) > 0 and scores.get('negative', 0.0) > 0:
                    pred.confidence = float(max(scores.get(pred.sentiment, 0.0) * mismatch_pen, 0.5))
                else:
                    pred.confidence = float(max(scores.get(pred.sentiment, 0.0), 0.5))

                pred.scores = scores
        except Exception:
            # Fall back silently to previous prediction
            pass

        return pred

    def predict(self, texts):
        try:
            rule = ModelRegistry.get('rule_based')
            return rule.predict(texts)
        except Exception:
            return [self.predict_single(t) for t in texts]

    def evaluate(self, texts, labels):
        try:
            rule = ModelRegistry.get('rule_based')
            return rule.evaluate(texts, labels)
        except Exception:
            return ModelMetrics(f1_score=0.0, dataset_size=len(texts))

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.save_metadata(path)

    def load(self, path):
        self.is_trained = True
        return
