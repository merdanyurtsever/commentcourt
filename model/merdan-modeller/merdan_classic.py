from model.base import BaseMLModel, PredictionResult, ModelMetrics
from model.registry import ModelRegistry


@ModelRegistry.register("merdan_classic", "Merdan's classic placeholder model")
class MerdanClassic(BaseMLModel):
    def __init__(self, name: str = "merdan_classic", version: str = "0.0.0"):
        super().__init__(name, version)

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
