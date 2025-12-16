from model.base import BaseMLModel, PredictionResult, ModelMetrics
from model.registry import ModelRegistry


@ModelRegistry.register("ethem_classic", "Ethem's classic placeholder model")
class EthemClassic(BaseMLModel):
    def __init__(self, name: str = "ethem_classic", version: str = "0.0.0"):
        super().__init__(name, version)

    def train(self, texts, labels, validation_split: float = 0.2, **kwargs) -> ModelMetrics:
        # Placeholder training — mark as trained and return a default metric
        self.is_trained = True
        self.metrics = ModelMetrics(f1_score=0.0, accuracy=0.0, precision=0.0, recall=0.0, dataset_size=len(texts))
        return self.metrics

    def predict_single(self, text: str) -> PredictionResult:
        # Delegate to rule-based baseline if available
        try:
            rule = ModelRegistry.get('rule_based')
            return rule.predict_single(text)
        except Exception:
            return PredictionResult(text=text, sentiment='neutral', confidence=0.5, scores={})

    def predict(self, texts):
        try:
            rule = ModelRegistry.get('rule_based')
            return rule.predict(texts)
        except Exception:
            return [self.predict_single(t) for t in texts]

    def evaluate(self, texts, labels):
        # Delegate if possible
        try:
            rule = ModelRegistry.get('rule_based')
            return rule.evaluate(texts, labels)
        except Exception:
            return ModelMetrics(f1_score=0.0, accuracy=0.0, precision=0.0, recall=0.0, dataset_size=len(texts))

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.save_metadata(path)

    def load(self, path):
        # No artifacts yet — treat as trained
        self.is_trained = True
        return
