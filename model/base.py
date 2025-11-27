"""
Base classes for ML models in the CommentCourt framework.

This module provides abstract base classes that all ML models must inherit from,
ensuring a consistent interface for training, prediction, and model management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime


@dataclass
class PredictionResult:
    """Container for a single prediction result."""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0.0 to 1.0
    scores: Dict[str, float] = field(default_factory=dict)  # per-class scores
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'scores': self.scores
        }


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    evaluation_time: Optional[datetime] = None
    dataset_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix,
            'classification_report': self.classification_report,
            'evaluation_time': self.evaluation_time.isoformat() if self.evaluation_time else None,
            'dataset_size': self.dataset_size
        }


class BaseMLModel(ABC):
    """
    Abstract base class for all ML models in the CommentCourt framework.
    
    All models must implement this interface to ensure consistent behavior
    across the pipeline. Models are designed to be:
    - Pluggable: Easy to add new models
    - Comparable: Uniform evaluation metrics
    - Serializable: Save/load model state
    
    Attributes:
        name (str): Unique identifier for the model
        version (str): Model version string
        is_trained (bool): Whether the model has been trained
        metrics (ModelMetrics): Latest evaluation metrics
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self._config: Dict[str, Any] = {}
    
    @property
    def model_id(self) -> str:
        """Unique identifier combining name and version."""
        return f"{self.name}_v{self.version}"
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[str], 
              validation_split: float = 0.2, **kwargs) -> ModelMetrics:
        """
        Train the model on the provided data.
        
        Args:
            texts: List of input text samples
            labels: List of corresponding labels ('positive', 'negative', 'neutral')
            validation_split: Fraction of data to use for validation
            **kwargs: Model-specific training parameters
            
        Returns:
            ModelMetrics with training results
        """
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """
        Generate predictions for input texts.
        
        Args:
            texts: List of text samples to classify
            
        Returns:
            List of PredictionResult objects with sentiment and confidence
        """
        pass
    
    @abstractmethod
    def predict_single(self, text: str) -> PredictionResult:
        """
        Generate prediction for a single text.
        
        Args:
            text: Single text sample to classify
            
        Returns:
            PredictionResult with sentiment and confidence
        """
        pass
    
    @abstractmethod
    def evaluate(self, texts: List[str], labels: List[str]) -> ModelMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            texts: List of test text samples
            labels: List of ground truth labels
            
        Returns:
            ModelMetrics with evaluation results
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model state to disk.
        
        Args:
            path: Directory path to save model files
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model state from disk.
        
        Args:
            path: Directory path containing model files
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'name': self.name,
            'version': self.version,
            'model_id': self.model_id,
            'is_trained': self.is_trained,
            'config': self._config
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set model configuration."""
        self._config.update(config)
    
    def save_metadata(self, path: Path) -> None:
        """Save model metadata to JSON file."""
        metadata = {
            'name': self.name,
            'version': self.version,
            'model_id': self.model_id,
            'is_trained': self.is_trained,
            'config': self._config,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'saved_at': datetime.now().isoformat()
        }
        metadata_path = path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_metadata(self, path: Path) -> Dict[str, Any]:
        """Load model metadata from JSON file."""
        metadata_path = path / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"<{self.__class__.__name__}(name={self.name}, version={self.version}, {status})>"


class BasePreprocessor(ABC):
    """
    Abstract base class for text preprocessors.
    
    Preprocessors handle text cleaning and normalization before
    feeding data to models. Different models may require different
    preprocessing strategies.
    """
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text sample.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        pass
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple text samples.
        
        Args:
            texts: List of raw input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class TurkishTextPreprocessor(BasePreprocessor):
    """
    Preprocessor for Turkish text with language-specific handling.
    
    Handles Turkish-specific characters and common preprocessing
    steps for Turkish e-commerce comments.
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_emojis: bool = False):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_emojis = remove_emojis
        
        # Turkish-specific character mappings
        self.turkish_chars = {
            'ı': 'i', 'İ': 'I',
            'ğ': 'g', 'Ğ': 'G',
            'ü': 'u', 'Ü': 'U',
            'ş': 's', 'Ş': 'S',
            'ö': 'o', 'Ö': 'O',
            'ç': 'c', 'Ç': 'C'
        }
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess Turkish text.
        
        Steps:
        1. Convert to lowercase (preserving Turkish chars)
        2. Remove/normalize punctuation
        3. Optionally remove numbers
        4. Normalize whitespace
        """
        if not text:
            return ""
        
        # Turkish-aware lowercase
        if self.lowercase:
            # Handle Turkish I/ı properly
            text = text.replace('I', 'ı').replace('İ', 'i')
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            import string
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
        
        # Remove numbers
        if self.remove_numbers:
            import re
            text = re.sub(r'\d+', '', text)
        
        # Remove emojis
        if self.remove_emojis:
            import re
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub('', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
