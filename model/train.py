"""
Training Framework for CommentCourt ML Models.

This module provides a unified training interface for all registered models,
including:
- Dataset loading and splitting
- Multi-model training and evaluation
- Model comparison and selection
- Experiment tracking
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

import numpy as np
from sklearn.model_selection import train_test_split

from model.base import BaseMLModel, ModelMetrics
from model.registry import ModelRegistry, auto_discover_models


logger = logging.getLogger(__name__)


class Dataset:
    """
    Container for training, validation, and test data.
    
    Handles data loading, splitting, and stratification for
    sentiment analysis tasks.
    """
    
    def __init__(self, texts: List[str], labels: List[str]):
        """
        Initialize dataset with texts and labels.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels ('positive', 'negative', 'neutral')
        """
        self.texts = texts
        self.labels = labels
        
        self.train_texts: List[str] = []
        self.train_labels: List[str] = []
        self.val_texts: List[str] = []
        self.val_labels: List[str] = []
        self.test_texts: List[str] = []
        self.test_labels: List[str] = []
        
        self._is_split = False
    
    @property
    def size(self) -> int:
        """Total number of samples."""
        return len(self.texts)
    
    @property
    def class_distribution(self) -> Dict[str, int]:
        """Distribution of labels."""
        from collections import Counter
        return dict(Counter(self.labels))
    
    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
              test_ratio: float = 0.15, random_state: int = 42) -> None:
        """
        Split data into train, validation, and test sets.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # First split: train vs (val + test)
        self.train_texts, temp_texts, self.train_labels, temp_labels = train_test_split(
            self.texts, self.labels,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=self.labels
        )
        
        # Second split: val vs test
        val_fraction = val_ratio / (val_ratio + test_ratio)
        self.val_texts, self.test_texts, self.val_labels, self.test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=(1 - val_fraction),
            random_state=random_state,
            stratify=temp_labels
        )
        
        self._is_split = True
        
        logger.info(f"Dataset split: train={len(self.train_texts)}, "
                    f"val={len(self.val_texts)}, test={len(self.test_texts)}")
    
    @classmethod
    def from_json(cls, path: Path) -> 'Dataset':
        """
        Load dataset from JSON file.
        
        Expected format:
        [
            {"text": "...", "label": "positive"},
            {"text": "...", "label": "negative"},
            ...
        ]
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        logger.info(f"Loaded {len(texts)} samples from {path}")
        return cls(texts, labels)
    
    @classmethod
    def from_csv(cls, path: Path, text_col: str = 'text', 
                 label_col: str = 'label') -> 'Dataset':
        """Load dataset from CSV file."""
        import pandas as pd
        
        df = pd.read_csv(path)
        texts = df[text_col].tolist()
        labels = df[label_col].tolist()
        
        logger.info(f"Loaded {len(texts)} samples from {path}")
        return cls(texts, labels)
    
    def to_json(self, path: Path) -> None:
        """Save dataset to JSON file."""
        data = [
            {'text': text, 'label': label}
            for text, label in zip(self.texts, self.labels)
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved dataset to {path}")
    
    @classmethod
    def create_sample_dataset(cls, size: int = 1000) -> 'Dataset':
        """
        Create a sample dataset for testing.
        
        Generates synthetic Turkish e-commerce comments.
        """
        import random
        
        positive_templates = [
            "Çok güzel bir ürün, memnun kaldım",
            "Hızlı kargo, kaliteli ürün teşekkürler",
            "Mükemmel! Tam beklediğim gibi",
            "Harika bir alışveriş deneyimi",
            "Ürün çok kaliteli, kesinlikle tavsiye ederim",
            "Fiyat performans açısından süper",
            "Çok şık, bayıldım",
            "Tam zamanında geldi, çok mutluyum",
            "Satıcı çok ilgili, ürün harika",
            "Kesinlikle tekrar alırım"
        ]
        
        negative_templates = [
            "Berbat bir ürün, hiç beğenmedim",
            "Kargo çok geç geldi, ürün bozuktu",
            "Rezalet! İade edeceğim",
            "Kalitesiz, parasına değmez",
            "Fotoğraftaki gibi değil, hayal kırıklığı",
            "Çok kötü, pişman oldum",
            "Sahte ürün göndermişler",
            "İade sürecinde sorun yaşadım",
            "Hiç tavsiye etmiyorum",
            "Defolu geldi, çok sinir oldum"
        ]
        
        neutral_templates = [
            "Fena değil, idare eder",
            "Normal bir ürün",
            "Beklentimi karşıladı",
            "Standart kalite",
            "Fiyatına göre normal",
            "Orta seviye bir ürün",
            "Ne iyi ne kötü",
            "İşini görüyor",
            "Vasat, abartılacak bir şey yok",
            "Olabilir"
        ]
        
        texts = []
        labels = []
        
        for _ in range(size):
            r = random.random()
            if r < 0.4:
                texts.append(random.choice(positive_templates))
                labels.append('positive')
            elif r < 0.8:
                texts.append(random.choice(negative_templates))
                labels.append('negative')
            else:
                texts.append(random.choice(neutral_templates))
                labels.append('neutral')
        
        logger.info(f"Created sample dataset with {size} samples")
        return cls(texts, labels)


class ExperimentTracker:
    """
    Tracks experiments and model performance over time.
    
    Stores training runs, metrics, and enables comparison
    across different models and configurations.
    """
    
    def __init__(self, experiment_name: str, save_dir: Path):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save experiment data
        """
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
    
    def log_run(self, model_name: str, metrics: ModelMetrics, 
                config: Dict[str, Any], duration_seconds: float) -> None:
        """
        Log a training run.
        
        Args:
            model_name: Name of the model
            metrics: Evaluation metrics
            config: Model configuration
            duration_seconds: Training duration
        """
        run = {
            'model_name': model_name,
            'metrics': metrics.to_dict(),
            'config': config,
            'duration_seconds': duration_seconds,
            'timestamp': datetime.now().isoformat()
        }
        
        self.runs.append(run)
        logger.info(f"Logged run for {model_name}: F1={metrics.f1_score:.4f}")
    
    def get_best_run(self, metric: str = 'f1_score') -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric."""
        if not self.runs:
            return None
        
        return max(self.runs, key=lambda r: r['metrics'].get(metric, 0))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            'experiment_name': self.experiment_name,
            'total_runs': len(self.runs),
            'start_time': self.start_time.isoformat(),
            'best_run': self.get_best_run(),
            'all_runs': self.runs
        }
    
    def save(self) -> None:
        """Save experiment data to disk."""
        path = self.save_dir / f'{self.experiment_name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved experiment to {path}")
    
    def load(self) -> None:
        """Load experiment data from disk."""
        path = self.save_dir / f'{self.experiment_name}.json'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.runs = data.get('all_runs', [])
            logger.info(f"Loaded experiment from {path}")


class ModelTrainer:
    """
    Unified trainer for all ML models.
    
    Provides a consistent interface for training, evaluating,
    and comparing multiple models on the same dataset.
    """
    
    def __init__(self, dataset: Dataset, experiment_name: str = "default",
                 models_dir: Path = Path("model/weights"),
                 experiments_dir: Path = Path("model/experiments")):
        """
        Initialize trainer.
        
        Args:
            dataset: Dataset for training and evaluation
            experiment_name: Name for this experiment
            models_dir: Directory to save trained models
            experiments_dir: Directory to save experiment logs
        """
        self.dataset = dataset
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = ExperimentTracker(experiment_name, experiments_dir)
        
        # Auto-discover models
        auto_discover_models()
    
    def train_model(self, model_name: str, **kwargs) -> Tuple[BaseMLModel, ModelMetrics]:
        """
        Train a single model.
        
        Args:
            model_name: Registered model name
            **kwargs: Additional training parameters
            
        Returns:
            Tuple of (trained model, metrics)
        """
        import time
        
        if not self.dataset._is_split:
            self.dataset.split()
        
        logger.info(f"Training model: {model_name}")
        start_time = time.time()
        
        model = ModelRegistry.get(model_name)
        
        # Train
        metrics = model.train(
            self.dataset.train_texts,
            self.dataset.train_labels,
            **kwargs
        )
        
        # Evaluate on test set
        test_metrics = model.evaluate(
            self.dataset.test_texts,
            self.dataset.test_labels
        )
        
        duration = time.time() - start_time
        
        # Log and save
        self.tracker.log_run(model_name, test_metrics, model.get_config(), duration)
        ModelRegistry.update_metrics(model_name, test_metrics)
        
        # Save model
        model_path = self.models_dir / model_name
        model.save(model_path)
        
        logger.info(f"Trained {model_name} in {duration:.2f}s, F1={test_metrics.f1_score:.4f}")
        
        return model, test_metrics
    
    def train_all_models(self, model_names: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, ModelMetrics]:
        """
        Train all (or specified) registered models.
        
        Args:
            model_names: List of model names to train. If None, trains all.
            **kwargs: Additional training parameters
            
        Returns:
            Dict mapping model names to their metrics
        """
        if model_names is None:
            model_names = [m['name'] for m in ModelRegistry.list_models()]
        
        results = {}
        
        for name in model_names:
            try:
                _, metrics = self.train_model(name, **kwargs)
                results[name] = metrics
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Save experiment
        self.tracker.save()
        
        return results
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare all trained models.
        
        Returns:
            Comparison summary with rankings
        """
        comparison = ModelRegistry.get_comparison_table()
        
        if not comparison:
            logger.warning("No models have been evaluated")
            return {}
        
        # Select best
        best_name = ModelRegistry.select_best_model('f1_score')
        
        summary = {
            'models': comparison,
            'best_model': best_name,
            'best_metrics': ModelRegistry.get_metrics(best_name).to_dict() if best_name else None
        }
        
        # Print comparison table
        logger.info("\n=== Model Comparison ===")
        logger.info(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        logger.info("-" * 65)
        
        for m in comparison:
            marker = " *" if m['is_best'] else ""
            logger.info(f"{m['name']:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                        f"{m['recall']:>10.4f} {m['f1_score']:>10.4f}{marker}")
        
        logger.info(f"\n* Best model: {best_name}")
        
        return summary
    
    def get_best_model(self) -> Optional[BaseMLModel]:
        """Get the best performing model."""
        return ModelRegistry.get_best_model()
    
    def load_model(self, model_name: str) -> BaseMLModel:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model = ModelRegistry.get(model_name)
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            model.load(model_path)
            logger.info(f"Loaded model {model_name} from {model_path}")
        else:
            raise FileNotFoundError(f"No saved model found at {model_path}")
        
        return model


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument('--data', type=str, help='Path to training data (JSON or CSV)')
    parser.add_argument('--models', type=str, nargs='+', help='Models to train (default: all)')
    parser.add_argument('--experiment', type=str, default='default', help='Experiment name')
    parser.add_argument('--sample', action='store_true', help='Use sample dataset')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load or create dataset
    if args.sample or not args.data:
        dataset = Dataset.create_sample_dataset(1000)
    elif args.data.endswith('.json'):
        dataset = Dataset.from_json(Path(args.data))
    else:
        dataset = Dataset.from_csv(Path(args.data))
    
    # Split dataset
    dataset.split()
    
    # Train models
    trainer = ModelTrainer(dataset, experiment_name=args.experiment)
    trainer.train_all_models(model_names=args.models)
    
    # Compare results
    trainer.compare_models()


if __name__ == '__main__':
    main()
