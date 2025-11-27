"""
Model Registry for CommentCourt ML Framework.

This module provides a centralized registry for ML models, enabling:
- Automatic model discovery and registration via decorators
- Model instantiation by name
- Listing all available models
- Model comparison and selection
"""

from typing import Dict, Type, List, Optional, Callable, Any
from pathlib import Path
import logging

from model.base import BaseMLModel, ModelMetrics


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for ML models.
    
    Provides decorator-based registration and factory methods for
    instantiating models by name. Supports model comparison and
    automatic selection of the best-performing model.
    
    Usage:
        # Register a model
        @ModelRegistry.register("my_model")
        class MyModel(BaseMLModel):
            ...
        
        # Get a model instance
        model = ModelRegistry.get("my_model")
        
        # List all models
        models = ModelRegistry.list_models()
    """
    
    _models: Dict[str, Type[BaseMLModel]] = {}
    _instances: Dict[str, BaseMLModel] = {}
    _metrics: Dict[str, ModelMetrics] = {}
    _best_model: Optional[str] = None
    
    @classmethod
    def register(cls, name: str, description: str = "") -> Callable:
        """
        Decorator to register a model class.
        
        Args:
            name: Unique identifier for the model
            description: Human-readable description
            
        Returns:
            Decorator function
            
        Example:
            @ModelRegistry.register("bert_turkish", "BERT model for Turkish text")
            class BertTurkishModel(BaseMLModel):
                ...
        """
        def decorator(model_class: Type[BaseMLModel]) -> Type[BaseMLModel]:
            if name in cls._models:
                logger.warning(f"Model '{name}' is already registered. Overwriting.")
            
            cls._models[name] = model_class
            model_class._registry_name = name
            model_class._registry_description = description
            
            logger.info(f"Registered model: {name} ({model_class.__name__})")
            return model_class
        
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs) -> BaseMLModel:
        """
        Get or create a model instance by name.
        
        Args:
            name: Registered model name
            **kwargs: Arguments to pass to model constructor
            
        Returns:
            Model instance
            
        Raises:
            KeyError: If model name is not registered
        """
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        
        # Create new instance
        model_class = cls._models[name]
        instance = model_class(**kwargs)
        
        return instance
    
    @classmethod
    def get_cached(cls, name: str, **kwargs) -> BaseMLModel:
        """
        Get a cached model instance or create one.
        
        Uses singleton pattern to avoid loading the same model multiple times.
        
        Args:
            name: Registered model name
            **kwargs: Arguments to pass to model constructor (only used on first call)
            
        Returns:
            Cached model instance
        """
        if name not in cls._instances:
            cls._instances[name] = cls.get(name, **kwargs)
        
        return cls._instances[name]
    
    @classmethod
    def list_models(cls) -> List[Dict[str, str]]:
        """
        List all registered models.
        
        Returns:
            List of dicts with model name, class, and description
        """
        return [
            {
                'name': name,
                'class': model_class.__name__,
                'description': getattr(model_class, '_registry_description', '')
            }
            for name, model_class in cls._models.items()
        ]
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseMLModel]:
        """
        Get the model class (not instance) by name.
        
        Args:
            name: Registered model name
            
        Returns:
            Model class
        """
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not found.")
        return cls._models[name]
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model name is registered."""
        return name in cls._models
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a model from the registry.
        
        Args:
            name: Model name to remove
        """
        if name in cls._models:
            del cls._models[name]
            logger.info(f"Unregistered model: {name}")
        
        if name in cls._instances:
            del cls._instances[name]
        
        if name in cls._metrics:
            del cls._metrics[name]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered models and cached instances."""
        cls._models.clear()
        cls._instances.clear()
        cls._metrics.clear()
        cls._best_model = None
        logger.info("Cleared model registry")
    
    @classmethod
    def update_metrics(cls, name: str, metrics: ModelMetrics) -> None:
        """
        Update stored metrics for a model.
        
        Args:
            name: Model name
            metrics: Evaluation metrics
        """
        cls._metrics[name] = metrics
        logger.info(f"Updated metrics for {name}: F1={metrics.f1_score:.4f}")
    
    @classmethod
    def get_metrics(cls, name: str) -> Optional[ModelMetrics]:
        """Get stored metrics for a model."""
        return cls._metrics.get(name)
    
    @classmethod
    def get_all_metrics(cls) -> Dict[str, ModelMetrics]:
        """Get metrics for all evaluated models."""
        return cls._metrics.copy()
    
    @classmethod
    def select_best_model(cls, metric: str = "f1_score") -> Optional[str]:
        """
        Select the best model based on a metric.
        
        Args:
            metric: Metric to use for comparison ('accuracy', 'f1_score', 'precision', 'recall')
            
        Returns:
            Name of the best-performing model, or None if no metrics available
        """
        if not cls._metrics:
            logger.warning("No model metrics available for comparison")
            return None
        
        best_name = None
        best_value = -1.0
        
        for name, metrics in cls._metrics.items():
            value = getattr(metrics, metric, 0.0)
            if value > best_value:
                best_value = value
                best_name = name
        
        cls._best_model = best_name
        logger.info(f"Selected best model: {best_name} ({metric}={best_value:.4f})")
        
        return best_name
    
    @classmethod
    def get_best_model(cls) -> Optional[BaseMLModel]:
        """
        Get the best model instance.
        
        Returns:
            Best model instance, or None if not determined
        """
        if cls._best_model is None:
            cls.select_best_model()
        
        if cls._best_model is None:
            return None
        
        return cls.get_cached(cls._best_model)
    
    @classmethod
    def compare_models(cls, texts: List[str], labels: List[str]) -> Dict[str, ModelMetrics]:
        """
        Evaluate all registered models on the same dataset.
        
        Args:
            texts: Test text samples
            labels: Ground truth labels
            
        Returns:
            Dict mapping model names to their metrics
        """
        results = {}
        
        for name in cls._models.keys():
            try:
                model = cls.get_cached(name)
                if model.is_trained:
                    metrics = model.evaluate(texts, labels)
                    cls.update_metrics(name, metrics)
                    results[name] = metrics
                    logger.info(f"Evaluated {name}: F1={metrics.f1_score:.4f}")
                else:
                    logger.warning(f"Model {name} is not trained, skipping evaluation")
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        return results
    
    @classmethod
    def get_comparison_table(cls) -> List[Dict[str, Any]]:
        """
        Get a comparison table of all evaluated models.
        
        Returns:
            List of dicts with model names and metrics
        """
        table = []
        for name, metrics in cls._metrics.items():
            table.append({
                'name': name,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'is_best': name == cls._best_model
            })
        
        # Sort by F1 score descending
        table.sort(key=lambda x: x['f1_score'], reverse=True)
        
        return table


def auto_discover_models(package_path: str = "model") -> None:
    """
    Automatically discover and import model modules.
    
    This function imports all Python files in the model package,
    triggering the @register decorators.
    
    Args:
        package_path: Path to the model package
    """
    import importlib
    import pkgutil
    
    package = importlib.import_module(package_path)
    
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if modname not in ('base', 'registry'):
            try:
                importlib.import_module(f"{package_path}.{modname}")
                logger.info(f"Auto-discovered model module: {modname}")
            except ImportError as e:
                logger.warning(f"Could not import {modname}: {e}")
