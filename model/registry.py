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

from typing import Dict, Type, List, Optional, Callable, Any
import importlib
import pkgutil
import logging

from model.base import BaseMLModel, ModelMetrics


logger = logging.getLogger(__name__)


# Very small and simple registry implemented with plain dicts.
# This keeps the API similar but uses straightforward code.
_models: Dict[str, Type[BaseMLModel]] = {}
_instances: Dict[str, BaseMLModel] = {}
_metrics: Dict[str, ModelMetrics] = {}
_best_model: Optional[str] = None


class ModelRegistry:
    """Tiny compatibility wrapper around simple module-level registry."""

    @staticmethod
    def register(name: str, description: str = "") -> Callable:
        """Return a decorator that registers a model class under `name`."""
        def decorator(cls):
            _models[name] = cls
            cls._registry_name = name
            cls._registry_description = description
            return cls
        return decorator

    @staticmethod
    def get(name: str, **kwargs) -> BaseMLModel:
        if name not in _models:
            available = ', '.join(_models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return _models[name](**kwargs)

    @staticmethod
    def get_cached(name: str, **kwargs) -> BaseMLModel:
        if name not in _instances:
            _instances[name] = ModelRegistry.get(name, **kwargs)
        return _instances[name]

    @staticmethod
    def list_models() -> List[Dict[str, str]]:
        return [{'name': n, 'class': c.__name__, 'description': getattr(c, '_registry_description', '')} for n, c in _models.items()]

    @staticmethod
    def is_registered(name: str) -> bool:
        return name in _models

    @staticmethod
    def unregister(name: str) -> None:
        _models.pop(name, None)
        _instances.pop(name, None)
        _metrics.pop(name, None)

    @staticmethod
    def clear() -> None:
        _models.clear()
        _instances.clear()
        _metrics.clear()
        global _best_model
        _best_model = None

    @staticmethod
    def update_metrics(name: str, metrics: ModelMetrics) -> None:
        _metrics[name] = metrics

    @staticmethod
    def get_metrics(name: str) -> Optional[ModelMetrics]:
        return _metrics.get(name)

    @staticmethod
    def get_all_metrics() -> Dict[str, ModelMetrics]:
        return dict(_metrics)

    @staticmethod
    def select_best_model(metric: str = 'f1_score') -> Optional[str]:
        if not _metrics:
            return None
        best = None
        best_val = -1.0
        for n, m in _metrics.items():
            val = getattr(m, metric, 0.0)
            if val > best_val:
                best_val = val
                best = n
        global _best_model
        _best_model = best
        return best

    @staticmethod
    def get_best_model() -> Optional[BaseMLModel]:
        global _best_model
        if _best_model is None:
            ModelRegistry.select_best_model()
        if _best_model is None:
            return None
        return ModelRegistry.get_cached(_best_model)

    @staticmethod
    def compare_models(texts: List[str], labels: List[str]) -> Dict[str, ModelMetrics]:
        results: Dict[str, ModelMetrics] = {}
        for name in list(_models.keys()):
            try:
                model = ModelRegistry.get_cached(name)
                if getattr(model, 'is_trained', False):
                    metrics = model.evaluate(texts, labels)
                    ModelRegistry.update_metrics(name, metrics)
                    results[name] = metrics
            except Exception:
                pass
        return results

    @staticmethod
    def get_comparison_table() -> List[Dict[str, Any]]:
        table = []
        for name, metrics in _metrics.items():
            table.append({
                'name': name,
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'is_best': name == _best_model
            })
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
                # logger.info(f"Auto-discovered model module: {modname}")
            except Exception:
                # Keep it simple: ignore modules that fail to import (training scripts, data ops, etc.)
                pass
