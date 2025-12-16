"""
Configuration Management for CommentCourt (moved to `utils`).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
import yaml


logger = logging.getLogger(__name__)

"""Small, simple config helpers.

This module intentionally keeps configuration as a plain dictionary
for simplicity and readability. It provides a few helper functions
used throughout the project: `load_config`, `get_config`,
`get_config_value`, and `set_config_value`.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# Default config file path (a file or directory named `config`)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config"


# A simple default configuration dictionary
DEFAULTS: Dict[str, Any] = {
    'app_name': 'CommentCourt',
    'version': '1.0.0',
    'environment': 'development',
    'database': {
        'path': 'database/db.sqlite3'
    },
    'model': {
        'models_dir': 'model/weights',
        'selection_metric': 'f1_score',
        'default_models': ['rule_based']
    },
    'pipeline': {
        'raw_data_dir': 'database/raw',
        'processed_data_dir': 'database/processed',
        'batch_size': 100,
        'max_comments_per_run': 10000
    },
    'scoring': {
        'scale': 10
    },
    'gui': {
        'host': '127.0.0.1',
        'port': 5000,
        'debug': False,
        'secret_key': 'change-this-in-production'
    },
    'logging': {
        'level': 'INFO'
    }
}


# Global configuration object (plain dict)
CONFIG: Dict[str, Any] = {}


def _deep_update(target: Dict[str, Any], values: Dict[str, Any]) -> None:
    """Recursively update target dict with values dict."""
    for k, v in values.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_update(target[k], v)
        else:
            target[k] = v


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML (if present) and environment.

    The function mutates and returns the global `CONFIG` dict.
    """
    global CONFIG

    # Start from defaults
    CONFIG = {}
    _deep_update(CONFIG, DEFAULTS)

    # Load YAML if provided
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if config_path.exists():
        try:
            data = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
            if isinstance(data, dict):
                _deep_update(CONFIG, data)
        except Exception:
            # Keep defaults on error (explicit, easy to understand)
            pass

    # Simple environment overrides
    if os.getenv('COMMENTCOURT_DB_PATH'):
        CONFIG.setdefault('database', {})['path'] = os.getenv('COMMENTCOURT_DB_PATH')
    if os.getenv('COMMENTCOURT_HOST'):
        CONFIG.setdefault('gui', {})['host'] = os.getenv('COMMENTCOURT_HOST')
    if os.getenv('COMMENTCOURT_PORT'):
        try:
            CONFIG.setdefault('gui', {})['port'] = int(os.getenv('COMMENTCOURT_PORT'))
        except Exception:
            pass
    if os.getenv('COMMENTCOURT_DEBUG'):
        CONFIG.setdefault('gui', {})['debug'] = os.getenv('COMMENTCOURT_DEBUG').lower() == 'true'
    if os.getenv('COMMENTCOURT_SECRET_KEY'):
        CONFIG.setdefault('gui', {})['secret_key'] = os.getenv('COMMENTCOURT_SECRET_KEY')
    if os.getenv('COMMENTCOURT_LOG_LEVEL'):
        CONFIG.setdefault('logging', {})['level'] = os.getenv('COMMENTCOURT_LOG_LEVEL')

    return CONFIG


def get_config() -> Dict[str, Any]:
    """Return the global configuration, loading defaults if needed."""
    global CONFIG
    if not CONFIG:
        load_config()
    return CONFIG


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation, e.g. 'gui.port'."""
    parts = key.split('.')
    node = get_config()
    for p in parts:
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return default
    return node


def set_config_value(key: str, value: Any) -> None:
    """Set a configuration value using dot notation."""
    parts = key.split('.')
    node = get_config()
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value
