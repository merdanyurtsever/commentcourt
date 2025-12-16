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


# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "database/db.sqlite3"
    echo: bool = False
    pool_size: int = 5


@dataclass
class ModelConfig:
    """ML model configuration."""
    models_dir: str = "model/weights"
    experiments_dir: str = "model/experiments"
    default_models: List[str] = field(default_factory=lambda: [
        "rule_based",
        "logistic_regression",
        "lstm",
        "bert_turkish"
    ])
    selection_metric: str = "f1_score"
    auto_train: bool = True


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    raw_data_dir: str = "database/raw"
    processed_data_dir: str = "database/processed"
    batch_size: int = 100
    max_comments_per_run: int = 10000


@dataclass
class ScoringConfig:
    """Scoring configuration."""
    scale: int = 10
    positive_weight: float = 1.0
    negative_weight: float = -0.5
    neutral_weight: float = 0.1
    min_comments_for_ranking: int = 5
    use_confidence_weighting: bool = True
    use_temporal_weighting: bool = True


@dataclass
class GUIConfig:
    """GUI configuration."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    secret_key: str = "change-this-in-production"
    items_per_page: int = 20
    theme: str = "light"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/app.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "CommentCourt"
    version: str = "1.0.0"
    environment: str = "development"
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Influencer config (loaded from separate file)
    influencers: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConfigManager:
    """
    Manages application configuration.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize if not already done."""
        if self._config is None:
            self._config = AppConfig()
    
    def load(self, config_path: Optional[Path] = None) -> AppConfig:
        """
        Load configuration from YAML file.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        
        config_path = Path(config_path)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_content = yaml.safe_load(f) or {}
                
                self._apply_yaml_config(yaml_content)
                logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        else:
            logger.info(f"Config file not found at {config_path}, using defaults")
        
        # Apply environment overrides
        self._apply_env_overrides()
        
        return self._config
    
    def _apply_yaml_config(self, yaml_data: Dict[str, Any]) -> None:
        """Apply YAML configuration to AppConfig."""
        # Database
        if 'database' in yaml_data:
            db_config = yaml_data['database']
            self._config.database.path = db_config.get('path', self._config.database.path)
            self._config.database.echo = db_config.get('echo', self._config.database.echo)
        
        # Model
        if 'model' in yaml_data:
            model_config = yaml_data['model']
            self._config.model.models_dir = model_config.get('models_dir', self._config.model.models_dir)
            self._config.model.selection_metric = model_config.get('selection_metric', self._config.model.selection_metric)
            self._config.model.default_models = model_config.get('default_models', self._config.model.default_models)
        
        # Pipeline
        if 'pipeline' in yaml_data:
            pipeline_config = yaml_data['pipeline']
            self._config.pipeline.batch_size = pipeline_config.get('batch_size', self._config.pipeline.batch_size)
            self._config.pipeline.max_comments_per_run = pipeline_config.get('max_comments_per_run', self._config.pipeline.max_comments_per_run)
        
        # Scoring
        if 'scoring' in yaml_data:
            scoring_config = yaml_data['scoring']
            self._config.scoring.scale = scoring_config.get('scale', self._config.scoring.scale)
            self._config.scoring.positive_weight = scoring_config.get('positive_weight', self._config.scoring.positive_weight)
            self._config.scoring.negative_weight = scoring_config.get('negative_weight', self._config.scoring.negative_weight)
        
        # GUI
        if 'gui' in yaml_data:
            gui_config = yaml_data['gui']
            self._config.gui.host = gui_config.get('host', self._config.gui.host)
            self._config.gui.port = gui_config.get('port', self._config.gui.port)
            self._config.gui.debug = gui_config.get('debug', self._config.gui.debug)
            self._config.gui.secret_key = gui_config.get('secret_key', self._config.gui.secret_key)
        
        # Logging
        if 'logging' in yaml_data:
            log_config = yaml_data['logging']
            self._config.logging.level = log_config.get('level', self._config.logging.level)
            self._config.logging.file = log_config.get('file', self._config.logging.file)
        
        # Influencers
        if 'influencers' in yaml_data:
            self._config.influencers = yaml_data['influencers']
        
        # App-level
        self._config.app_name = yaml_data.get('app_name', self._config.app_name)
        self._config.version = yaml_data.get('version', self._config.version)
        self._config.environment = yaml_data.get('environment', self._config.environment)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Database
        if os.getenv('COMMENTCOURT_DB_PATH'):
            self._config.database.path = os.getenv('COMMENTCOURT_DB_PATH')
        
        # GUI
        if os.getenv('COMMENTCOURT_HOST'):
            self._config.gui.host = os.getenv('COMMENTCOURT_HOST')
        if os.getenv('COMMENTCOURT_PORT'):
            self._config.gui.port = int(os.getenv('COMMENTCOURT_PORT'))
        if os.getenv('COMMENTCOURT_DEBUG'):
            self._config.gui.debug = os.getenv('COMMENTCOURT_DEBUG').lower() == 'true'
        if os.getenv('COMMENTCOURT_SECRET_KEY'):
            self._config.gui.secret_key = os.getenv('COMMENTCOURT_SECRET_KEY')
        
        # Logging
        if os.getenv('COMMENTCOURT_LOG_LEVEL'):
            self._config.logging.level = os.getenv('COMMENTCOURT_LOG_LEVEL')
        
        # Environment
        if os.getenv('COMMENTCOURT_ENV'):
            self._config.environment = os.getenv('COMMENTCOURT_ENV')
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        """
        parts = key.split('.')
        value = self._config
        
        try:
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict):
                    value = value.get(part)
                else:
                    return default
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        """
        parts = key.split('.')
        obj = self._config
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.warning(f"Invalid config key: {key}")
                return
        
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
        else:
            logger.warning(f"Invalid config key: {key}")
    
    def save(self, config_path: Optional[Path] = None) -> None:
        """
        Save current configuration to YAML file.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        
        config_path = Path(config_path)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config.to_dict(), f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved configuration to {config_path}")
    
    def reload(self, config_path: Optional[Path] = None) -> AppConfig:
        """Reload configuration from file."""
        self._config = AppConfig()
        return self.load(config_path)


# Convenience functions

_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """Get the global configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load()
    return _config_manager.config


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from file."""
    global _config_manager
    _config_manager = ConfigManager()
    return _config_manager.load(config_path)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value by key."""
    global _config_manager
    if _config_manager is None:
        get_config()
    return _config_manager.get(key, default)


def set_config_value(key: str, value: Any) -> None:
    """Set a configuration value."""
    global _config_manager
    if _config_manager is None:
        get_config()
    _config_manager.set(key, value)
