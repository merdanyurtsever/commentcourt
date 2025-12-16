# REMOVED: program.utils.config shim deleted (cleared)

# Import from `utils.config` directly.




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
