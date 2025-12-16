# REMOVED: program.utils.i18n shim deleted (cleared)

# Import and use `utils.i18n` directly.
    
    def __new__(cls) -> 'I18n':
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize i18n with default or configured language."""
        # Check environment variable
        env_lang = os.environ.get("COMMENTCOURT_LANGUAGE", "").lower()
        if env_lang in self._translations:
            self._current_language = env_lang
    
    @property
    def current_language(self) -> str:
        """Get current language code."""
        return self._current_language
    
    @property
    def languages(self) -> Dict[str, Dict[str, str]]:
        """Get available languages."""
        return Translations.LANGUAGES
    
    def set_language(self, lang_code: str) -> bool:
        """
        Set the current language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'tr')
            
        Returns:
            True if language was set, False if not supported
        """
        if lang_code.lower() in self._translations:
            self._current_language = lang_code.lower()
            return True
        return False
    
    def t(self, key: str, *args, **kwargs) -> str:
        """
        Translate a key to the current language.
        
        Args:
            key: Translation key (e.g., 'nav.dashboard')
            *args: Positional arguments for string formatting
            **kwargs: Keyword arguments for string formatting
            
        Returns:
            Translated string, or key if not found
        """
        # Try current language
        translations = self._translations.get(self._current_language, {})
        text = translations.get(key)
        
        # Fallback to default language
        if text is None:
            translations = self._translations.get(self._default_language, {})
            text = translations.get(key, key)
        
        # Apply formatting if args/kwargs provided
        if args or kwargs:
            try:
                if args:
                    text = text.format(*args)
                elif kwargs:
                    text = text.format(**kwargs)
            except (KeyError, IndexError):
                pass
        
        return text
    
    def get_all_translations(self) -> Dict[str, str]:
        """Get all translations for current language."""
        return self._translations.get(
            self._current_language,
            self._translations.get(self._default_language, {})
        )
    
    def add_translation(self, lang_code: str, key: str, value: str) -> None:
        """
        Add or update a translation.
        
        Args:
            lang_code: Language code
            key: Translation key
            value: Translated string
        """
        if lang_code not in self._translations:
            self._translations[lang_code] = {}
        self._translations[lang_code][key] = value


# Global instance
i18n = I18n()

# Convenience function
def t(key: str, *args, **kwargs) -> str:
    """Shortcut for i18n.t()"""
    return i18n.t(key, *args, **kwargs)


def get_language() -> str:
    """Get current language code."""
    return i18n.current_language


def set_language(lang_code: str) -> bool:
    """Set current language."""
    return i18n.set_language(lang_code)


def get_languages() -> Dict[str, Dict[str, str]]:
    """Get available languages."""
    return i18n.languages
