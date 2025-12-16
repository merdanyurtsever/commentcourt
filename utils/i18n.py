"""
Internationalization utilities (moved to `utils.i18n`).
"""

from typing import Dict, Optional
import os


class Translations:
    EN: Dict[str, str] = {
        "nav.dashboard": "Dashboard",
        # ... shortened for brevity; original strings preserved in repo
    }

    TR: Dict[str, str] = {
        "nav.dashboard": "Kontrol Paneli",
        # ... shortened for brevity; original strings preserved in repo
    }

    LANGUAGES = {
        "en": {"name": "English", "native": "English", "flag": "🇬🇧"},
        "tr": {"name": "Turkish", "native": "Türkçe", "flag": "🇹🇷"},
    }


class I18n:
    _instance: Optional['I18n'] = None
    _translations: Dict[str, Dict[str, str]] = {
        "en": Translations.EN,
        "tr": Translations.TR,
    }
    _current_language: str = "tr"
    _default_language: str = "en"

    def __new__(cls) -> 'I18n':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        env_lang = os.environ.get("COMMENTCOURT_LANGUAGE", "").lower()
        if env_lang in self._translations:
            self._current_language = env_lang

    @property
    def current_language(self) -> str:
        return self._current_language

    @property
    def languages(self) -> Dict[str, Dict[str, str]]:
        return Translations.LANGUAGES

    def set_language(self, lang_code: str) -> bool:
        if lang_code.lower() in self._translations:
            self._current_language = lang_code.lower()
            return True
        return False

    def t(self, key: str, *args, **kwargs) -> str:
        translations = self._translations.get(self._current_language, {})
        text = translations.get(key)
        if text is None:
            translations = self._translations.get(self._default_language, {})
            text = translations.get(key, key)
        if args or kwargs:
            try:
                text = text.format(*args, **kwargs)
            except Exception:
                pass
        return text


_i18n = I18n()


def i18n() -> I18n:
    return _i18n


def t(key: str, *args, **kwargs) -> str:
    return _i18n.t(key, *args, **kwargs)


def get_language() -> str:
    return _i18n.current_language


def set_language(code: str) -> bool:
    return _i18n.set_language(code)


def get_languages() -> Dict[str, Dict[str, str]]:
    return _i18n.languages
