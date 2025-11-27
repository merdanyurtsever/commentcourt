"""
CommentCourt - Internationalization (i18n) Module
Provides multi-language support for the application.
"""

from typing import Dict, Optional
import os


class Translations:
    """Translation strings for supported languages."""
    
    # English translations
    EN: Dict[str, str] = {
        # Navigation
        "nav.dashboard": "Dashboard",
        "nav.influencers": "Influencers",
        "nav.analysis": "Analysis",
        "nav.comments": "Comments",
        "nav.language": "Language",
        
        # Dashboard
        "dashboard.title": "Dashboard",
        "dashboard.welcome": "Welcome to CommentCourt",
        "dashboard.subtitle": "Turkish E-Commerce Influencer Analysis Platform",
        "dashboard.total_influencers": "Total Influencers",
        "dashboard.total_comments": "Total Comments",
        "dashboard.avg_score": "Average Score",
        "dashboard.models_active": "Active Models",
        "dashboard.sentiment_distribution": "Sentiment Distribution",
        "dashboard.positive": "Positive",
        "dashboard.negative": "Negative",
        "dashboard.neutral": "Neutral",
        "dashboard.top_influencers": "Top Influencers",
        "dashboard.recent_comments": "Recent Comments",
        "dashboard.view_all": "View All",
        "dashboard.quick_stats": "Quick Statistics",
        "dashboard.analysis_overview": "Analysis Overview",
        
        # Influencers
        "influencers.title": "Influencers",
        "influencers.subtitle": "View and manage influencer rankings",
        "influencers.search": "Search influencers...",
        "influencers.filter_platform": "Filter by Platform",
        "influencers.all_platforms": "All Platforms",
        "influencers.sort_by": "Sort By",
        "influencers.sort_score": "Score",
        "influencers.sort_name": "Name",
        "influencers.sort_comments": "Comments",
        "influencers.sort_date": "Date Added",
        "influencers.rank": "Rank",
        "influencers.name": "Name",
        "influencers.platform": "Platform",
        "influencers.score": "Score",
        "influencers.comments": "Comments",
        "influencers.sentiment": "Sentiment",
        "influencers.actions": "Actions",
        "influencers.view_details": "View Details",
        "influencers.no_results": "No influencers found",
        "influencers.loading": "Loading influencers...",
        
        # Influencer Detail
        "influencer.profile": "Profile",
        "influencer.statistics": "Statistics",
        "influencer.sentiment_breakdown": "Sentiment Breakdown",
        "influencer.comment_history": "Comment History",
        "influencer.model_results": "Model Results",
        "influencer.score_history": "Score History",
        "influencer.total_comments": "Total Comments",
        "influencer.avg_sentiment": "Average Sentiment",
        "influencer.consistency": "Consistency",
        "influencer.last_updated": "Last Updated",
        "influencer.back_to_list": "Back to List",
        
        # Analysis
        "analysis.title": "Analysis",
        "analysis.subtitle": "Run sentiment analysis on comments",
        "analysis.run_analysis": "Run Analysis",
        "analysis.running": "Analysis Running...",
        "analysis.complete": "Analysis Complete",
        "analysis.failed": "Analysis Failed",
        "analysis.select_model": "Select Model",
        "analysis.all_models": "All Models",
        "analysis.best_model": "Best Model",
        "analysis.model_comparison": "Model Comparison",
        "analysis.accuracy": "Accuracy",
        "analysis.precision": "Precision",
        "analysis.recall": "Recall",
        "analysis.f1_score": "F1 Score",
        "analysis.processing_time": "Processing Time",
        "analysis.results": "Results",
        "analysis.no_data": "No analysis data available",
        "analysis.start_new": "Start New Analysis",
        
        # Comments
        "comments.title": "Comments",
        "comments.subtitle": "Browse and filter product comments",
        "comments.search": "Search comments...",
        "comments.filter_sentiment": "Filter by Sentiment",
        "comments.all_sentiments": "All Sentiments",
        "comments.filter_influencer": "Filter by Influencer",
        "comments.all_influencers": "All Influencers",
        "comments.date_range": "Date Range",
        "comments.text": "Comment",
        "comments.sentiment": "Sentiment",
        "comments.confidence": "Confidence",
        "comments.date": "Date",
        "comments.influencer": "Influencer",
        "comments.product": "Product",
        "comments.no_results": "No comments found",
        "comments.loading": "Loading comments...",
        
        # Common
        "common.loading": "Loading...",
        "common.error": "An error occurred",
        "common.retry": "Retry",
        "common.cancel": "Cancel",
        "common.save": "Save",
        "common.delete": "Delete",
        "common.edit": "Edit",
        "common.confirm": "Confirm",
        "common.yes": "Yes",
        "common.no": "No",
        "common.search": "Search",
        "common.filter": "Filter",
        "common.reset": "Reset",
        "common.export": "Export",
        "common.import": "Import",
        "common.refresh": "Refresh",
        "common.settings": "Settings",
        "common.help": "Help",
        "common.about": "About",
        "common.page": "Page",
        "common.of": "of",
        "common.showing": "Showing",
        "common.to": "to",
        "common.entries": "entries",
        "common.previous": "Previous",
        "common.next": "Next",
        "common.first": "First",
        "common.last": "Last",
        
        # Errors
        "error.404.title": "Page Not Found",
        "error.404.message": "The page you're looking for doesn't exist.",
        "error.500.title": "Server Error",
        "error.500.message": "Something went wrong on our end.",
        "error.back_home": "Back to Home",
        
        # Footer
        "footer.copyright": "CommentCourt - Turkish E-Commerce Analysis",
        "footer.version": "Version",
        
        # Sentiment labels
        "sentiment.positive": "Positive",
        "sentiment.negative": "Negative",
        "sentiment.neutral": "Neutral",
        
        # Time
        "time.just_now": "Just now",
        "time.minutes_ago": "{} minutes ago",
        "time.hours_ago": "{} hours ago",
        "time.days_ago": "{} days ago",
        "time.weeks_ago": "{} weeks ago",
    }
    
    # Turkish translations
    TR: Dict[str, str] = {
        # Navigation
        "nav.dashboard": "Kontrol Paneli",
        "nav.influencers": "Influencer'lar",
        "nav.analysis": "Analiz",
        "nav.comments": "Yorumlar",
        "nav.language": "Dil",
        
        # Dashboard
        "dashboard.title": "Kontrol Paneli",
        "dashboard.welcome": "CommentCourt'a Hoş Geldiniz",
        "dashboard.subtitle": "Türk E-Ticaret Influencer Analiz Platformu",
        "dashboard.total_influencers": "Toplam Influencer",
        "dashboard.total_comments": "Toplam Yorum",
        "dashboard.avg_score": "Ortalama Puan",
        "dashboard.models_active": "Aktif Model",
        "dashboard.sentiment_distribution": "Duygu Dağılımı",
        "dashboard.positive": "Olumlu",
        "dashboard.negative": "Olumsuz",
        "dashboard.neutral": "Nötr",
        "dashboard.top_influencers": "En İyi Influencer'lar",
        "dashboard.recent_comments": "Son Yorumlar",
        "dashboard.view_all": "Tümünü Gör",
        "dashboard.quick_stats": "Hızlı İstatistikler",
        "dashboard.analysis_overview": "Analiz Özeti",
        
        # Influencers
        "influencers.title": "Influencer'lar",
        "influencers.subtitle": "Influencer sıralamalarını görüntüle ve yönet",
        "influencers.search": "Influencer ara...",
        "influencers.filter_platform": "Platforma Göre Filtrele",
        "influencers.all_platforms": "Tüm Platformlar",
        "influencers.sort_by": "Sırala",
        "influencers.sort_score": "Puan",
        "influencers.sort_name": "İsim",
        "influencers.sort_comments": "Yorumlar",
        "influencers.sort_date": "Eklenme Tarihi",
        "influencers.rank": "Sıra",
        "influencers.name": "İsim",
        "influencers.platform": "Platform",
        "influencers.score": "Puan",
        "influencers.comments": "Yorumlar",
        "influencers.sentiment": "Duygu",
        "influencers.actions": "İşlemler",
        "influencers.view_details": "Detayları Gör",
        "influencers.no_results": "Influencer bulunamadı",
        "influencers.loading": "Influencer'lar yükleniyor...",
        
        # Influencer Detail
        "influencer.profile": "Profil",
        "influencer.statistics": "İstatistikler",
        "influencer.sentiment_breakdown": "Duygu Dağılımı",
        "influencer.comment_history": "Yorum Geçmişi",
        "influencer.model_results": "Model Sonuçları",
        "influencer.score_history": "Puan Geçmişi",
        "influencer.total_comments": "Toplam Yorum",
        "influencer.avg_sentiment": "Ortalama Duygu",
        "influencer.consistency": "Tutarlılık",
        "influencer.last_updated": "Son Güncelleme",
        "influencer.back_to_list": "Listeye Dön",
        
        # Analysis
        "analysis.title": "Analiz",
        "analysis.subtitle": "Yorumlar üzerinde duygu analizi çalıştır",
        "analysis.run_analysis": "Analizi Başlat",
        "analysis.running": "Analiz Çalışıyor...",
        "analysis.complete": "Analiz Tamamlandı",
        "analysis.failed": "Analiz Başarısız",
        "analysis.select_model": "Model Seç",
        "analysis.all_models": "Tüm Modeller",
        "analysis.best_model": "En İyi Model",
        "analysis.model_comparison": "Model Karşılaştırması",
        "analysis.accuracy": "Doğruluk",
        "analysis.precision": "Kesinlik",
        "analysis.recall": "Duyarlılık",
        "analysis.f1_score": "F1 Skoru",
        "analysis.processing_time": "İşlem Süresi",
        "analysis.results": "Sonuçlar",
        "analysis.no_data": "Analiz verisi bulunamadı",
        "analysis.start_new": "Yeni Analiz Başlat",
        
        # Comments
        "comments.title": "Yorumlar",
        "comments.subtitle": "Ürün yorumlarını incele ve filtrele",
        "comments.search": "Yorum ara...",
        "comments.filter_sentiment": "Duyguya Göre Filtrele",
        "comments.all_sentiments": "Tüm Duygular",
        "comments.filter_influencer": "Influencer'a Göre Filtrele",
        "comments.all_influencers": "Tüm Influencer'lar",
        "comments.date_range": "Tarih Aralığı",
        "comments.text": "Yorum",
        "comments.sentiment": "Duygu",
        "comments.confidence": "Güven",
        "comments.date": "Tarih",
        "comments.influencer": "Influencer",
        "comments.product": "Ürün",
        "comments.no_results": "Yorum bulunamadı",
        "comments.loading": "Yorumlar yükleniyor...",
        
        # Common
        "common.loading": "Yükleniyor...",
        "common.error": "Bir hata oluştu",
        "common.retry": "Tekrar Dene",
        "common.cancel": "İptal",
        "common.save": "Kaydet",
        "common.delete": "Sil",
        "common.edit": "Düzenle",
        "common.confirm": "Onayla",
        "common.yes": "Evet",
        "common.no": "Hayır",
        "common.search": "Ara",
        "common.filter": "Filtrele",
        "common.reset": "Sıfırla",
        "common.export": "Dışa Aktar",
        "common.import": "İçe Aktar",
        "common.refresh": "Yenile",
        "common.settings": "Ayarlar",
        "common.help": "Yardım",
        "common.about": "Hakkında",
        "common.page": "Sayfa",
        "common.of": "/",
        "common.showing": "Gösterilen",
        "common.to": "-",
        "common.entries": "kayıt",
        "common.previous": "Önceki",
        "common.next": "Sonraki",
        "common.first": "İlk",
        "common.last": "Son",
        
        # Errors
        "error.404.title": "Sayfa Bulunamadı",
        "error.404.message": "Aradığınız sayfa mevcut değil.",
        "error.500.title": "Sunucu Hatası",
        "error.500.message": "Bir şeyler ters gitti.",
        "error.back_home": "Ana Sayfaya Dön",
        
        # Footer
        "footer.copyright": "CommentCourt - Türk E-Ticaret Analizi",
        "footer.version": "Sürüm",
        
        # Sentiment labels
        "sentiment.positive": "Olumlu",
        "sentiment.negative": "Olumsuz",
        "sentiment.neutral": "Nötr",
        
        # Time
        "time.just_now": "Az önce",
        "time.minutes_ago": "{} dakika önce",
        "time.hours_ago": "{} saat önce",
        "time.days_ago": "{} gün önce",
        "time.weeks_ago": "{} hafta önce",
    }
    
    # Language metadata
    LANGUAGES = {
        "en": {"name": "English", "native": "English", "flag": "🇬🇧"},
        "tr": {"name": "Turkish", "native": "Türkçe", "flag": "🇹🇷"},
    }


class I18n:
    """Internationalization handler for the application."""
    
    _instance: Optional['I18n'] = None
    _translations: Dict[str, Dict[str, str]] = {
        "en": Translations.EN,
        "tr": Translations.TR,
    }
    _current_language: str = "tr"  # Default to Turkish
    _default_language: str = "en"
    
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
