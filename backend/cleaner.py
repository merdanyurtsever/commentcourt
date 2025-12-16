"""Text cleaning utilities for Turkish comments (simplified)."""
import re
import logging
import html
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class CleanerConfig:
    """Configuration for text cleaner."""
    
    # Basic cleaning
    lowercase: bool = True
    strip_whitespace: bool = True
    normalize_whitespace: bool = True
    
    # Removal options
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_html_tags: bool = True
    remove_punctuation: bool = False  # Keep punctuation for better context
    remove_numbers: bool = False
    remove_emojis: bool = False  # Emojis can carry sentiment
    
    # Turkish-specific
    normalize_turkish_chars: bool = False  # ğ→g, ü→u, etc.
    expand_turkish_abbreviations: bool = True
    
    # Advanced
    min_word_length: int = 1
    max_word_length: int = 50
    min_text_length: int = 2
    
    # Patterns to remove
    custom_patterns: List[str] = field(default_factory=list)


class TextCleaner:
    """
    Main text cleaner for Turkish e-commerce comments.
    """
    
    def __init__(self, config: Optional[CleanerConfig] = None):
        """
        Initialize cleaner.
        """
        self.config = config or CleanerConfig()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Turkish abbreviation expansions
        # small set of common abbreviations for readability
        self.abbreviations = {
            'tşk': 'teşekkür',
            'mrb': 'merhaba',
            'slm': 'selam',
            'ok': 'tamam',
            'tşkler': 'teşekkürler'
        }
        
        # Turkish character mappings (for normalization)
        self.turkish_char_map = {
            'ı': 'i', 'İ': 'I',
            'ğ': 'g', 'Ğ': 'G',
            'ü': 'u', 'Ü': 'U',
            'ş': 's', 'Ş': 'S',
            'ö': 'o', 'Ö': 'O',
            'ç': 'c', 'Ç': 'C'
        }
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        # URL pattern
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+',
            re.IGNORECASE
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number pattern (Turkish format)
        self.phone_pattern = re.compile(
            r'(?:\+90|0)?[- ]?(?:5\d{2}|[1-9]\d{2})[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}'
        )
        
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Multiple spaces
        self.multispace_pattern = re.compile(r'\s+')
        
        # Repeated characters (e.g., "güzelllll" → "güzel")
        self.repeated_chars_pattern = re.compile(r'(.)\1{2,}')
        
        # Emoji pattern
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        
        # Punctuation (excluding sentiment markers like ! and ?)
        self.punctuation_pattern = re.compile(r'[^\w\s!?.,]')
        
        # Numbers
        self.number_pattern = re.compile(r'\d+')
    
    def clean(self, text: str) -> str:
        """
        Clean and preprocess text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        if self.config.remove_html_tags:
            text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = self.phone_pattern.sub(' ', text)
        
        # Remove emojis
        if self.config.remove_emojis:
            text = self.emoji_pattern.sub(' ', text)
        
        # Normalize repeated characters
        text = self.repeated_chars_pattern.sub(r'\1\1', text)
        
        # Expand Turkish abbreviations
        if self.config.expand_turkish_abbreviations:
            text = self._expand_abbreviations(text)
        
        # Normalize Turkish characters (optional)
        if self.config.normalize_turkish_chars:
            text = self._normalize_turkish_chars(text)
        
        # Lowercase
        if self.config.lowercase:
            # Turkish-aware lowercasing
            text = self._turkish_lower(text)
        
        # Remove punctuation
        if self.config.remove_punctuation:
            text = self.punctuation_pattern.sub(' ', text)
        
        # Remove numbers
        if self.config.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        # Remove custom patterns
        for pattern in self.config.custom_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.multispace_pattern.sub(' ', text)
        
        # Strip
        if self.config.strip_whitespace:
            text = text.strip()
        
        # Length filtering
        if len(text) < self.config.min_text_length:
            return ""
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts.
        """
        return [self.clean(text) for text in texts]
    
    def _turkish_lower(self, text: str) -> str:
        """Turkish-aware lowercasing."""
        # Handle Turkish I/ı properly
        text = text.replace('I', 'ı').replace('İ', 'i')
        return text.lower()
    
    def _normalize_turkish_chars(self, text: str) -> str:
        """Replace Turkish special characters with ASCII equivalents."""
        for turkish, ascii_char in self.turkish_char_map.items():
            text = text.replace(turkish, ascii_char)
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Turkish abbreviations."""
        words = text.split()
        expanded = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word in self.abbreviations:
                expanded.append(self.abbreviations[lower_word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        """
        tokens = text.split()
        
        # Filter by length
        tokens = [
            t for t in tokens
            if self.config.min_word_length <= len(t) <= self.config.max_word_length
        ]
        
        return tokens
    
    def get_stats(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about text.
        """
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        
        return {
            'original_length': len(text),
            'cleaned_length': len(cleaned),
            'token_count': len(tokens),
            'has_emoji': bool(self.emoji_pattern.search(text)),
            'has_url': bool(self.url_pattern.search(text)),
            'unique_tokens': len(set(tokens)),
            'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        }


class CommentCleaner(TextCleaner):
    """
    Specialized cleaner for e-commerce comments.
    """
    
    def __init__(self, config: Optional[CleanerConfig] = None):
        """Initialize with comment-specific defaults."""
        if config is None:
            config = CleanerConfig(
                lowercase=True,
                remove_urls=True,
                remove_emails=True,
                remove_phone_numbers=True,
                remove_html_tags=True,
                remove_punctuation=False,  # Keep for sentiment
                remove_numbers=False,
                remove_emojis=False,  # Emojis carry sentiment
                expand_turkish_abbreviations=True
            )
        
        super().__init__(config)
        
        # E-commerce specific patterns
        self.order_number_pattern = re.compile(r"sipariş\s*(?:no|numarası?)?\s*:?\s*\d+", re.IGNORECASE)
        self.date_pattern = re.compile(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}')
        self.price_pattern = re.compile(r'\d+[.,]?\d*\s*(?:TL|₺|lira)', re.IGNORECASE)
    
    def clean(self, text: str) -> str:
        """
        Clean e-commerce comment.
        """
        if not text:
            return ""
        
        # Remove order numbers
        text = self.order_number_pattern.sub(' ', text)
        
        # Keep prices but normalize
        text = self.price_pattern.sub(' [FİYAT] ', text)
        
        # Remove dates
        text = self.date_pattern.sub(' ', text)
        
        # Apply base cleaning
        return super().clean(text)


# Convenience functions

def clean_text(text: str, **kwargs) -> str:
    config = CleanerConfig(**kwargs)
    cleaner = TextCleaner(config)
    return cleaner.clean(text)


def clean_texts(texts: List[str], **kwargs) -> List[str]:
    config = CleanerConfig(**kwargs)
    cleaner = TextCleaner(config)
    return cleaner.clean_batch(texts)
