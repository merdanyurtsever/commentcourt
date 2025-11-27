"""
Sentiment Analysis Models for CommentCourt.

This module implements multiple sentiment analysis models for Turkish e-commerce
comments. Each model is registered with the ModelRegistry and follows the 
BaseMLModel interface for consistent usage.

Available Models:
1. RuleBasedModel - Simple lexicon-based sentiment analysis
2. LogisticRegressionModel - Traditional ML with TF-IDF features
3. LSTMModel - Deep learning with word embeddings
4. BertTurkishModel - Transformer-based using dbmdz/bert-base-turkish-cased
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import re

import numpy as np

from model.base import BaseMLModel, PredictionResult, ModelMetrics, TurkishTextPreprocessor
from model.registry import ModelRegistry


logger = logging.getLogger(__name__)


# =============================================================================
# Model 1: Rule-Based Sentiment Analysis
# =============================================================================

@ModelRegistry.register("rule_based", "Simple lexicon-based sentiment analysis for Turkish")
class RuleBasedModel(BaseMLModel):
    """
    Rule-based sentiment analyzer using Turkish sentiment lexicons.
    
    This model uses predefined word lists for positive and negative
    sentiments. It's fast, interpretable, and doesn't require training,
    making it useful as a baseline.
    """
    
    def __init__(self, name: str = "rule_based", version: str = "1.0.0"):
        super().__init__(name, version)
        self.preprocessor = TurkishTextPreprocessor(lowercase=True, remove_punctuation=True)
        
        # Turkish sentiment lexicon
        self.positive_words = {
            'güzel', 'harika', 'mükemmel', 'süper', 'muhteşem', 'kaliteli',
            'iyi', 'başarılı', 'hızlı', 'teşekkür', 'teşekkürler', 'memnun',
            'beğendim', 'sevdim', 'tavsiye', 'ederim', 'öneririm', 'şahane',
            'fevkalade', 'kusursuz', 'şık', 'zarif', 'rahat', 'konforlu',
            'sağlam', 'dayanıklı', 'pratik', 'kullanışlı', 'ideal', 'uygun',
            'olumlu', 'pozitif', 'hoş', 'tatlı', 'sevimli', 'efsane', 'bence',
            'bayıldım', 'aşık', 'oldum', 'müthiş', 'enfes', 'lezzetli',
            'temiz', 'hijyenik', 'güvenilir', 'dürüst', 'samimi', 'nazik'
        }
        
        self.negative_words = {
            'kötü', 'berbat', 'rezalet', 'felaket', 'korkunç', 'kalitesiz',
            'bozuk', 'yırtık', 'kırık', 'defolu', 'sahte', 'çakma', 'fake',
            'yavaş', 'geç', 'gecikme', 'iade', 'değişim', 'sorun', 'problem',
            'şikayet', 'memnun', 'değilim', 'beğenmedim', 'sevmedim', 'pişman',
            'aldatıldım', 'kandırıldım', 'hayal', 'kırıklığı', 'beklentimin',
            'altında', 'pahalı', 'fahiş', 'yüksek', 'fiyat', 'ucuz', 'görünüyor',
            'kokuyor', 'lekeli', 'kirli', 'eski', 'kullanılmış', 'çalışmıyor',
            'arızalı', 'bozuldu', 'patladı', 'yandı', 'sızdırıyor', 'dökülüyor',
            'maalesef', 'üzgün', 'sinir', 'kızgın', 'berbat', 'iğrenç'
        }
        
        self.negation_words = {'değil', 'yok', 'hiç', 'asla', 'kesinlikle'}
        self.intensifiers = {'çok', 'aşırı', 'fazla', 'son', 'derece', 'oldukça', 'gayet'}
        
        # Model is always "trained" for rule-based
        self.is_trained = True
    
    def train(self, texts: List[str], labels: List[str], 
              validation_split: float = 0.2, **kwargs) -> ModelMetrics:
        """Rule-based model doesn't need training, but we can optimize weights."""
        # Optionally learn word weights from data
        logger.info("Rule-based model doesn't require training")
        self.is_trained = True
        return self.evaluate(texts, labels)
    
    def predict_single(self, text: str) -> PredictionResult:
        """Predict sentiment for a single text."""
        processed = self.preprocessor.preprocess(text)
        words = processed.split()
        
        pos_score = 0.0
        neg_score = 0.0
        multiplier = 1.0
        negate = False
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.intensifiers:
                multiplier = 1.5
                continue
            
            # Check for negation
            if word in self.negation_words:
                negate = True
                continue
            
            # Score words
            if word in self.positive_words:
                if negate:
                    neg_score += multiplier
                else:
                    pos_score += multiplier
            elif word in self.negative_words:
                if negate:
                    pos_score += multiplier
                else:
                    neg_score += multiplier
            
            # Reset after scoring
            multiplier = 1.0
            negate = False
        
        # Determine sentiment
        total = pos_score + neg_score
        if total == 0:
            sentiment = 'neutral'
            confidence = 0.5
        elif pos_score > neg_score:
            sentiment = 'positive'
            confidence = pos_score / (total + 0.1)
        else:
            sentiment = 'negative'
            confidence = neg_score / (total + 0.1)
        
        # Normalize confidence to [0.5, 1.0] range
        confidence = 0.5 + (confidence * 0.5)
        confidence = min(confidence, 1.0)
        
        return PredictionResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            scores={
                'positive': pos_score / (total + 0.1) if total > 0 else 0.33,
                'negative': neg_score / (total + 0.1) if total > 0 else 0.33,
                'neutral': 0.33 if total == 0 else 0.1
            }
        )
    
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Predict sentiment for multiple texts."""
        return [self.predict_single(text) for text in texts]
    
    def evaluate(self, texts: List[str], labels: List[str]) -> ModelMetrics:
        """Evaluate model on test data."""
        predictions = self.predict(texts)
        pred_labels = [p.sentiment for p in predictions]
        
        # Calculate metrics
        correct = sum(1 for p, l in zip(pred_labels, labels) if p == l)
        accuracy = correct / len(labels) if labels else 0
        
        # Per-class metrics
        classes = ['positive', 'negative', 'neutral']
        precision_scores = []
        recall_scores = []
        
        for cls in classes:
            tp = sum(1 for p, l in zip(pred_labels, labels) if p == cls and l == cls)
            fp = sum(1 for p, l in zip(pred_labels, labels) if p == cls and l != cls)
            fn = sum(1 for p, l in zip(pred_labels, labels) if p != cls and l == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        self.metrics = ModelMetrics(
            accuracy=accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=f1,
            evaluation_time=datetime.now(),
            dataset_size=len(texts)
        )
        
        return self.metrics
    
    def save(self, path: Path) -> None:
        """Save lexicons and configuration."""
        path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'positive_words': list(self.positive_words),
            'negative_words': list(self.negative_words),
            'negation_words': list(self.negation_words),
            'intensifiers': list(self.intensifiers)
        }
        
        with open(path / 'lexicon.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.save_metadata(path)
        logger.info(f"Saved rule-based model to {path}")
    
    def load(self, path: Path) -> None:
        """Load lexicons from disk."""
        lexicon_path = path / 'lexicon.json'
        if lexicon_path.exists():
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.positive_words = set(data.get('positive_words', []))
            self.negative_words = set(data.get('negative_words', []))
            self.negation_words = set(data.get('negation_words', []))
            self.intensifiers = set(data.get('intensifiers', []))
        
        self.is_trained = True
        logger.info(f"Loaded rule-based model from {path}")


# =============================================================================
# Model 2: Logistic Regression with TF-IDF
# =============================================================================

@ModelRegistry.register("logistic_regression", "TF-IDF + Logistic Regression for Turkish sentiment")
class LogisticRegressionModel(BaseMLModel):
    """
    Traditional machine learning model using TF-IDF features and Logistic Regression.
    
    This model is fast to train and provides good interpretability through
    feature weights. Suitable for medium-sized datasets.
    """
    
    def __init__(self, name: str = "logistic_regression", version: str = "1.0.0"):
        super().__init__(name, version)
        self.preprocessor = TurkishTextPreprocessor(lowercase=True, remove_punctuation=False)
        
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        
        self._config = {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'C': 1.0,
            'max_iter': 1000
        }
    
    def _init_models(self):
        """Initialize sklearn models."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        
        self.vectorizer = TfidfVectorizer(
            max_features=self._config['max_features'],
            ngram_range=self._config['ngram_range'],
            sublinear_tf=True
        )
        
        self.classifier = LogisticRegression(
            C=self._config['C'],
            max_iter=self._config['max_iter'],
            class_weight='balanced',
            random_state=42
        )
        
        self.label_encoder = LabelEncoder()
    
    def train(self, texts: List[str], labels: List[str],
              validation_split: float = 0.2, **kwargs) -> ModelMetrics:
        """Train the logistic regression model."""
        from sklearn.model_selection import train_test_split
        
        self._init_models()
        
        # Preprocess texts
        processed = self.preprocessor.preprocess_batch(texts)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            processed, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Fit vectorizer and transform
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        # Train classifier
        logger.info("Training Logistic Regression model...")
        self.classifier.fit(X_train_vec, y_train_enc)
        
        self.is_trained = True
        
        # Evaluate on validation set
        return self.evaluate(X_val, y_val)
    
    def predict_single(self, text: str) -> PredictionResult:
        """Predict sentiment for a single text."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        processed = self.preprocessor.preprocess(text)
        X = self.vectorizer.transform([processed])
        
        proba = self.classifier.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        sentiment = self.label_encoder.inverse_transform([pred_idx])[0]
        
        scores = {
            label: float(prob) 
            for label, prob in zip(self.label_encoder.classes_, proba)
        }
        
        return PredictionResult(
            text=text,
            sentiment=sentiment,
            confidence=float(proba[pred_idx]),
            scores=scores
        )
    
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Predict sentiment for multiple texts."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        processed = self.preprocessor.preprocess_batch(texts)
        X = self.vectorizer.transform(processed)
        
        probas = self.classifier.predict_proba(X)
        predictions = []
        
        for text, proba in zip(texts, probas):
            pred_idx = np.argmax(proba)
            sentiment = self.label_encoder.inverse_transform([pred_idx])[0]
            
            scores = {
                label: float(prob)
                for label, prob in zip(self.label_encoder.classes_, proba)
            }
            
            predictions.append(PredictionResult(
                text=text,
                sentiment=sentiment,
                confidence=float(proba[pred_idx]),
                scores=scores
            ))
        
        return predictions
    
    def evaluate(self, texts: List[str], labels: List[str]) -> ModelMetrics:
        """Evaluate model on test data."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        predictions = self.predict(texts)
        pred_labels = [p.sentiment for p in predictions]
        
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='macro', zero_division=0
        )
        conf_matrix = confusion_matrix(labels, pred_labels).tolist()
        
        self.metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            evaluation_time=datetime.now(),
            dataset_size=len(texts)
        )
        
        return self.metrics
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(path / 'classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        self.save_metadata(path)
        logger.info(f"Saved LogisticRegression model to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        with open(path / 'vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(path / 'classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        
        with open(path / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"Loaded LogisticRegression model from {path}")


# =============================================================================
# Model 3: LSTM Neural Network
# =============================================================================

@ModelRegistry.register("lstm", "LSTM neural network for Turkish sentiment analysis")
class LSTMModel(BaseMLModel):
    """
    LSTM-based deep learning model for sentiment analysis.
    
    Uses word embeddings and bidirectional LSTM layers for
    sequence modeling. Suitable for capturing long-range dependencies
    in text.
    """
    
    def __init__(self, name: str = "lstm", version: str = "1.0.0"):
        super().__init__(name, version)
        self.preprocessor = TurkishTextPreprocessor(lowercase=True, remove_punctuation=False)
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        
        self._config = {
            'max_words': 20000,
            'max_len': 200,
            'embedding_dim': 128,
            'lstm_units': 64,
            'dropout': 0.3,
            'epochs': 10,
            'batch_size': 32
        }
    
    def _build_model(self, num_classes: int):
        """Build LSTM model architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
        
        self.model = Sequential([
            Embedding(
                self._config['max_words'],
                self._config['embedding_dim'],
                input_length=self._config['max_len']
            ),
            Bidirectional(LSTM(self._config['lstm_units'], return_sequences=True)),
            Dropout(self._config['dropout']),
            Bidirectional(LSTM(self._config['lstm_units'] // 2)),
            Dropout(self._config['dropout']),
            Dense(64, activation='relu'),
            Dropout(self._config['dropout']),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, texts: List[str], labels: List[str],
              validation_split: float = 0.2, **kwargs) -> ModelMetrics:
        """Train the LSTM model."""
        try:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
        except ImportError:
            logger.error("Required packages not installed")
            raise
        
        # Update config with kwargs
        self._config.update(kwargs)
        
        # Preprocess texts
        processed = self.preprocessor.preprocess_batch(texts)
        
        # Tokenize
        self.tokenizer = Tokenizer(num_words=self._config['max_words'])
        self.tokenizer.fit_on_texts(processed)
        sequences = self.tokenizer.texts_to_sequences(processed)
        X = pad_sequences(sequences, maxlen=self._config['max_len'])
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Build and train model
        num_classes = len(self.label_encoder.classes_)
        self._build_model(num_classes)
        
        logger.info("Training LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self._config['epochs'],
            batch_size=self._config['batch_size'],
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        val_texts = [texts[i] for i, x in enumerate(X) if any(np.array_equal(x, xv) for xv in X_val)]
        val_labels = [labels[i] for i in range(len(labels)) if y[i] in y_val[:10]]
        
        return self.evaluate(texts[-len(X_val):], labels[-len(X_val):])
    
    def predict_single(self, text: str) -> PredictionResult:
        """Predict sentiment for a single text."""
        results = self.predict([text])
        return results[0]
    
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Predict sentiment for multiple texts."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        processed = self.preprocessor.preprocess_batch(texts)
        sequences = self.tokenizer.texts_to_sequences(processed)
        X = pad_sequences(sequences, maxlen=self._config['max_len'])
        
        probas = self.model.predict(X, verbose=0)
        predictions = []
        
        for text, proba in zip(texts, probas):
            pred_idx = np.argmax(proba)
            sentiment = self.label_encoder.inverse_transform([pred_idx])[0]
            
            scores = {
                label: float(prob)
                for label, prob in zip(self.label_encoder.classes_, proba)
            }
            
            predictions.append(PredictionResult(
                text=text,
                sentiment=sentiment,
                confidence=float(proba[pred_idx]),
                scores=scores
            ))
        
        return predictions
    
    def evaluate(self, texts: List[str], labels: List[str]) -> ModelMetrics:
        """Evaluate model on test data."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        predictions = self.predict(texts)
        pred_labels = [p.sentiment for p in predictions]
        
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='macro', zero_division=0
        )
        conf_matrix = confusion_matrix(labels, pred_labels).tolist()
        
        self.metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            evaluation_time=datetime.now(),
            dataset_size=len(texts)
        )
        
        return self.metrics
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path / 'lstm_model.keras')
        
        with open(path / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open(path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        self.save_metadata(path)
        logger.info(f"Saved LSTM model to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        from tensorflow.keras.models import load_model
        
        self.model = load_model(path / 'lstm_model.keras')
        
        with open(path / 'tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open(path / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"Loaded LSTM model from {path}")


# =============================================================================
# Model 4: BERT Turkish
# =============================================================================

@ModelRegistry.register("bert_turkish", "BERT Turkish transformer model for sentiment analysis")
class BertTurkishModel(BaseMLModel):
    """
    BERT-based transformer model fine-tuned for Turkish sentiment analysis.
    
    Uses the dbmdz/bert-base-turkish-cased pretrained model with a
    classification head for sentiment prediction. Most accurate but
    resource-intensive.
    """
    
    def __init__(self, name: str = "bert_turkish", version: str = "1.0.0"):
        super().__init__(name, version)
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.device = None
        
        self._config = {
            'model_name': 'dbmdz/bert-base-turkish-cased',
            'max_length': 256,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'epochs': 3,
            'warmup_steps': 100
        }
    
    def _init_model(self, num_labels: int):
        """Initialize BERT model and tokenizer."""
        try:
            import torch
            from transformers import BertTokenizer, BertForSequenceClassification
        except ImportError:
            logger.error("PyTorch/Transformers not installed")
            raise
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self._config['model_name'])
        self.model = BertForSequenceClassification.from_pretrained(
            self._config['model_name'],
            num_labels=num_labels
        )
        self.model.to(self.device)
    
    def train(self, texts: List[str], labels: List[str],
              validation_split: float = 0.2, **kwargs) -> ModelMetrics:
        """Fine-tune BERT model."""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from transformers import AdamW, get_linear_schedule_with_warmup
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
        except ImportError:
            logger.error("Required packages not installed")
            raise
        
        self._config.update(kwargs)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        # Initialize model
        num_labels = len(self.label_encoder.classes_)
        self._init_model(num_labels)
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self._config['max_length'],
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(y)
        )
        
        # Split
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._config['batch_size']
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self._config['learning_rate'])
        total_steps = len(train_loader) * self._config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info("Fine-tuning BERT model...")
        self.model.train()
        
        for epoch in range(self._config['epochs']):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, batch_labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{self._config['epochs']}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        # Evaluate on validation set
        val_texts = texts[-val_size:]
        val_labels = labels[-val_size:]
        return self.evaluate(val_texts, val_labels)
    
    def predict_single(self, text: str) -> PredictionResult:
        """Predict sentiment for a single text."""
        results = self.predict([text])
        return results[0]
    
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        """Predict sentiment for multiple texts."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        import torch
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self._config['batch_size']):
                batch_texts = texts[i:i + self._config['batch_size']]
                
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self._config['max_length'],
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probas = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                
                for text, proba in zip(batch_texts, probas):
                    pred_idx = np.argmax(proba)
                    sentiment = self.label_encoder.inverse_transform([pred_idx])[0]
                    
                    scores = {
                        label: float(prob)
                        for label, prob in zip(self.label_encoder.classes_, proba)
                    }
                    
                    predictions.append(PredictionResult(
                        text=text,
                        sentiment=sentiment,
                        confidence=float(proba[pred_idx]),
                        scores=scores
                    ))
        
        return predictions
    
    def evaluate(self, texts: List[str], labels: List[str]) -> ModelMetrics:
        """Evaluate model on test data."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        predictions = self.predict(texts)
        pred_labels = [p.sentiment for p in predictions]
        
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='macro', zero_division=0
        )
        conf_matrix = confusion_matrix(labels, pred_labels).tolist()
        
        self.metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=conf_matrix,
            evaluation_time=datetime.now(),
            dataset_size=len(texts)
        )
        
        return self.metrics
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(path / 'bert_model')
        self.tokenizer.save_pretrained(path / 'bert_model')
        
        with open(path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        self.save_metadata(path)
        logger.info(f"Saved BERT model to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BertTokenizer.from_pretrained(path / 'bert_model')
        self.model = BertForSequenceClassification.from_pretrained(path / 'bert_model')
        self.model.to(self.device)
        
        with open(path / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_trained = True
        logger.info(f"Loaded BERT model from {path}")
