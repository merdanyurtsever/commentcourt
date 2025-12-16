"""
Database Models and Utilities for CommentCourt.

This module provides SQLAlchemy ORM models for storing:
- Influencer profiles and rankings
- Comments and their sentiment analysis
- Model predictions and metrics
- Analysis history

The database is designed to be extensible and supports
multiple ML models with versioned predictions.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import json

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, DateTime,
    Boolean, ForeignKey, JSON, Enum as SQLEnum, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.pool import StaticPool
import enum


logger = logging.getLogger(__name__)

Base = declarative_base()


# =============================================================================
# Enums
# =============================================================================

class SentimentLabel(enum.Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Platform(enum.Enum):
    """Supported e-commerce platforms."""
    TRENDYOL = "trendyol"
    HEPSIBURADA = "hepsiburada"
    N11 = "n11"
    MILLA = "milla"
    OTHER = "other"


# =============================================================================
# ORM Models
# =============================================================================

class Influencer(Base):
    __tablename__ = 'influencers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    bio = Column(Text, nullable=True)
    profile_image = Column(String(500), nullable=True)
    
    instagram_url = Column(String(500), nullable=True)
    youtube_url = Column(String(500), nullable=True)
    tiktok_url = Column(String(500), nullable=True)
    website_url = Column(String(500), nullable=True)
    
    overall_score = Column(Float, default=0.0)
    positive_ratio = Column(Float, default=0.0)
    negative_ratio = Column(Float, default=0.0)
    neutral_ratio = Column(Float, default=0.0)
    total_comments = Column(Integer, default=0)
    rank = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    storefronts = relationship("Storefront", back_populates="influencer", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="influencer", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Influencer(id={self.id}, name='{self.name}', score={self.overall_score:.2f})>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'bio': self.bio,
            'profile_image': self.profile_image,
            'instagram_url': self.instagram_url,
            'youtube_url': self.youtube_url,
            'tiktok_url': self.tiktok_url,
            'website_url': self.website_url,
            'overall_score': round(self.overall_score, 2),
            'positive_ratio': round(self.positive_ratio, 2),
            'negative_ratio': round(self.negative_ratio, 2),
            'neutral_ratio': round(self.neutral_ratio, 2),
            'total_comments': self.total_comments,
            'rank': self.rank,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Storefront(Base):
    __tablename__ = 'storefronts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    influencer_id = Column(Integer, ForeignKey('influencers.id'), nullable=False)
    
    platform = Column(SQLEnum(Platform), nullable=False)
    url = Column(String(500), nullable=False)
    store_name = Column(String(255), nullable=True)
    
    product_count = Column(Integer, default=0)
    last_scraped = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    influencer = relationship("Influencer", back_populates="storefronts")
    
    __table_args__ = (
        UniqueConstraint('influencer_id', 'url', name='uix_influencer_url'),
    )
    
    def __repr__(self):
        return f"<Storefront(id={self.id}, platform={self.platform.value}, url='{self.url[:30]}...')>"


class Comment(Base):
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    influencer_id = Column(Integer, ForeignKey('influencers.id'), nullable=False)
    
    text = Column(Text, nullable=False)
    text_cleaned = Column(Text, nullable=True)
    
    platform = Column(SQLEnum(Platform), nullable=True)
    source_url = Column(String(500), nullable=True)
    product_name = Column(String(500), nullable=True)
    original_rating = Column(Float, nullable=True)
    comment_date = Column(DateTime, nullable=True)
    
    sentiment = Column(SQLEnum(SentimentLabel), nullable=True)
    confidence = Column(Float, nullable=True)
    sentiment_scores = Column(JSON, nullable=True)
    
    analyzed_at = Column(DateTime, nullable=True)
    model_used = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    scraped_at = Column(DateTime, nullable=True)
    
    influencer = relationship("Influencer", back_populates="comments")
    predictions = relationship("ModelPrediction", back_populates="comment", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_comments_sentiment', 'sentiment'),
        Index('ix_comments_influencer_sentiment', 'influencer_id', 'sentiment'),
    )
    
    def __repr__(self):
        return f"<Comment(id={self.id}, sentiment={self.sentiment}, text='{self.text[:30]}...')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'influencer_id': self.influencer_id,
            'text': self.text,
            'platform': self.platform.value if self.platform else None,
            'product_name': self.product_name,
            'original_rating': self.original_rating,
            'sentiment': self.sentiment.value if self.sentiment else None,
            'confidence': self.confidence,
            'sentiment_scores': self.sentiment_scores,
            'model_used': self.model_used,
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None,
            'comment_date': self.comment_date.isoformat() if self.comment_date else None
        }


class ModelPrediction(Base):
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    comment_id = Column(Integer, ForeignKey('comments.id'), nullable=False)
    
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)
    
    sentiment = Column(SQLEnum(SentimentLabel), nullable=False)
    confidence = Column(Float, nullable=False)
    scores = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    comment = relationship("Comment", back_populates="predictions")
    
    __table_args__ = (
        UniqueConstraint('comment_id', 'model_name', 'model_version', name='uix_comment_model'),
        Index('ix_predictions_model', 'model_name', 'model_version'),
    )
    
    def __repr__(self):
        return f"<ModelPrediction(model={self.model_name}, sentiment={self.sentiment.value})>"


class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    confusion_matrix = Column(JSON, nullable=True)
    classification_report = Column(JSON, nullable=True)
    dataset_size = Column(Integer, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    
    is_best_model = Column(Boolean, default=False)
    
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_metrics_f1', 'f1_score'),
        Index('ix_metrics_model', 'model_name', 'model_version'),
    )
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_name}, f1={self.f1_score:.4f})>"


class AnalysisRun(Base):
    __tablename__ = 'analysis_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default='running')
    
    comments_processed = Column(Integer, default=0)
    influencers_updated = Column(Integer, default=0)
    model_used = Column(String(100), nullable=True)
    
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<AnalysisRun(id={self.id}, status={self.status})>"


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """Manages database connections and provides helper methods."""
    
    def __init__(self, db_path: str = "database/db.sqlite3"):
        """Initialize database manager."""
        self.db_path = db_path
        self.engine = None
        self.Session = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize SQLAlchemy engine."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        
        self.Session = scoped_session(sessionmaker(bind=self.engine))
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)
        logger.info("Database tables dropped")
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def add_influencer(self, name: str, bio: str = None, **kwargs) -> Influencer:
        with self.session_scope() as session:
            influencer = Influencer(name=name, bio=bio, **kwargs)
            session.add(influencer)
            session.flush()
            return influencer
    
    def get_influencer(self, influencer_id: int) -> Optional[Influencer]:
        with self.session_scope() as session:
            return session.query(Influencer).filter_by(id=influencer_id).first()
    
    def get_influencer_by_name(self, name: str) -> Optional[Influencer]:
        with self.session_scope() as session:
            return session.query(Influencer).filter_by(name=name).first()
    
    def get_all_influencers(self, active_only: bool = True) -> List[Influencer]:
        with self.session_scope() as session:
            query = session.query(Influencer)
            if active_only:
                query = query.filter_by(is_active=True)
            return query.order_by(Influencer.rank.asc().nullslast()).all()
    
    def get_top_influencers(self, limit: int = 10) -> List[Influencer]:
        with self.session_scope() as session:
            return session.query(Influencer)\
                .filter(Influencer.is_active == True)\
                .order_by(Influencer.overall_score.desc())\
                .limit(limit)\
                .all()
    
    def update_influencer_scores(self, influencer_id: int) -> None:
        with self.session_scope() as session:
            influencer = session.query(Influencer).filter_by(id=influencer_id).first()
            if not influencer:
                return
            
            comments = session.query(Comment)\
                .filter_by(influencer_id=influencer_id)\
                .filter(Comment.sentiment.isnot(None))\
                .all()
            
            total = len(comments)
            if total == 0:
                return
            
            positive = sum(1 for c in comments if c.sentiment == SentimentLabel.POSITIVE)
            negative = sum(1 for c in comments if c.sentiment == SentimentLabel.NEGATIVE)
            neutral = sum(1 for c in comments if c.sentiment == SentimentLabel.NEUTRAL)
            
            influencer.total_comments = total
            influencer.positive_ratio = positive / total
            influencer.negative_ratio = negative / total
            influencer.neutral_ratio = neutral / total
            
            raw_score = (influencer.positive_ratio * 10) - (influencer.negative_ratio * 5) + (influencer.neutral_ratio * 2.5)
            influencer.overall_score = max(0, min(10, raw_score))
            
            influencer.updated_at = datetime.utcnow()
    
    def update_all_rankings(self) -> None:
        with self.session_scope() as session:
            influencers = session.query(Influencer)\
                .filter_by(is_active=True)\
                .order_by(Influencer.overall_score.desc())\
                .all()
            
            for rank, influencer in enumerate(influencers, start=1):
                influencer.rank = rank
    
    def add_comment(self, influencer_id: int, text: str, **kwargs) -> Comment:
        with self.session_scope() as session:
            comment = Comment(influencer_id=influencer_id, text=text, **kwargs)
            session.add(comment)
            session.flush()
            return comment
    
    def add_comments_batch(self, comments: List[Dict[str, Any]]) -> int:
        with self.session_scope() as session:
            count = 0
            for comment_data in comments:
                comment = Comment(**comment_data)
                session.add(comment)
                count += 1
            return count
    
    def get_unanalyzed_comments(self, limit: int = 1000) -> List[Comment]:
        with self.session_scope() as session:
            return session.query(Comment)\
                .filter(Comment.sentiment.is_(None))\
                .limit(limit)\
                .all()
    
    def get_comments_by_influencer(self, influencer_id: int, 
                                    sentiment: Optional[SentimentLabel] = None) -> List[Comment]:
        with self.session_scope() as session:
            query = session.query(Comment).filter_by(influencer_id=influencer_id)
            if sentiment:
                query = query.filter_by(sentiment=sentiment)
            return query.order_by(Comment.created_at.desc()).all()
    
    def update_comment_sentiment(self, comment_id: int, sentiment: str,
                                  confidence: float, scores: Dict[str, float],
                                  model_name: str) -> None:
        with self.session_scope() as session:
            comment = session.query(Comment).filter_by(id=comment_id).first()
            if comment:
                comment.sentiment = SentimentLabel(sentiment)
                comment.confidence = confidence
                comment.sentiment_scores = scores
                comment.model_used = model_name
                comment.analyzed_at = datetime.utcnow()
    
    def save_model_metrics(self, model_name: str, metrics: Dict[str, Any],
                           model_version: str = None) -> ModelMetrics:
        with self.session_scope() as session:
            model_metrics = ModelMetrics(
                model_name=model_name,
                model_version=model_version,
                accuracy=metrics.get('accuracy'),
                precision=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1_score=metrics.get('f1_score'),
                confusion_matrix=metrics.get('confusion_matrix'),
                classification_report=metrics.get('classification_report'),
                dataset_size=metrics.get('dataset_size'),
                training_duration_seconds=metrics.get('training_duration_seconds')
            )
            session.add(model_metrics)
            session.flush()
            return model_metrics
    
    def get_best_model(self) -> Optional[ModelMetrics]:
        with self.session_scope() as session:
            return session.query(ModelMetrics)\
                .filter_by(is_best_model=True)\
                .first()
    
    def set_best_model(self, model_name: str, model_version: str = None) -> None:
        with self.session_scope() as session:
            session.query(ModelMetrics).update({ModelMetrics.is_best_model: False})
            
            query = session.query(ModelMetrics).filter_by(model_name=model_name)
            if model_version:
                query = query.filter_by(model_version=model_version)
            
            metrics = query.order_by(ModelMetrics.evaluated_at.desc()).first()
            if metrics:
                metrics.is_best_model = True
    
    def get_model_comparison(self) -> List[Dict[str, Any]]:
        with self.session_scope() as session:
            metrics_list = session.query(ModelMetrics)\
                .order_by(ModelMetrics.f1_score.desc())\
                .all()
            
            return [
                {
                    'model_name': m.model_name,
                    'model_version': m.model_version,
                    'accuracy': m.accuracy,
                    'precision': m.precision,
                    'recall': m.recall,
                    'f1_score': m.f1_score,
                    'is_best': m.is_best_model,
                    'evaluated_at': m.evaluated_at.isoformat() if m.evaluated_at else None
                }
                for m in metrics_list
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        with self.session_scope() as session:
            total_influencers = session.query(Influencer).count()
            active_influencers = session.query(Influencer).filter_by(is_active=True).count()
            total_comments = session.query(Comment).count()
            analyzed_comments = session.query(Comment).filter(Comment.sentiment.isnot(None)).count()
            
            sentiment_dist = {}
            for sentiment in SentimentLabel:
                count = session.query(Comment).filter_by(sentiment=sentiment).count()
                sentiment_dist[sentiment.value] = count
            
            return {
                'total_influencers': total_influencers,
                'active_influencers': active_influencers,
                'total_comments': total_comments,
                'analyzed_comments': analyzed_comments,
                'unanalyzed_comments': total_comments - analyzed_comments,
                'sentiment_distribution': sentiment_dist
            }


# =============================================================================
# Convenience Functions
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_db(db_path: str = "database/db.sqlite3") -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager


def init_db(db_path: str = "database/db.sqlite3") -> DatabaseManager:
    """Initialize database with tables."""
    db = get_db(db_path)
    db.create_tables()
    return db
