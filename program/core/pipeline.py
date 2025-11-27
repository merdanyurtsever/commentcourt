"""
Analysis Pipeline for CommentCourt.

This module orchestrates the complete data processing flow:
1. Load raw data (from scraped JSON files or database)
2. Clean and preprocess text
3. Run sentiment analysis with multiple models
4. Select best model and use its predictions
5. Update influencer scores and rankings
6. Store results in database

The pipeline is designed to be modular and supports:
- Batch processing
- Incremental updates
- Model comparison
- Error recovery
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import json

from program.utils.db import (
    DatabaseManager, get_db, init_db,
    Influencer, Comment, SentimentLabel, Platform
)
from program.core.cleaner import TextCleaner
from program.core.scorer import InfluencerScorer
from model.registry import ModelRegistry, auto_discover_models
from model.base import BaseMLModel, PredictionResult


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    
    # Data paths
    raw_data_dir: Path = field(default_factory=lambda: Path("database/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("database/processed"))
    models_dir: Path = field(default_factory=lambda: Path("model/weights"))
    
    # Database
    db_path: str = "database/db.sqlite3"
    
    # Processing
    batch_size: int = 100
    max_comments_per_run: int = 10000
    
    # Model selection
    models_to_use: Optional[List[str]] = None  # None means all
    selection_metric: str = "f1_score"
    
    # Scoring
    score_scale: int = 10  # Score out of 10 or 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'raw_data_dir': str(self.raw_data_dir),
            'processed_data_dir': str(self.processed_data_dir),
            'models_dir': str(self.models_dir),
            'db_path': self.db_path,
            'batch_size': self.batch_size,
            'max_comments_per_run': self.max_comments_per_run,
            'models_to_use': self.models_to_use,
            'selection_metric': self.selection_metric,
            'score_scale': self.score_scale
        }


@dataclass
class PipelineResult:
    """Results from a pipeline run."""
    
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Counts
    comments_loaded: int = 0
    comments_cleaned: int = 0
    comments_analyzed: int = 0
    influencers_updated: int = 0
    
    # Model info
    best_model: Optional[str] = None
    model_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'comments_loaded': self.comments_loaded,
            'comments_cleaned': self.comments_cleaned,
            'comments_analyzed': self.comments_analyzed,
            'influencers_updated': self.influencers_updated,
            'best_model': self.best_model,
            'model_metrics': self.model_metrics,
            'errors': self.errors
        }


class AnalysisPipeline:
    """
    Main analysis pipeline for processing influencer comments.
    
    Orchestrates the complete flow from raw data to scored influencers.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.db = get_db(self.config.db_path)
        self.cleaner = TextCleaner()
        self.scorer = InfluencerScorer(scale=self.config.score_scale)
        
        self.result = PipelineResult()
        self._best_model: Optional[BaseMLModel] = None
        
        # Ensure directories exist
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-discover models
        auto_discover_models()
    
    def run(self, skip_training: bool = False) -> PipelineResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            skip_training: If True, use existing trained models
            
        Returns:
            PipelineResult with execution details
        """
        self.result = PipelineResult()
        self.result.started_at = datetime.now()
        self.result.status = "running"
        
        try:
            # Step 1: Initialize database
            logger.info("Step 1: Initializing database")
            self.db.create_tables()
            
            # Step 2: Load and import raw data
            logger.info("Step 2: Loading raw data")
            self._load_raw_data()
            
            # Step 3: Clean comments
            logger.info("Step 3: Cleaning comments")
            self._clean_comments()
            
            # Step 4: Train/load models and analyze
            logger.info("Step 4: Running sentiment analysis")
            self._run_analysis(skip_training=skip_training)
            
            # Step 5: Update influencer scores
            logger.info("Step 5: Updating influencer scores")
            self._update_scores()
            
            # Step 6: Update rankings
            logger.info("Step 6: Updating rankings")
            self._update_rankings()
            
            self.result.status = "completed"
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.result.errors.append(str(e))
            self.result.status = "failed"
        
        finally:
            self.result.completed_at = datetime.now()
        
        # Log summary
        self._log_summary()
        
        return self.result
    
    def _load_raw_data(self) -> None:
        """Load raw data from JSON files into database."""
        json_files = list(self.config.raw_data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._import_scraped_data(data)
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                self.result.errors.append(f"Load error: {json_file}")
    
    def _import_scraped_data(self, data: Dict[str, Any]) -> None:
        """
        Import scraped data into database.
        
        Expected format (from scraper):
        {
            "influencer": "Name",
            "platform": "trendyol",
            "storefront_url": "...",
            "comments": [
                {"text": "...", "rating": 5, "date": "..."},
                ...
            ]
        }
        """
        influencer_name = data.get('influencer', 'Unknown')
        platform_str = data.get('platform', 'other')
        comments_data = data.get('comments', [])
        
        # Get or create influencer
        with self.db.session_scope() as session:
            influencer = session.query(Influencer)\
                .filter_by(name=influencer_name)\
                .first()
            
            if not influencer:
                influencer = Influencer(name=influencer_name)
                session.add(influencer)
                session.flush()
            
            influencer_id = influencer.id
        
        # Map platform
        try:
            platform = Platform(platform_str.lower())
        except ValueError:
            platform = Platform.OTHER
        
        # Add comments
        for comment_data in comments_data:
            with self.db.session_scope() as session:
                # Check for duplicate
                existing = session.query(Comment)\
                    .filter_by(influencer_id=influencer_id, text=comment_data.get('text', ''))\
                    .first()
                
                if not existing:
                    comment = Comment(
                        influencer_id=influencer_id,
                        text=comment_data.get('text', ''),
                        platform=platform,
                        original_rating=comment_data.get('rating'),
                        product_name=comment_data.get('product'),
                        source_url=data.get('storefront_url')
                    )
                    session.add(comment)
                    self.result.comments_loaded += 1
    
    def _clean_comments(self) -> None:
        """Clean and preprocess all unprocessed comments."""
        with self.db.session_scope() as session:
            # Get comments without cleaned text
            comments = session.query(Comment)\
                .filter(Comment.text_cleaned.is_(None))\
                .limit(self.config.max_comments_per_run)\
                .all()
            
            for comment in comments:
                cleaned = self.cleaner.clean(comment.text)
                comment.text_cleaned = cleaned
                self.result.comments_cleaned += 1
    
    def _run_analysis(self, skip_training: bool = False) -> None:
        """Run sentiment analysis using best available model."""
        # Get available models
        models = self._get_models()
        
        if not models:
            raise RuntimeError("No models available for analysis")
        
        # Select best model
        self._best_model = self._select_best_model(models, skip_training)
        
        if not self._best_model:
            raise RuntimeError("Could not select best model")
        
        self.result.best_model = self._best_model.name
        
        # Analyze unanalyzed comments
        self._analyze_comments()
    
    def _get_models(self) -> List[BaseMLModel]:
        """Get list of models to use."""
        model_names = self.config.models_to_use
        
        if not model_names:
            model_names = [m['name'] for m in ModelRegistry.list_models()]
        
        models = []
        for name in model_names:
            try:
                model = ModelRegistry.get(name)
                models.append(model)
            except Exception as e:
                logger.warning(f"Could not load model {name}: {e}")
        
        return models
    
    def _select_best_model(self, models: List[BaseMLModel], 
                           skip_training: bool) -> Optional[BaseMLModel]:
        """
        Select the best model based on evaluation metrics.
        
        If models are not trained, trains them first (unless skip_training=True).
        """
        from model.train import Dataset, ModelTrainer
        
        # Try to load existing best model
        best_in_db = self.db.get_best_model()
        if best_in_db and skip_training:
            try:
                model = ModelRegistry.get(best_in_db.model_name)
                model_path = self.config.models_dir / best_in_db.model_name
                if model_path.exists():
                    model.load(model_path)
                    logger.info(f"Loaded best model from DB: {best_in_db.model_name}")
                    return model
            except Exception as e:
                logger.warning(f"Could not load saved best model: {e}")
        
        # If we need to train/evaluate models
        if not skip_training:
            # Create or load training dataset
            dataset_path = self.config.processed_data_dir / "training_data.json"
            
            if dataset_path.exists():
                dataset = Dataset.from_json(dataset_path)
            else:
                # Create sample dataset for initial setup
                logger.warning("No training data found, using sample dataset")
                dataset = Dataset.create_sample_dataset(500)
                dataset.to_json(dataset_path)
            
            dataset.split()
            
            # Train and evaluate each model
            for model in models:
                try:
                    logger.info(f"Training {model.name}...")
                    metrics = model.train(
                        dataset.train_texts,
                        dataset.train_labels
                    )
                    
                    # Evaluate on test set
                    test_metrics = model.evaluate(
                        dataset.test_texts,
                        dataset.test_labels
                    )
                    
                    # Store metrics
                    ModelRegistry.update_metrics(model.name, test_metrics)
                    self.db.save_model_metrics(model.name, test_metrics.to_dict())
                    
                    # Save model
                    model_path = self.config.models_dir / model.name
                    model.save(model_path)
                    
                    self.result.model_metrics[model.name] = test_metrics.to_dict()
                    
                except Exception as e:
                    logger.error(f"Error training {model.name}: {e}")
                    self.result.errors.append(f"Training error: {model.name}")
        
        # Select best based on metric
        best_name = ModelRegistry.select_best_model(self.config.selection_metric)
        
        if best_name:
            self.db.set_best_model(best_name)
            return ModelRegistry.get_cached(best_name)
        
        # Fallback to first model
        return models[0] if models else None
    
    def _analyze_comments(self) -> None:
        """Analyze comments using the best model."""
        if not self._best_model:
            return
        
        with self.db.session_scope() as session:
            # Get unanalyzed comments in batches
            offset = 0
            
            while True:
                comments = session.query(Comment)\
                    .filter(Comment.sentiment.is_(None))\
                    .filter(Comment.text_cleaned.isnot(None))\
                    .offset(offset)\
                    .limit(self.config.batch_size)\
                    .all()
                
                if not comments:
                    break
                
                # Get texts for batch prediction
                texts = [c.text_cleaned or c.text for c in comments]
                
                try:
                    predictions = self._best_model.predict(texts)
                    
                    for comment, pred in zip(comments, predictions):
                        comment.sentiment = SentimentLabel(pred.sentiment)
                        comment.confidence = pred.confidence
                        comment.sentiment_scores = pred.scores
                        comment.model_used = self._best_model.name
                        comment.analyzed_at = datetime.now()
                        
                        self.result.comments_analyzed += 1
                    
                except Exception as e:
                    logger.error(f"Error analyzing batch: {e}")
                    self.result.errors.append(f"Analysis error: {e}")
                
                offset += self.config.batch_size
                
                if offset >= self.config.max_comments_per_run:
                    break
    
    def _update_scores(self) -> None:
        """Update influencer scores based on analyzed comments."""
        with self.db.session_scope() as session:
            influencers = session.query(Influencer).filter_by(is_active=True).all()
            
            for influencer in influencers:
                self.db.update_influencer_scores(influencer.id)
                self.result.influencers_updated += 1
    
    def _update_rankings(self) -> None:
        """Update influencer rankings."""
        self.db.update_all_rankings()
    
    def _log_summary(self) -> None:
        """Log pipeline execution summary."""
        duration = None
        if self.result.started_at and self.result.completed_at:
            duration = (self.result.completed_at - self.result.started_at).total_seconds()
        
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Status: {self.result.status}")
        logger.info(f"Duration: {duration:.2f}s" if duration else "Duration: N/A")
        logger.info(f"Comments loaded: {self.result.comments_loaded}")
        logger.info(f"Comments cleaned: {self.result.comments_cleaned}")
        logger.info(f"Comments analyzed: {self.result.comments_analyzed}")
        logger.info(f"Influencers updated: {self.result.influencers_updated}")
        logger.info(f"Best model: {self.result.best_model}")
        
        if self.result.errors:
            logger.warning(f"Errors: {len(self.result.errors)}")
            for err in self.result.errors[:5]:
                logger.warning(f"  - {err}")
        
        logger.info("=" * 50)
    
    def analyze_single(self, text: str) -> PredictionResult:
        """
        Analyze a single text using the best model.
        
        Args:
            text: Text to analyze
            
        Returns:
            PredictionResult
        """
        if not self._best_model:
            # Try to load best model
            best_in_db = self.db.get_best_model()
            if best_in_db:
                self._best_model = ModelRegistry.get(best_in_db.model_name)
                model_path = self.config.models_dir / best_in_db.model_name
                if model_path.exists():
                    self._best_model.load(model_path)
            else:
                # Fallback to rule-based model
                self._best_model = ModelRegistry.get('rule_based')
        
        cleaned = self.cleaner.clean(text)
        return self._best_model.predict_single(cleaned)


class DataLoader:
    """
    Utility class for loading data from various sources.
    
    Supports loading from:
    - JSON files (scraped data)
    - CSV files
    - Database
    """
    
    @staticmethod
    def load_from_json(file_path: Path) -> List[Dict[str, Any]]:
        """Load comments from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_from_csv(file_path: Path, text_col: str = 'text',
                      label_col: str = 'label') -> List[Dict[str, Any]]:
        """Load comments from CSV file."""
        import pandas as pd
        
        df = pd.read_csv(file_path)
        return [
            {'text': row[text_col], 'label': row.get(label_col)}
            for _, row in df.iterrows()
        ]
    
    @staticmethod
    def load_from_db(db: DatabaseManager, 
                     influencer_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load comments from database."""
        with db.session_scope() as session:
            query = session.query(Comment)
            
            if influencer_id:
                query = query.filter_by(influencer_id=influencer_id)
            
            comments = query.all()
            return [c.to_dict() for c in comments]


def run_pipeline(config: Optional[PipelineConfig] = None,
                 skip_training: bool = False) -> PipelineResult:
    """
    Convenience function to run the analysis pipeline.
    
    Args:
        config: Pipeline configuration
        skip_training: Skip model training
        
    Returns:
        Pipeline execution result
    """
    pipeline = AnalysisPipeline(config)
    return pipeline.run(skip_training=skip_training)


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Run analysis pipeline')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training, use existing models')
    parser.add_argument('--db', type=str, default='database/db.sqlite3',
                        help='Database path')
    
    args = parser.parse_args()
    
    config = PipelineConfig(db_path=args.db)
    result = run_pipeline(config, skip_training=args.skip_training)
    
    print(json.dumps(result.to_dict(), indent=2))
