"""Data loading helpers for the pipeline.

This module centralizes data-loading utilities that were previously
embedded in `backend/pipeline.py`.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from database.db import DatabaseManager, Comment


class DataLoader:
    """Utility class for loading data from various sources.

    Supports loading from:
    - JSON files
    - CSV files
    - Database
    - Exporting a training dataset JSON
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

    @staticmethod
    def export_training_dataset(db: DatabaseManager, out_path: Path) -> None:
        """Export comments with sentiment labels to a training JSON file.

        The output format is a list of objects:
            {'text': '...', 'label': 'positive'|'negative'|'neutral'}
        """
        samples = []
        with db.session_scope() as session:
            comments = session.query(Comment).filter(Comment.sentiment.isnot(None)).all()

            for c in comments:
                # comment.sentiment may be enum-like; try to get a string
                label = getattr(c.sentiment, 'value', None) or str(c.sentiment)
                text = c.text_cleaned or c.text or ""

                if label:
                    samples.append({'text': text, 'label': label})

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        return
