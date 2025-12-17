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

    @staticmethod
    def load_from_excel(file_path: Path, text_col: str = 'comment', rating_col: str = 'rating',
                        drop_edges: bool = True) -> List[Dict[str, Any]]:
        """Load comments from an Excel file and infer labels from ratings.

        Simple, explicit rules:
         - rating >= 4 => 'positive'
         - rating <= 2 => 'negative'
         - otherwise => 'neutral'
         - edge cases (missing text or missing rating) are flagged and can be dropped
        """
        import pandas as pd
        from backend.preprocessor import Preprocessor

        df = pd.read_excel(file_path)

        samples = []
        pre = Preprocessor()

        # prefer columns that match common names
        if text_col not in df.columns:
            # try common alternatives
            for alt in ('text', 'comment_text', 'yorum'):
                if alt in df.columns:
                    text_col = alt
                    break

        if rating_col not in df.columns:
            for alt in ('rating', 'puan', 'stars'):
                if alt in df.columns:
                    rating_col = alt
                    break

        for _, row in df.iterrows():
            raw_text = row.get(text_col)
            rating = row.get(rating_col)

            if raw_text is None or not str(raw_text).strip():
                if drop_edges:
                    continue
                label = 'neutral'
                edge = True
            else:
                edge = False
                cleaned = pre.preprocess(str(raw_text))

                # Use rating as proxy label if present and numeric
                try:
                    r = float(rating)
                    if r >= 4.0:
                        label = 'positive'
                    elif r <= 2.0:
                        label = 'negative'
                    else:
                        label = 'neutral'
                except Exception:
                    # No numeric rating: fallback to neutral but mark edge
                    label = 'neutral'
                    edge = True

            samples.append({'text': cleaned if not edge else (str(raw_text) or ''),
                            'label': label,
                            'edge_case': edge})

        return samples

    @staticmethod
    def prepare_training_from_excel(file_path: Path, out_path: Path) -> None:
        """Read Excel and write a processed training JSON to out_path."""
        samples = DataLoader.load_from_excel(file_path)
        # Filter out edge cases by default
        filtered = [s for s in samples if not s.get('edge_case')]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump([{'text': s['text'], 'label': s['label']} for s in filtered], f, ensure_ascii=False, indent=2)

        return
