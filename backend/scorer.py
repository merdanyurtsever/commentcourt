"""Simple influencer scoring utilities.

This module provides a compact, easy-to-read scoring function used by
the pipeline to compute influencer scores from comment sentiments.
"""

from typing import List, Dict


class InfluencerScorer:
    """Minimal scorer: final_score = scale * normalized(positive - 0.5*negative)."""

    def __init__(self, scale: int = 10):
        self.scale = scale

    def calculate_score(self, influencer_id: int, influencer_name: str, comments: List[Dict]) -> Dict:
        """Compute a compact score summary for an influencer.

        comments: list of dicts with keys: 'sentiment' (str) and optional 'confidence'.
        Returns a small dict with final_score (0..scale) and counts.
        """
        total = len(comments)
        positive = sum(1 for c in comments if c.get('sentiment') == 'positive')
        negative = sum(1 for c in comments if c.get('sentiment') == 'negative')
        neutral = sum(1 for c in comments if c.get('sentiment') == 'neutral')

        if total == 0:
            return {
                'influencer_id': influencer_id,
                'influencer_name': influencer_name,
                'total_comments': 0,
                'final_score': 0.0
            }

        raw = (positive - 0.5 * negative) / total  # simple weighted mean

        # Map raw from [-0.5,1] to [0,1]
        raw_min, raw_max = -0.5, 1.0
        normalized = max(0.0, min(1.0, (raw - raw_min) / (raw_max - raw_min)))
        final_score = round(normalized * self.scale, 2)

        return {
            'influencer_id': influencer_id,
            'influencer_name': influencer_name,
            'total_comments': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'final_score': final_score
        }
"""
Influencer Scoring System for CommentCourt.

This module calculates and ranks influencer scores based on sentiment and other metrics.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import math


logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for influencer scoring."""
    
    # Score scale
    scale: int = 10  # Score out of 10 (or 5)
    
    # Sentiment weights
    positive_weight: float = 1.0
    negative_weight: float = -0.5
    neutral_weight: float = 0.1
    
    # Confidence weighting
    use_confidence_weighting: bool = True
    min_confidence_threshold: float = 0.5
    
    # Volume considerations
    min_comments_for_ranking: int = 5
    volume_bonus_max: float = 0.5  # Max bonus for high volume
    volume_bonus_threshold: int = 100  # Comments needed for max bonus
    
    # Temporal weighting (recent comments worth more)
    use_temporal_weighting: bool = True
    temporal_decay_days: int = 90  # Half-life in days
    
    # Normalization
    normalize_scores: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scale': self.scale,
            'positive_weight': self.positive_weight,
            'negative_weight': self.negative_weight,
            'neutral_weight': self.neutral_weight,
            'use_confidence_weighting': self.use_confidence_weighting,
            'min_confidence_threshold': self.min_confidence_threshold,
            'min_comments_for_ranking': self.min_comments_for_ranking,
            'volume_bonus_max': self.volume_bonus_max,
            'volume_bonus_threshold': self.volume_bonus_threshold,
            'use_temporal_weighting': self.use_temporal_weighting,
            'temporal_decay_days': self.temporal_decay_days,
            'normalize_scores': self.normalize_scores
        }


@dataclass
class CommentData:
    """Simplified comment data for scoring."""
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class InfluencerScore:
    """Calculated score for an influencer."""
    
    influencer_id: int
    influencer_name: str
    
    # Raw metrics
    total_comments: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    
    # Ratios
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    neutral_ratio: float = 0.0
    
    # Calculated scores
    raw_score: float = 0.0
    weighted_score: float = 0.0
    final_score: float = 0.0  # Normalized to scale
    
    # Additional metrics
    avg_confidence: float = 0.0
    trend_direction: str = "stable"  # improving, declining, stable
    trend_magnitude: float = 0.0
    
    # Ranking
    rank: Optional[int] = None
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'influencer_id': self.influencer_id,
            'influencer_name': self.influencer_name,
            'total_comments': self.total_comments,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'neutral_count': self.neutral_count,
            'positive_ratio': round(self.positive_ratio, 3),
            'negative_ratio': round(self.negative_ratio, 3),
            'neutral_ratio': round(self.neutral_ratio, 3),
            'raw_score': round(self.raw_score, 3),
            'weighted_score': round(self.weighted_score, 3),
            'final_score': round(self.final_score, 2),
            'avg_confidence': round(self.avg_confidence, 3),
            'trend_direction': self.trend_direction,
            'trend_magnitude': round(self.trend_magnitude, 3),
            'rank': self.rank,
            'calculated_at': self.calculated_at.isoformat()
        }


class InfluencerScorer:
    """
    Calculates and ranks influencer scores.
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None, scale: int = 10):
        """
        Initialize scorer.
        """
        if config is None:
            config = ScoringConfig(scale=scale)
        
        self.config = config
        self._all_scores: List[InfluencerScore] = []
    
    def calculate_score(self, influencer_id: int, influencer_name: str,
                        comments: List[CommentData]) -> InfluencerScore:
        """
        Calculate score for a single influencer.
        """
        score = InfluencerScore(
            influencer_id=influencer_id,
            influencer_name=influencer_name
        )
        
        if not comments:
            return score
        
        # Filter by confidence if needed
        if self.config.use_confidence_weighting:
            valid_comments = [
                c for c in comments
                if c.confidence >= self.config.min_confidence_threshold
            ]
        else:
            valid_comments = comments
        
        if not valid_comments:
            return score
        
        # Count sentiments
        score.total_comments = len(valid_comments)
        score.positive_count = sum(1 for c in valid_comments if c.sentiment == 'positive')
        score.negative_count = sum(1 for c in valid_comments if c.sentiment == 'negative')
        score.neutral_count = sum(1 for c in valid_comments if c.sentiment == 'neutral')
        
        # Calculate ratios
        total = score.total_comments
        score.positive_ratio = score.positive_count / total
        score.negative_ratio = score.negative_count / total
        score.neutral_ratio = score.neutral_count / total
        
        # Calculate raw score
        score.raw_score = self._calculate_raw_score(valid_comments)
        
        # Apply confidence weighting
        if self.config.use_confidence_weighting:
            score.weighted_score = self._apply_confidence_weighting(valid_comments)
            score.avg_confidence = sum(c.confidence for c in valid_comments) / len(valid_comments)
        else:
            score.weighted_score = score.raw_score
        
        # Apply volume bonus
        volume_bonus = self._calculate_volume_bonus(score.total_comments)
        score.weighted_score += volume_bonus
        
        # Calculate trend
        if self.config.use_temporal_weighting:
            trend_dir, trend_mag = self._calculate_trend(valid_comments)
            score.trend_direction = trend_dir
            score.trend_magnitude = trend_mag
        
        # Normalize to scale
        score.final_score = self._normalize_score(score.weighted_score)
        
        return score
    
    def _calculate_raw_score(self, comments: List[CommentData]) -> float:
        """Calculate raw score from sentiment counts."""
        if not comments:
            return 0.0
        
        positive = sum(1 for c in comments if c.sentiment == 'positive')
        negative = sum(1 for c in comments if c.sentiment == 'negative')
        neutral = sum(1 for c in comments if c.sentiment == 'neutral')
        
        weighted_sum = (
            positive * self.config.positive_weight +
            negative * self.config.negative_weight +
            neutral * self.config.neutral_weight
        )
        
        # Normalize by total
        return weighted_sum / len(comments)
    
    def _apply_confidence_weighting(self, comments: List[CommentData]) -> float:
        """Apply confidence weighting to scores."""
        if not comments:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for comment in comments:
            weight = comment.confidence
            
            if comment.sentiment == 'positive':
                value = self.config.positive_weight
            elif comment.sentiment == 'negative':
                value = self.config.negative_weight
            else:
                value = self.config.neutral_weight
            
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_volume_bonus(self, comment_count: int) -> float:
        """Calculate bonus for high comment volume."""
        if comment_count < self.config.min_comments_for_ranking:
            return 0.0
        
        # Logarithmic scaling for volume bonus
        ratio = min(comment_count / self.config.volume_bonus_threshold, 1.0)
        bonus = self.config.volume_bonus_max * math.log1p(ratio * 10) / math.log1p(10)
        
        return bonus
    
    def _calculate_trend(self, comments: List[CommentData]) -> tuple:
        """
        Calculate sentiment trend over time.
        
        Returns:
            Tuple of (direction, magnitude)
        """
        # Filter comments with timestamps
        dated_comments = [c for c in comments if c.timestamp]
        
        if len(dated_comments) < 10:
            return ("stable", 0.0)
        
        # Sort by date
        dated_comments.sort(key=lambda c: c.timestamp)
        
        # Split into halves
        mid = len(dated_comments) // 2
        first_half = dated_comments[:mid]
        second_half = dated_comments[mid:]
        
        # Calculate sentiment score for each half
        first_score = self._calculate_raw_score(first_half)
        second_score = self._calculate_raw_score(second_half)
        
        diff = second_score - first_score
        
        if diff > 0.1:
            return ("improving", abs(diff))
        elif diff < -0.1:
            return ("declining", abs(diff))
        else:
            return ("stable", abs(diff))
    
    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize score to configured scale.
        """
        # Map from [-0.5, 1.0] to [0, scale]
        min_raw = self.config.negative_weight
        max_raw = self.config.positive_weight
        
        # Clamp
        raw_score = max(min_raw, min(max_raw, raw_score))
        
        # Normalize to [0, 1]
        normalized = (raw_score - min_raw) / (max_raw - min_raw)
        
        # Scale
        final = normalized * self.config.scale
        
        return round(final, 2)
    
    def calculate_all_scores(self, influencer_comments: Dict[tuple, List[CommentData]]) -> List[InfluencerScore]:
        """
        Calculate scores for multiple influencers.
        """
        scores = []
        
        for (inf_id, inf_name), comments in influencer_comments.items():
            score = self.calculate_score(inf_id, inf_name, comments)
            scores.append(score)
        
        # Sort by final score
        scores.sort(key=lambda s: s.final_score, reverse=True)
        
        # Assign ranks
        for rank, score in enumerate(scores, start=1):
            score.rank = rank
        
        self._all_scores = scores
        return scores
    
    def get_rankings(self) -> List[Dict[str, Any]]:
        """
        Get ranked list of influencers.
        """
        if not self._all_scores:
            return []
        
        return [
            {
                'rank': s.rank,
                'name': s.influencer_name,
                'score': s.final_score,
                'total_comments': s.total_comments,
                'positive_ratio': round(s.positive_ratio * 100, 1),
                'trend': s.trend_direction
            }
            for s in self._all_scores
        ]
    
    def get_top_influencers(self, n: int = 10) -> List[InfluencerScore]:
        """Get top n influencers by score."""
        return self._all_scores[:n]
    
    def get_bottom_influencers(self, n: int = 10) -> List[InfluencerScore]:
        """Get bottom n influencers by score."""
        return self._all_scores[-n:]
    
    def get_improving_influencers(self) -> List[InfluencerScore]:
        """Get influencers with improving trends."""
        return [s for s in self._all_scores if s.trend_direction == 'improving']
