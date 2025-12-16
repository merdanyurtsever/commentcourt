from backend.pipeline import AnalysisPipeline, PipelineConfig
from database.db import DatabaseManager


def test_pipeline_compute_trust(tmp_path):
    db_path = str(tmp_path / "test_trust.sqlite3")

    # Initialize DB
    db = DatabaseManager(db_path)
    db.create_tables()

    # Add influencer and comments
    influencer = db.add_influencer(name="Test", bio="")
    db.add_comment(influencer.id, text="Çok güzel!", original_rating=5.0)
    db.add_comment(influencer.id, text="Berbat ürün.", original_rating=1.0)

    config = PipelineConfig(db_path=db_path)
    pipeline = AnalysisPipeline(config)

    # Compute trust scores using fallback (rule-based)
    result = pipeline.compute_trust_scores(limit=10)
    assert result['processed'] == 2

    # Verify comments updated with trust_score
    with db.session_scope() as session:
        comments = session.query(type(session.query()._raw_columns[0].entity_type.__table__))

    # Instead of complex reflections, just inspect via helper
    comments = db.get_comments_by_influencer(influencer.id)
    assert len(comments) == 2
    for c in comments:
        assert c.sentiment_scores is not None
        assert 'trust_score' in c.sentiment_scores
        assert isinstance(c.sentiment_scores['trust_score'], float)
