"""
Flask Web Application for CommentCourt.

This module provides a web GUI for:
- Viewing influencer rankings and scores
- Browsing influencer profiles and comments
- Running analysis pipeline
- Model management
- Configuration
- Multi-language support (Turkish/English)

Routes:
- / : Dashboard with top influencers
- /influencers : List all influencers
- /influencer/<id> : Influencer detail page
- /analysis : Analysis controls
- /api/* : REST API endpoints
- /language/<code> : Change language
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, flash, session
)

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from program.utils.config import get_config, load_config
from program.utils.db import get_db, init_db, Influencer, Comment, SentimentLabel
from program.core.pipeline import AnalysisPipeline, PipelineConfig
from program.utils.i18n import i18n, t, get_language, set_language, get_languages
from model.registry import ModelRegistry, auto_discover_models


logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Load configuration
config = load_config()
app.secret_key = config.gui.secret_key


# =============================================================================
# Template Filters
# =============================================================================

@app.template_filter('format_score')
def format_score(score: float) -> str:
    """Format score for display."""
    if score is None:
        return "N/A"
    return f"{score:.1f}"


@app.template_filter('format_percent')
def format_percent(value: float) -> str:
    """Format ratio as percentage."""
    if value is None:
        return "0%"
    return f"{value * 100:.1f}%"


@app.template_filter('sentiment_color')
def sentiment_color(sentiment: str) -> str:
    """Get color class for sentiment."""
    colors = {
        'positive': 'success',
        'negative': 'danger',
        'neutral': 'secondary'
    }
    return colors.get(sentiment, 'secondary')


@app.template_filter('trend_icon')
def trend_icon(trend: str) -> str:
    """Get icon for trend."""
    icons = {
        'improving': '↑',
        'declining': '↓',
        'stable': '→'
    }
    return icons.get(trend, '→')


# =============================================================================
# Context Processors
# =============================================================================

@app.context_processor
def inject_globals():
    """Inject global variables into templates."""
    # Get language from session or use default
    lang = session.get('language', get_language())
    set_language(lang)
    
    return {
        'app_name': config.app_name,
        'score_scale': config.scoring.scale,
        'current_year': datetime.now().year,
        't': t,  # Translation function
        'current_language': lang,
        'languages': get_languages()
    }


# =============================================================================
# Language Route
# =============================================================================

@app.route('/language/<lang_code>')
def change_language(lang_code: str):
    """Change the current language."""
    if lang_code in get_languages():
        session['language'] = lang_code
        set_language(lang_code)
        flash(t('common.language_changed') if lang_code == 'tr' else 'Language changed', 'success')
    
    # Redirect back to the referring page or home
    return redirect(request.referrer or url_for('dashboard'))


# =============================================================================
# Web Routes
# =============================================================================

@app.route('/')
def dashboard():
    """Dashboard with top influencers and summary stats."""
    db = get_db()
    
    # Get statistics
    stats = db.get_statistics()
    
    # Get top influencers
    top_influencers = db.get_top_influencers(limit=10)
    
    # Get model comparison
    model_comparison = db.get_model_comparison()
    
    return render_template('dashboard.html',
                           stats=stats,
                           top_influencers=top_influencers,
                           model_comparison=model_comparison)


@app.route('/influencers')
def influencers_list():
    """List all influencers with pagination."""
    db = get_db()
    
    page = request.args.get('page', 1, type=int)
    per_page = config.gui.items_per_page
    sort_by = request.args.get('sort', 'rank')
    
    with db.session_scope() as session:
        query = session.query(Influencer).filter_by(is_active=True)
        
        # Sorting
        if sort_by == 'score':
            query = query.order_by(Influencer.overall_score.desc())
        elif sort_by == 'name':
            query = query.order_by(Influencer.name.asc())
        elif sort_by == 'comments':
            query = query.order_by(Influencer.total_comments.desc())
        else:  # rank
            query = query.order_by(Influencer.rank.asc().nullslast())
        
        # Pagination
        total = query.count()
        influencers = query.offset((page - 1) * per_page).limit(per_page).all()
        
        total_pages = (total + per_page - 1) // per_page
    
    return render_template('influencers.html',
                           influencers=influencers,
                           page=page,
                           total_pages=total_pages,
                           sort_by=sort_by)


@app.route('/influencer/<int:influencer_id>')
def influencer_detail(influencer_id: int):
    """Influencer detail page."""
    db = get_db()
    
    with db.session_scope() as session:
        influencer = session.query(Influencer).filter_by(id=influencer_id).first()
        
        if not influencer:
            flash('Influencer not found', 'error')
            return redirect(url_for('influencers_list'))
        
        # Get recent comments
        comments = session.query(Comment)\
            .filter_by(influencer_id=influencer_id)\
            .order_by(Comment.created_at.desc())\
            .limit(50)\
            .all()
        
        # Calculate sentiment breakdown
        sentiment_counts = {
            'positive': sum(1 for c in comments if c.sentiment == SentimentLabel.POSITIVE),
            'negative': sum(1 for c in comments if c.sentiment == SentimentLabel.NEGATIVE),
            'neutral': sum(1 for c in comments if c.sentiment == SentimentLabel.NEUTRAL)
        }
    
    return render_template('influencer_detail.html',
                           influencer=influencer,
                           comments=comments,
                           sentiment_counts=sentiment_counts)


@app.route('/analysis')
def analysis_page():
    """Analysis control page."""
    auto_discover_models()
    models = ModelRegistry.list_models()
    
    db = get_db()
    
    # Get recent analysis runs
    with db.session_scope() as session:
        from program.utils.db import AnalysisRun
        recent_runs = session.query(AnalysisRun)\
            .order_by(AnalysisRun.started_at.desc())\
            .limit(10)\
            .all()
    
    return render_template('analysis.html',
                           models=models,
                           recent_runs=recent_runs)


@app.route('/analysis/run', methods=['POST'])
def run_analysis():
    """Trigger analysis pipeline."""
    skip_training = request.form.get('skip_training', 'true') == 'true'
    
    try:
        pipeline_config = PipelineConfig(
            db_path=config.database.path,
            batch_size=config.pipeline.batch_size
        )
        
        pipeline = AnalysisPipeline(pipeline_config)
        result = pipeline.run(skip_training=skip_training)
        
        if result.status == 'completed':
            flash(f'Analysis completed. {result.comments_analyzed} comments analyzed.', 'success')
        else:
            flash(f'Analysis failed: {result.errors}', 'error')
        
    except Exception as e:
        logger.exception("Analysis error")
        flash(f'Error running analysis: {str(e)}', 'error')
    
    return redirect(url_for('analysis_page'))


@app.route('/comments')
def comments_list():
    """Browse comments."""
    db = get_db()
    
    page = request.args.get('page', 1, type=int)
    per_page = config.gui.items_per_page
    sentiment_filter = request.args.get('sentiment', None)
    influencer_filter = request.args.get('influencer', None, type=int)
    
    with db.session_scope() as session:
        query = session.query(Comment)
        
        if sentiment_filter:
            query = query.filter_by(sentiment=SentimentLabel(sentiment_filter))
        
        if influencer_filter:
            query = query.filter_by(influencer_id=influencer_filter)
        
        query = query.order_by(Comment.created_at.desc())
        
        total = query.count()
        comments = query.offset((page - 1) * per_page).limit(per_page).all()
        
        total_pages = (total + per_page - 1) // per_page
        
        # Get influencers for filter dropdown
        influencers = session.query(Influencer).filter_by(is_active=True).all()
    
    return render_template('comments.html',
                           comments=comments,
                           influencers=influencers,
                           page=page,
                           total_pages=total_pages,
                           sentiment_filter=sentiment_filter,
                           influencer_filter=influencer_filter)


# =============================================================================
# API Routes
# =============================================================================

@app.route('/api/influencers')
def api_influencers():
    """API: Get all influencers."""
    db = get_db()
    influencers = db.get_all_influencers()
    return jsonify([i.to_dict() for i in influencers])


@app.route('/api/influencer/<int:influencer_id>')
def api_influencer(influencer_id: int):
    """API: Get single influencer."""
    db = get_db()
    
    with db.session_scope() as session:
        influencer = session.query(Influencer).filter_by(id=influencer_id).first()
        
        if not influencer:
            return jsonify({'error': 'Not found'}), 404
        
        return jsonify(influencer.to_dict())


@app.route('/api/influencer/<int:influencer_id>/comments')
def api_influencer_comments(influencer_id: int):
    """API: Get influencer's comments."""
    db = get_db()
    
    limit = request.args.get('limit', 100, type=int)
    
    with db.session_scope() as session:
        comments = session.query(Comment)\
            .filter_by(influencer_id=influencer_id)\
            .order_by(Comment.created_at.desc())\
            .limit(limit)\
            .all()
        
        return jsonify([c.to_dict() for c in comments])


@app.route('/api/stats')
def api_stats():
    """API: Get statistics."""
    db = get_db()
    return jsonify(db.get_statistics())


@app.route('/api/models')
def api_models():
    """API: Get available models."""
    auto_discover_models()
    return jsonify(ModelRegistry.list_models())


@app.route('/api/model_comparison')
def api_model_comparison():
    """API: Get model comparison."""
    db = get_db()
    return jsonify(db.get_model_comparison())


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API: Analyze text."""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Text required'}), 400
    
    text = data['text']
    
    try:
        pipeline_config = PipelineConfig(db_path=config.database.path)
        pipeline = AnalysisPipeline(pipeline_config)
        
        result = pipeline.analyze_single(text)
        
        return jsonify(result.to_dict())
    
    except Exception as e:
        logger.exception("Analysis error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rankings')
def api_rankings():
    """API: Get influencer rankings."""
    db = get_db()
    
    limit = request.args.get('limit', 20, type=int)
    
    top = db.get_top_influencers(limit=limit)
    
    rankings = []
    for inf in top:
        rankings.append({
            'rank': inf.rank,
            'id': inf.id,
            'name': inf.name,
            'score': round(inf.overall_score, 2),
            'total_comments': inf.total_comments,
            'positive_ratio': round(inf.positive_ratio * 100, 1)
        })
    
    return jsonify(rankings)


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.exception("Internal server error")
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


# =============================================================================
# CLI Commands
# =============================================================================

@app.cli.command('init-db')
def init_database():
    """Initialize database."""
    init_db(config.database.path)
    print("Database initialized.")


@app.cli.command('run-analysis')
def run_analysis_cli():
    """Run analysis pipeline."""
    pipeline_config = PipelineConfig(db_path=config.database.path)
    pipeline = AnalysisPipeline(pipeline_config)
    result = pipeline.run()
    print(f"Analysis {result.status}. Processed {result.comments_analyzed} comments.")


# =============================================================================
# Main
# =============================================================================

def create_app(config_path: Optional[str] = None) -> Flask:
    """
    Application factory.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Flask application instance
    """
    global config
    
    if config_path:
        config = load_config(Path(config_path))
        app.secret_key = config.gui.secret_key
    
    # Initialize database
    init_db(config.database.path)
    
    # Auto-discover models
    auto_discover_models()
    
    return app


def run_server(host: str = None, port: int = None, debug: bool = None):
    """
    Run the Flask development server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    host = host or config.gui.host
    port = port or config.gui.port
    debug = debug if debug is not None else config.gui.debug
    
    # Initialize database
    init_db(config.database.path)
    
    logger.info(f"Starting CommentCourt server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run CommentCourt web server')
    parser.add_argument('--host', type=str, default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug)
