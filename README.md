# CommentCourt 🏛️

**Turkish E-Commerce Influencer Comment Analysis Framework**

🇹🇷 [Türkçe README için tıklayın](README.tr.md)

A modular machine learning framework for analyzing product comments from Turkish e-commerce platforms and ranking influencers based on sentiment analysis.

---

## 🎯 Features

- **Multi-Model ML Framework**: 4 sentiment analysis models with automatic best-model selection
- **Turkish Language Support**: Optimized for Turkish text preprocessing and analysis
- **Influencer Ranking**: Weighted scoring algorithm (0-10 scale) with leaderboard
- **Web Dashboard**: Flask-based GUI for viewing influencers, comments, and analysis results
- **Modular Architecture**: Easy to add new models, modify database schema, or expand UI

---

## 🏗️ Architecture

```
commentcourt/
├── model/                    # ML Framework
│   ├── base.py              # Abstract base class for models
│   ├── registry.py          # Model registration with @register decorator
│   ├── sentiment.py         # 4 sentiment models (BERT, LR, LSTM, RoBERTa)
│   └── train.py             # Training & evaluation framework
├── gui/                    # Web Interface (Flask)
│   ├── app.py               # Flask application & routes
│   ├── templates/           # Jinja2 HTML templates
│   └── static/              # CSS & JavaScript
├── backend/                 # Core backend logic
│   ├── pipeline.py          # Analysis orchestration
│   ├── cleaner.py           # Text preprocessing
│   └── scorer.py            # Simplified influencer scoring
├── utils/                   # Utilities
│   ├── config.py            # YAML configuration loader
│   ├── io.py                # File I/O operations
│   └── log.py               # Logging setup
└── database/                # Database layer and data files
    ├── db.sqlite3           # SQLite database
    ├── raw/                 # Raw data (if any)
    └── processed/           # Processed datasets
└── database/
    ├── db.sqlite3           # SQLite database
    ├── raw/                 # Raw scraped data
    └── processed/           # Processed datasets
```

---

## 🤖 ML Models

| Model | Description | Best For |
|-------|-------------|----------|
| **BERT-Turkish** | `dbmdz/bert-base-turkish-cased` transformer | Highest accuracy |
| **RoBERTa** | `xlm-roberta-base` multilingual transformer | Cross-language support |
| **LSTM** | Bidirectional LSTM with attention | Sequence patterns |
| **LogisticRegression** | TF-IDF + Logistic Regression | Fast baseline |

The framework automatically evaluates all models and selects the best performer based on F1-score.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd commentcourt

# Install dependencies
pip install -r program/build/requirements.txt

# Initialize database
python -c "from database.db import DatabaseManager; DatabaseManager().create_tables()"
```

### Running the Web Interface

```bash
python -m gui.app
```

Then open http://localhost:5000 in your browser.

### Training Models

```python
from model.train import ModelTrainer
from model.registry import ModelRegistry

# Initialize trainer
trainer = ModelTrainer()

# Load your training data
data = [
    {"text": "Harika ürün, çok memnunum!", "label": "positive"},
    {"text": "Kötü kalite, para israfı.", "label": "negative"},
    # ... more samples
]

# Train all registered models
results = trainer.train_all_models(data)

# Get the best model
best_model = trainer.get_best_model()
print(f"Best model: {best_model.name} with F1: {best_model.metrics['f1']:.4f}")
```

### Running Analysis Pipeline

```python
from backend.pipeline import AnalysisPipeline

# Initialize pipeline
pipeline = AnalysisPipeline()

# Run full analysis
pipeline.run()

# Or analyze specific influencer
results = pipeline.analyze_influencer(influencer_id=1)
```

---

## 📊 Scoring Algorithm

Influencers are ranked on a 0-10 scale using weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Sentiment Score | 60% | Average sentiment of comments |
| Volume Score | 20% | Number of comments (normalized) |
| Consistency Score | 20% | Sentiment stability over time |

---

## ⚙️ Configuration

Edit `utils/config` (YAML format):

```yaml
database:
  path: database/db.sqlite3
  
models:
  default: bert-turkish
  device: auto  # cpu, cuda, or auto
  
analysis:
  batch_size: 32
  
gui:
  host: 0.0.0.0
  port: 5000
  debug: false
```

Environment variables override config values:
- `COMMENTCOURT_DB_PATH`
- `COMMENTCOURT_MODEL_DEVICE`
- `COMMENTCOURT_GUI_PORT`

---

## 🔌 Adding New Models

1. Create a new class inheriting from `BaseMLModel`:

```python
from model.base import BaseMLModel
from model.registry import ModelRegistry

@ModelRegistry.register("my-custom-model")
class MyCustomModel(BaseMLModel):
    def __init__(self):
        super().__init__(
            name="my-custom-model",
            version="1.0.0",
            description="My custom sentiment model"
        )
    
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        # Your prediction logic
        pass
    
    def train(self, data: List[Dict]) -> Dict:
        # Your training logic
        pass
```

2. The model is automatically registered and available in the framework.

---

## 📁 Database Schema

| Table | Description |
|-------|-------------|
| `influencers` | Influencer profiles and metadata |
| `comments` | Product comments with sentiment labels |
| `model_results` | Prediction results per model |
| `model_metrics` | Training/evaluation metrics |
| `influencer_scores` | Calculated rankings and scores |

---

## 🛠️ Development

```bash
# Run tests
pytest

# Check code style
flake8 model/ backend/ utils/ gui/

# Run with debug mode
FLASK_DEBUG=1 python -m gui.app
```

---

## 📜 License

MIT License

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See also: `docs/SIMPLIFIED.md` for details on the simplified code style and recent preprocessing/ui features.

---

*Built for Turkish e-commerce influencer analysis* 🇹🇷
