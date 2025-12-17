# CommentCourt 🏛️

A concise Turkish e-commerce influencer comment analysis framework.

Features:
- Multi-model sentiment analysis (BERT, RoBERTa, LSTM, LogisticRegression)
- Turkish preprocessing and influencer ranking (0–10)
- Flask web dashboard

Quick Start:
1. Install: `pip install -r requirements.txt`
2. Init DB: `python -c "from database.db import DatabaseManager; DatabaseManager().create_tables()"`
3. Run GUI: `python -m gui.app` (open `http://localhost:5000`)

Training example:
```py
from model.train import ModelTrainer
trainer = ModelTrainer()
# trainer.train_all_models(data)
```

Configuration: edit `utils/config` or use env vars `COMMENTCOURT_DB_PATH`, `COMMENTCOURT_MODEL_DEVICE`.

Development: `pytest`, `flake8`.

License: MIT

