# CommentCourt 🏛️

Kısa ve öz Türk e-ticaret influencer yorum analiz çerçevesi.

Özellikler:
- Çoklu model duygu analizi (BERT, RoBERTa, LSTM, Lojistik Regresyon)
- Türkçe ön işleme ve influencer sıralama (0–10)
- Flask tabanlı gösterge paneli

Hızlı Başlangıç:
1. Yükle: `pip install -r requirements.txt`
2. DB: `python -c "from database.db import DatabaseManager; DatabaseManager().create_tables()"`
3. GUI: `python -m gui.app` (aç `http://localhost:5000`)

Eğitim örneği:
```py
from model.train import ModelTrainer
trainer = ModelTrainer()
# trainer.train_all_models(data)
```

Yapılandırma: `utils/config` veya `COMMENTCOURT_DB_PATH`, `COMMENTCOURT_MODEL_DEVICE`.

Geliştirme: `pytest`, `flake8`.

Lisans: MIT
