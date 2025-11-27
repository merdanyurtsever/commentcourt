# CommentCourt 🏛️

**Türk E-Ticaret Influencer Yorum Analiz Çerçevesi**

🇬🇧 [Click here for English README](README.md)

Türk e-ticaret platformlarından ürün yorumlarını analiz eden ve duygu analizine dayalı influencer sıralaması yapan modüler bir makine öğrenmesi çerçevesi.

---

## 🎯 Özellikler

- **Çoklu Model ML Çerçevesi**: Otomatik en iyi model seçimi ile 4 duygu analizi modeli
- **Türkçe Dil Desteği**: Türkçe metin ön işleme ve analiz için optimize edilmiş
- **Influencer Sıralaması**: Ağırlıklı puanlama algoritması (0-10 ölçeği) ile liderlik tablosu
- **Web Arayüzü**: Influencer'ları, yorumları ve analiz sonuçlarını görüntülemek için Flask tabanlı GUI
- **Modüler Mimari**: Yeni modeller eklemek, veritabanı şemasını değiştirmek veya arayüzü genişletmek kolay

---

## 🏗️ Mimari

```
commentcourt/
├── model/                    # ML Çerçevesi
│   ├── base.py              # Modeller için soyut temel sınıf
│   ├── registry.py          # @register dekoratörü ile model kaydı
│   ├── sentiment.py         # 4 duygu modeli (BERT, LR, LSTM, RoBERTa)
│   └── train.py             # Eğitim ve değerlendirme çerçevesi
├── program/
│   ├── core/                # Çekirdek İşleme
│   │   ├── pipeline.py      # Analiz orkestrasyonu
│   │   ├── cleaner.py       # Türkçe metin ön işleme
│   │   ├── scorer.py        # Influencer puanlama algoritması
│   │   └── scraper.py       # Veri toplama (harici)
│   ├── gui/                 # Web Arayüzü
│   │   ├── app.py           # Flask uygulaması
│   │   ├── templates/       # Jinja2 HTML şablonları
│   │   └── static/          # CSS ve JavaScript
│   ├── utils/               # Yardımcı Araçlar
│   │   ├── db.py            # SQLAlchemy ORM modelleri
│   │   ├── config.py        # YAML yapılandırma yükleyici
│   │   ├── io.py            # Dosya I/O işlemleri
│   │   └── log.py           # Günlük kaydı kurulumu
│   └── build/
│       ├── requirements.txt # Python bağımlılıkları
│       └── Dockerfile       # Konteyner oluşturma
└── database/
    ├── db.sqlite3           # SQLite veritabanı
    ├── raw/                 # Ham kazınmış veri
    └── processed/           # İşlenmiş veri setleri
```

---

## 🤖 ML Modelleri

| Model | Açıklama | En İyi Kullanım |
|-------|----------|-----------------|
| **BERT-Turkish** | `dbmdz/bert-base-turkish-cased` transformer | En yüksek doğruluk |
| **RoBERTa** | `xlm-roberta-base` çok dilli transformer | Çapraz dil desteği |
| **LSTM** | Dikkat mekanizmalı çift yönlü LSTM | Dizi kalıpları |
| **LogisticRegression** | TF-IDF + Lojistik Regresyon | Hızlı temel çizgi |

Çerçeve, tüm modelleri otomatik olarak değerlendirir ve F1-skora göre en iyi performans göstereni seçer.

---

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Depoyu klonlayın
cd commentcourt

# Bağımlılıkları yükleyin
pip install -r program/build/requirements.txt

# Veritabanını başlatın
python -c "from program.utils.db import DatabaseManager; DatabaseManager().create_tables()"
```

### Web Arayüzünü Çalıştırma

```bash
python -m program.gui.app
```

Ardından tarayıcınızda http://localhost:5000 adresini açın.

### Model Eğitimi

```python
from model.train import ModelTrainer
from model.registry import ModelRegistry

# Eğiticiyi başlat
trainer = ModelTrainer()

# Eğitim verinizi yükleyin
data = [
    {"text": "Harika ürün, çok memnunum!", "label": "positive"},
    {"text": "Kötü kalite, para israfı.", "label": "negative"},
    # ... daha fazla örnek
]

# Tüm kayıtlı modelleri eğit
results = trainer.train_all_models(data)

# En iyi modeli al
best_model = trainer.get_best_model()
print(f"En iyi model: {best_model.name}, F1: {best_model.metrics['f1']:.4f}")
```

### Analiz Pipeline'ını Çalıştırma

```python
from program.core.pipeline import AnalysisPipeline

# Pipeline'ı başlat
pipeline = AnalysisPipeline()

# Tam analiz çalıştır
pipeline.run_full_analysis()

# Veya belirli bir influencer'ı analiz et
results = pipeline.analyze_influencer(influencer_id=1)
```

---

## 📊 Puanlama Algoritması

Influencer'lar ağırlıklı faktörler kullanılarak 0-10 ölçeğinde sıralanır:

| Faktör | Ağırlık | Açıklama |
|--------|---------|----------|
| Duygu Skoru | %60 | Yorumların ortalama duygu değeri |
| Hacim Skoru | %20 | Yorum sayısı (normalize edilmiş) |
| Tutarlılık Skoru | %20 | Zaman içinde duygu kararlılığı |

---

## ⚙️ Yapılandırma

`program/utils/config` dosyasını düzenleyin (YAML formatı):

```yaml
database:
  path: database/db.sqlite3
  
models:
  default: bert-turkish
  device: auto  # cpu, cuda veya auto
  
analysis:
  batch_size: 32
  
gui:
  host: 0.0.0.0
  port: 5000
  debug: false
  language: tr  # tr veya en
```

Ortam değişkenleri yapılandırma değerlerini geçersiz kılar:
- `COMMENTCOURT_DB_PATH`
- `COMMENTCOURT_MODEL_DEVICE`
- `COMMENTCOURT_GUI_PORT`
- `COMMENTCOURT_LANGUAGE`

---

## 🔌 Yeni Model Ekleme

1. `BaseMLModel`'den miras alan yeni bir sınıf oluşturun:

```python
from model.base import BaseMLModel
from model.registry import ModelRegistry

@ModelRegistry.register("benim-ozel-modelim")
class BenimOzelModelim(BaseMLModel):
    def __init__(self):
        super().__init__(
            name="benim-ozel-modelim",
            version="1.0.0",
            description="Benim özel duygu modelim"
        )
    
    def predict(self, texts: List[str]) -> List[PredictionResult]:
        # Tahmin mantığınız
        pass
    
    def train(self, data: List[Dict]) -> Dict:
        # Eğitim mantığınız
        pass
```

2. Model otomatik olarak kaydedilir ve çerçevede kullanılabilir hale gelir.

---

## 📁 Veritabanı Şeması

| Tablo | Açıklama |
|-------|----------|
| `influencers` | Influencer profilleri ve meta verileri |
| `comments` | Duygu etiketli ürün yorumları |
| `model_results` | Model başına tahmin sonuçları |
| `model_metrics` | Eğitim/değerlendirme metrikleri |
| `influencer_scores` | Hesaplanan sıralamalar ve puanlar |

---

## 🛠️ Geliştirme

```bash
# Testleri çalıştır
pytest

# Kod stilini kontrol et
flake8 model/ program/

# Debug modunda çalıştır
FLASK_DEBUG=1 python -m program.gui.app
```

---

## 📜 Lisans

MIT Lisansı

---

## 🤝 Katkıda Bulunma

1. Depoyu fork'layın
2. Bir özellik dalı oluşturun
3. Değişikliklerinizi yapın
4. Pull request gönderin

---

*Türk e-ticaret influencer analizi için geliştirildi* 🇹🇷
