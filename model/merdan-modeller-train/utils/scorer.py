 # ==============================
# 1. KÜTÜPHANELER
# ==============================
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 2. EĞİTİLMİŞ MODELİ YÜKLE
# ==============================
MODEL_PATH = "/model/merdan-modeller-train/weights/merdan_ultimate_ridge"  # <- EĞİTTİĞİN MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

print("Eğitilmiş sentiment modeli yüklendi.")

# ==============================
# 3. VERİYİ OKU (ETİKETSİZ)
# ==============================
df = pd.read_excel("Rabia_Veri_Seti.xlsx")

df = df.dropna(subset=["comment"])

# ==============================
# 4. TRUST SCORE İÇİN ÖN İŞLEME
# ==============================
df["rating_norm"] = df["rating"] / 5.0

if df["like_count"].max() > 0:
    df["like_norm"] = df["like_count"] / df["like_count"].max()
else:
    df["like_norm"] = 0.0

df["has_images"] = df["has_images"].astype(bool)

# 🔹 YENİ: Yorum uzunluğu (normalize)
df["comment_length"] = df["comment"].astype(str).apply(len)

if df["comment_length"].max() > 0:
    df["comment_length_norm"] = df["comment_length"] / df["comment_length"].max()
else:
    df["comment_length_norm"] = 0.0

# ==============================
# 5. SENTIMENT TAHMİNİ
# ==============================
sentiments = []

with torch.no_grad():
    for text in df["comment"].tolist():
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        output = model(**enc)
        sentiment = torch.sigmoid(output.logits).item()
        sentiments.append(sentiment)

df["sentiment_label"] = sentiments
df["predicted_class"] = (df["sentiment_label"] > 0.5).astype(int)

# ==============================
# 6. TRUST SCORE FORMÜLÜ (GÜNCELLENDİ)
# ==============================
def compute_trust_score(
    sentiment,
    rating_norm,
    has_image,
    like_norm,
    comment_length_norm
):
    image_score = 1.0 if has_image else 0.0
    uyum = 1 - abs(sentiment - rating_norm)

    trust = (
        0.40 * sentiment +
        0.35 * rating_norm +
        0.05 * image_score +
        0.05 * like_norm +
        0.10 * uyum +
        0.05 * comment_length_norm
    )
    return round(float(trust), 4)

df["trust_score"] = df.apply(
    lambda x: compute_trust_score(
        x["sentiment_label"],
        x["rating_norm"],
        x["has_images"],
        x["like_norm"],
        x["comment_length_norm"]
    ),
    axis=1
)

# ==============================
# 7. ÇIKTI (CSV KALDIRILDI)
# ==============================
# Artık trust_score verisi web uygulamasına df üzerinden direkt kullanılabilir

print("Sentiment ve Trust Score başarıyla üretildi.")