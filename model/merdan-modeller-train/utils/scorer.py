# ==============================
# 1. IMPORTS & SETUP
# ==============================
import pandas as pd
import numpy as np
import pickle
import os
import re
import unicodedata
from scipy.sparse import hstack, csr_matrix

# Dynamic Import for Turkish Stemmer
try:
    from TurkishStemmer import TurkishStemmer
    stemmer = TurkishStemmer()
except ImportError:
    class IdentityStemmer:
        def stem(self, w): return w
    stemmer = IdentityStemmer()

# Configuration
MODEL_PATH = 'model/merdan-modeller-train/weights/merdan_ultimate_ridge.pkl' 
DATA_PATH = "database/raw/Veri_Seti_Cleaned_v2.xlsx"
OUTPUT_FILE = "final_trust_report.txt"
NEGATION_WORDS = {'değil', 'yok', 'hiç', 'ama', 'fakat', 'asla', 'olmayan', 'hayır', 'ne'}

# ==============================
# 2. PREPROCESSING HELPERS (Must match training exactly)
# ==============================
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ''
    text = unicodedata.normalize('NFKC', text).replace('İ', 'i').replace('I', 'ı').lower()
    text = re.sub(r'http\S+|www\.[^\s]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zçğıöşü\s!:?\-\)\(\[\]\{\}\.,;"\'"\+\/\\]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_stemmed(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned: return ''
    tokens = cleaned.split()
    final = []
    for t in tokens:
        try: final.append(stemmer.stem(t))
        except: final.append(t)
    return ' '.join(final)

def preprocess_raw(text: str) -> str:
    return clean_text(text)

def extract_engineered_features(texts):
    feat_list = []
    for t in texts:
        s = t if isinstance(t, str) else ''
        char_len = len(s)
        tokens = s.split()
        token_count = len(tokens)
        exclam = s.count('!')
        question = s.count('?')
        uppers = sum(1 for ch in s if ch.isupper())
        upper_ratio = (uppers / char_len) if char_len > 0 else 0.0
        neg_flag = int(any(w in s.lower().split() for w in NEGATION_WORDS))
        emoticons = 0
        for emo in [':)', ':-)', ':(', ':-(', ':D', ':-D', ';)', ';-)', ':P', ':-P', ':/']:
            emoticons += s.count(emo)
        repeats = len([m.group(0) for m in re.finditer(r'(.)\1{2,}', s)])
        try: emoji_count = len([ch for ch in s if '\U0001F600' <= ch <= '\U0001F64F'])
        except: emoji_count = 0
        feat_list.append([char_len, token_count, exclam, question, upper_ratio, neg_flag, emoticons, repeats, emoji_count])
    return np.array(feat_list, dtype=float)

# ==============================
# 3. LOAD MODEL & DATA
# ==============================
print(f"Loading Model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH): raise FileNotFoundError("Model file not found!")

with open(MODEL_PATH, 'rb') as f:
    pipeline = pickle.load(f)

model = pipeline['model']
vec_stem = pipeline['vec_stem']
vec_raw = pipeline['vec_raw']
vec_char = pipeline['vec_char']

print(f"Loading Data from {DATA_PATH}...")
df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=["comment"])

# ==============================
# 4. PREDICT SENTIMENT
# ==============================
print("Predicting Sentiments...")
raw_texts = df["comment"].astype(str).values

# 3-View Transformation
X_stem = vec_stem.transform([preprocess_stemmed(t) for t in raw_texts])
X_raw_txt = [preprocess_raw(t) for t in raw_texts]
X_raw = vec_raw.transform(X_raw_txt)
X_char = vec_char.transform(X_raw_txt)
X_eng = extract_engineered_features(raw_texts)

X_final = hstack([X_stem, X_raw, X_char, csr_matrix(X_eng)], format='csr')

# Predict
preds = model.predict(X_final)
df["sentiment_score"] = np.clip(preds, 0.0, 1.0)

# ==============================
# 5. CALCULATE ROW-LEVEL TRUST SCORE
# ==============================
print("Calculating Individual Review Trust Scores...")

# Normalize Inputs
df["rating_norm"] = df["rating"] / 5.0
max_likes = df["like_count"].max()
df["like_norm"] = df["like_count"] / max_likes if max_likes > 0 else 0.0
df["has_images"] = df["has_images"].astype(bool)

# Define Trust Formula
def compute_trust(row):
    # Consistency: How well does text match rating?
    consistency = 1.0 - abs(row['sentiment_score'] - row['rating_norm'])
    
    trust = (
        0.35 * row['sentiment_score'] + # Content Positivity
        0.30 * row['rating_norm'] +     # Star Rating
        0.05 * (1.0 if row['has_images'] else 0.0) + # Image Evidence
        0.05 * row['like_norm'] +       # Social Proof
        0.20 * consistency +            # Logic/Truthfulness
        0.05 * (len(str(row['comment'])) / 500.0 if len(str(row['comment'])) < 500 else 1.0) # Effort
    )
    return round(float(trust), 4)

df["trust_score"] = df.apply(compute_trust, axis=1)

# ==============================
# 6. AGGREGATE: PRODUCT & INFLUENCER SCORES
# ==============================
print("Aggregating Scores...")

# A. Product Trust Score (Average of all reviews for that product)
product_scores = df.groupby('product_id')['trust_score'].mean().reset_index()
product_scores.columns = ['product_id', 'avg_product_trust']

# Merge product score back to main df so we know which product belongs to which influencer
# (Assuming 1 row per review, we need to map Product -> Influencer)
# If the dataset lists the influencer for every review:
product_influencer_map = df[['product_id', 'list_slug']].drop_duplicates()
product_scores = product_scores.merge(product_influencer_map, on='product_id', how='left')

# B. Influencer Trust Score (Average of the PRODUCT scores they promoted)
# We group by influencer and take the mean of their *products' averages*
influencer_scores = product_scores.groupby('list_slug')['avg_product_trust'].mean().reset_index()
influencer_scores.columns = ['list_slug', 'influencer_trust_score']

# Sort for reporting
product_scores = product_scores.sort_values('avg_product_trust', ascending=False)
influencer_scores = influencer_scores.sort_values('influencer_trust_score', ascending=False)

# ==============================
# 7. WRITE REPORT
# ==============================
print(f"Writing report to {OUTPUT_FILE}...")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("==================================================\n")
    f.write("          FİNAL GÜVEN SKORU RAPORU               \n")
    f.write("==================================================\n\n")
    
    f.write("--------------------------------------------------\n")
    f.write(f" EN İYİ İNFLUENCERLER (ortalama ürün güvenilirliği)\n")
    f.write("--------------------------------------------------\n")
    f.write(f"{'Influencer':<30} | {'Trust Score':<10}\n")
    f.write("-" * 45 + "\n")
    for _, row in influencer_scores.iterrows():
        f.write(f"{row['list_slug']:<30} | {row['influencer_trust_score']:.4f}\n")
    f.write("\n\n")

    f.write("--------------------------------------------------\n")
    f.write(f" ÜRÜN TRUST SCORELARI\n")
    f.write("--------------------------------------------------\n")
    f.write(f"{'Product':<50} | {'Influencer':<20} | {'Score':<10}\n")
    f.write("-" * 85 + "\n")
    for _, row in product_scores.iterrows():
        prod_name = (row['product_id'][:47] + '..') if len(str(row['product_id'])) > 47 else row['product_id']
        inf_name = (str(row['list_slug'])[:17] + '..') if len(str(row['list_slug'])) > 17 else str(row['list_slug'])
        f.write(f"{prod_name:<50} | {inf_name:<20} | {row['avg_product_trust']:.4f}\n")
    
print("Done! Report generated.")