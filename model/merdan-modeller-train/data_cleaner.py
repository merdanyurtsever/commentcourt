import pandas as pd
import numpy as np
import pickle
import os
import re
import unicodedata
from scipy.sparse import hstack, csr_matrix

# ---------- Config ----------
INPUT_PATH = 'database/raw/Veri_Seti_Cleaned.xlsx' # Input the ALREADY cleaned file
OUTPUT_PATH = 'database/raw/Veri_Seti_Cleaned_v2.xlsx' # Output version 2
MODEL_PATH = 'model/merdan-modeller-train/weights/merdan_ultimate_ridge.pkl'

# Threshold: Keep it high (0.6) to only fix obvious errors
CONFIDENCE_THRESHOLD = 0.6 
NEGATION_WORDS = {'değil', 'yok', 'hiç', 'ama', 'fakat', 'asla', 'olmayan', 'hayır', 'ne'}

# ---------- Preprocessing Logic (Must match merdan_ridge.py EXACTLY) ----------

# Dynamic Import for Turkish Stemmer
try:
    from TurkishStemmer import TurkishStemmer
    stemmer = TurkishStemmer()
except ImportError:
    class IdentityStemmer:
        def stem(self, w): return w
    stemmer = IdentityStemmer()

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
    # Stopwords usually handled by vectorizer, but for stemming loop we just stem
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

def derive_score(val, num_min=None, num_max=None):
    try:
        num = float(val)
        if num_min is None or num_max is None or num_max == num_min: return None
        return float((num - num_min) / (num_max - num_min))
    except:
        s = str(val).strip().lower()
        if s in ['positive', 'pos', 'pozitif', '1']: return 1.0
        if s in ['negative', 'neg', 'negatif', '0']: return 0.0
        if s in ['neutral', 'nötr', 'nötral', 'orta']: return 0.5
        return None

def main():
    print("Step 1: Loading Data and Ultimate Model...")
    if not os.path.exists(INPUT_PATH):
        print(f"File {INPUT_PATH} not found!")
        return

    df = pd.read_excel(INPUT_PATH)
    
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    
    model = pipeline['model']
    vec_stem = pipeline['vec_stem']
    vec_raw = pipeline['vec_raw']
    vec_char = pipeline['vec_char']
    
    # Detect Columns
    comment_col = next(c for c in df.columns if c.lower() in ['comment', 'yorum', 'text', 'content'])
    rating_col = next(c for c in df.columns if c.lower() in ['rating', 'puan', 'score', 'label'])
    
    df = df.dropna(subset=[comment_col])
    
    # Normalize Truth Labels
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = float(numeric_ratings.min())
    num_max = float(numeric_ratings.max())
    
    # We create a temporary normalized score column for comparison
    df['temp_norm_score'] = df[rating_col].apply(lambda v: derive_score(v, num_min, num_max))
    df = df.dropna(subset=['temp_norm_score'])
    
    print("Step 2: Transforming Data (3-View Pipeline)...")
    
    raw_texts = df[comment_col].astype(str).values
    
    # 1. Stemmed View
    X_stem_txt = [preprocess_stemmed(t) for t in raw_texts]
    X_vec_stem = vec_stem.transform(X_stem_txt)
    
    # 2. Raw View
    X_raw_txt = [preprocess_raw(t) for t in raw_texts]
    X_vec_raw = vec_raw.transform(X_raw_txt)
    
    # 3. Char View
    X_vec_char = vec_char.transform(X_raw_txt)
    
    # 4. Engineered Features
    X_eng = extract_engineered_features(raw_texts)
    
    # Stack
    X = hstack([X_vec_stem, X_vec_raw, X_vec_char, csr_matrix(X_eng)], format='csr')
    
    print("Step 3: Predicting and Fixing...")
    df['predicted_score'] = model.predict(X)
    df['predicted_score'] = df['predicted_score'].clip(0, 1)
    
    def corrector(row):
        true_s = row['temp_norm_score']
        pred_s = row['predicted_score']
        diff = abs(true_s - pred_s)
        
        # Iterative cleaning: We trust the Ultimate Model
        if diff > CONFIDENCE_THRESHOLD:
            return pred_s, True
        return true_s, False
    
    results = df.apply(corrector, axis=1, result_type='expand')
    df['new_clean_score'] = results[0]
    df['was_fixed_v2'] = results[1]
    
    # Convert back to Star Rating
    df['clean_rating_v2'] = df['new_clean_score'] * (num_max - num_min) + num_min
    df['clean_rating_v2'] = df['clean_rating_v2'].round().astype(int)
    
    fixed_count = df['was_fixed_v2'].sum()
    print(f"\nTotal Rows: {len(df)}")
    print(f"Rows Fixed in Round 2: {fixed_count} ({(fixed_count/len(df))*100:.2f}%)")
    
    if fixed_count > 0:
        print("\nExamples of Round 2 Fixes:")
        print("-" * 60)
        examples = df[df['was_fixed_v2'] == True].head(10)
        for _, row in examples.iterrows():
            print(f"Was: {row[rating_col]} -> Now: {row['clean_rating_v2']} | Text: {str(row[comment_col])[:50]}...")
        print("-" * 60)
    
    # Update and Save
    df['rating_v1'] = df[rating_col] # Keep history
    df[rating_col] = df['clean_rating_v2']
    
    # Cleanup cols
    save_df = df.drop(columns=['temp_norm_score', 'predicted_score', 'new_clean_score', 'was_fixed_v2', 'clean_rating_v2'])
    
    save_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSaved Version 2 Dataset to: {OUTPUT_PATH}")
    print("Now run 'merdan_ridge.py' using this new V2 file!")

if __name__ == '__main__':
    main()