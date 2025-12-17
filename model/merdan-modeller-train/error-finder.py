import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error

# Config
DATA_PATH = 'database/raw/Veri_Seti.xlsx'
MODEL_PATH = 'model/merdan-modeller-train/weights/merdan_classic_model.pkl'

# Helper (Must match your training logic)
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
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    
    model = pipeline['model']
    word_vec = pipeline['word_vectorizer']
    char_vec = pipeline['char_vectorizer']
    # If you saved a custom preprocessor function, you'd need it here, 
    # but for inspection we can re-use the raw text flow if the vectorizers handle it.
    
    # 2. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    
    # Auto-detect columns (simplified from your main script)
    comment_col = next(c for c in df.columns if c.lower() in ['comment', 'yorum', 'text', 'content'])
    rating_col = next(c for c in df.columns if c.lower() in ['rating', 'puan', 'score', 'label'])
    
    df = df.dropna(subset=[comment_col])
    
    # Normalize Truth Labels
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = float(numeric_ratings.min())
    num_max = float(numeric_ratings.max())
    
    df['true_score'] = df[rating_col].apply(lambda v: derive_score(v, num_min, num_max))
    df = df.dropna(subset=['true_score'])
    
    # 3. Predict
    # Note: We must replicate the exact preprocessing logic used in training
    # For this diagnostic, we will rely on the vectorizer's internal analyzer 
    # if it was saved with the pipeline. 
    # IF NOT, you might need to copy-paste the 'clean_text' and 'preprocess_text' 
    # functions here again. assuming standard raw text input for now:
    
    # We need to re-import the custom stemmer fallback just in case pickle needs it
    try:
        from TurkishStemmer import TurkishStemmer
    except ImportError:
        class TurkishStemmer:
            def stem(self, w): return w

    print("Transforming data...")
    # NOTE: In your training script, you stemmed the text BEFORE vectorizing.
    # We need to do that here too or the vectorizer sees different words.
    # To keep this script simple, I'll assume you can copy the 'preprocess_text' function 
    # from your training script.
    
    # *** INSERT PREPROCESS_TEXT FUNCTION HERE FROM YOUR TRAINING SCRIPT ***
    # (I will assume raw text for now to show you the logic, but results will be weak without stemming)
    # The clean way:
    from merdan_classic import preprocess_text, extract_engineered_features, clean_text
    
    # Preprocess
    texts = [preprocess_text(t) for t in df[comment_col].astype(str).values]
    
    # Vectorize
    from scipy.sparse import hstack, csr_matrix
    X_word = word_vec.transform(texts)
    X_char = char_vec.transform(texts)
    X_eng, _ = extract_engineered_features(df[comment_col].astype(str).values)
    
    X = hstack([X_word, X_char, csr_matrix(X_eng)], format='csr')
    
    # Predict
    df['pred_score'] = model.predict(X)
    df['pred_score'] = df['pred_score'].clip(0, 1) # Clip to valid range
    
    # 4. Analyze Errors
    df['error'] = abs(df['true_score'] - df['pred_score'])
    
    # Sort by worst errors
    worst_fails = df.sort_values('error', ascending=False).head(20)
    
    print("\n" + "="*80)
    print("TOP 20 WORST PREDICTIONS (High Error)")
    print("="*80)
    print(f"{'True':<6} | {'Pred':<6} | {'Diff':<6} | {'Text'}")
    print("-" * 80)
    
    for _, row in worst_fails.iterrows():
        txt = row[comment_col].replace('\n', ' ')[:80] # Truncate for display
        print(f"{row['true_score']:.2f}   | {row['pred_score']:.2f}   | {row['error']:.2f}   | {txt}")
        
    print("-" * 80)
    print("If you see 'Great Product' with True Score 0.20, your dataset has labeling errors.")
    print("If you see sarcasm, that's a limitation of the model.")

if __name__ == '__main__':
    main()