import re
import unicodedata
import numpy as np
import pandas as pd
import os
import pickle

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import hstack, csr_matrix

# Dynamic Import for Turkish Stemmer
try:
    from TurkishStemmer import TurkishStemmer
    stemmer = TurkishStemmer()
    print("✓ TurkishStemmer loaded successfully")
except ImportError:
    print("! TurkishStemmer not found. Installing fallback...")
    # Fallback to simple identity if not found, but user should install it
    class IdentityStemmer:
        def stem(self, w): return w
    stemmer = IdentityStemmer()

# ---------- Config ----------
DATA_PATH = 'database/raw/Veri_Seti.xlsx'
MODEL_OUT = 'model/merdan-modeller-train/weights/merdan_ridge_old.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF tuning (Aggressive culling to reduce noise)
TFIDF_MAX_WORD_FEATURES = 10000  # Reduced to focus on strong signals
TFIDF_MAX_CHAR_FEATURES = 5000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.90

# Critical Negation Words (Do NOT remove these as stop words)
NEGATION_WORDS = {'değil', 'yok', 'hiç', 'ama', 'fakat', 'asla', 'olmayan', 'hayır', 'ne'}

def detect_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    
    # Remove URL/Mentions
    text = re.sub(r'http\S+|www\.[^\s]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Keep only Turkish letters and essential punctuation for engineered features
    text = re.sub(r'[^a-zçğıöşü\s!:?\-\)\(\[\]\{\}\.,;"\'"\+\/\\]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text: str, stop_words_list=None) -> str:
    """ cleans + stems + removes stopwords """
    cleaned = clean_text(text)
    if not cleaned:
        return ''
    
    tokens = cleaned.split()
    
    # 1. Stemming first (crucial for matching stopwords correctly if they are stemmed)
    # Actually for Turkish, it's safer to check stopwords raw, then stem content
    final_tokens = []
    for t in tokens:
        # Skip stop words (unless they are negation words)
        if stop_words_list and t in stop_words_list and t not in NEGATION_WORDS:
            continue
        # Stem
        try:
            stemmed = stemmer.stem(t)
            final_tokens.append(stemmed)
        except:
            final_tokens.append(t)
            
    return ' '.join(final_tokens)

def extract_engineered_features(texts):
    feat_list = []
    names = ['char_len', 'token_count', 'exclam_count', 'question_count', 'upper_ratio', 'negation', 'emoticon_count', 'repeat_char_count', 'emoji_count']
    
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
        
        try:
            emoji_count = len([ch for ch in s if '\U0001F600' <= ch <= '\U0001F64F'])
        except:
            emoji_count = 0
            
        feat_list.append([char_len, token_count, exclam, question, upper_ratio, neg_flag, emoticons, repeats, emoji_count])
        
    return np.array(feat_list, dtype=float), names

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
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)

    comment_col = detect_column(df, ['comment', 'yorum', 'text', 'content'])
    rating_col = detect_column(df, ['rating', 'puan', 'score', 'label'])

    if not comment_col or not rating_col:
        raise ValueError("Could not find required columns.")

    df = df.dropna(subset=[comment_col])
    raw_comments = df[comment_col].astype(str).values 

    # Normalize Scores
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = None if numeric_ratings.dropna().empty else float(numeric_ratings.min())
    num_max = None if numeric_ratings.dropna().empty else float(numeric_ratings.max())

    df['score'] = df[rating_col].apply(lambda v: derive_score(v, num_min=num_min, num_max=num_max))
    df = df.dropna(subset=['score'])
    df['score'] = df['score'].astype(float)
    
    # Sync raw_comments
    raw_comments = df[comment_col].astype(str).values

    print(f"Data loaded. Samples: {len(df)}")

    # Prepare Stop Words (NLTK)
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    from nltk.corpus import stopwords as nltk_stopwords
    stop_words = list(nltk_stopwords.words('turkish'))

    # Preprocess
    print("Preprocessing texts (with TurkishStemmer)...")
    X_texts_pre = [preprocess_text(t, stop_words_list=stop_words) for t in raw_comments]

    # Vectorization
    print("Vectorizing...")
    # Word: Unigrams & Bigrams
    word_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_WORD_FEATURES, 
        stop_words=None, # Already handled manually to protect negation
        ngram_range=(1, 2), 
        min_df=TFIDF_MIN_DF, 
        max_df=TFIDF_MAX_DF, 
        sublinear_tf=True
    )
    # Char: 3-5 grams (Captures roots even if stemming fails)
    char_vectorizer = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(3, 5), 
        max_features=TFIDF_MAX_CHAR_FEATURES, 
        sublinear_tf=True
    )
    
    X_word = word_vectorizer.fit_transform(X_texts_pre)
    X_char = char_vectorizer.fit_transform(X_texts_pre)

    print("Extracting engineered features...")
    X_eng, eng_names = extract_engineered_features(raw_comments)
    X_eng_sparse = csr_matrix(X_eng)

    # Stack
    X = hstack([X_word, X_char, X_eng_sparse], format='csr')
    y = df['score'].values

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Grid Search for best Alpha
    print("Tuning Ridge Regression (GridSearch)...")
    param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Alpha found: {grid.best_params_['alpha']}")

    # Evaluation
    y_pred_test = best_model.predict(X_test)
    # Clip predictions to 0-1 range (Ridge can overshoot)
    y_pred_test = np.clip(y_pred_test, 0.0, 1.0)

    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    corr = np.corrcoef(y_test, y_pred_test)[0, 1]

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Pearson correlation: {corr:.6f}")

    # Save
    out_dir = os.path.dirname(MODEL_OUT) or '.'
    os.makedirs(out_dir, exist_ok=True)
    
    # Save Report
    with open(os.path.join(out_dir, 'merdan_classic_regression_report.txt'), 'w', encoding='utf-8') as rf:
        rf.write(f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nR2: {r2:.6f}\nPearson: {corr:.6f}\n")

    # Save Pickle
    pipeline_out = {
        'model': best_model,
        'word_vectorizer': word_vectorizer,
        'char_vectorizer': char_vectorizer,
        'eng_feature_names': eng_names
    }
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(pipeline_out, f)
    print(f"Saved optimized model to {MODEL_OUT}")

if __name__ == '__main__':
    main()