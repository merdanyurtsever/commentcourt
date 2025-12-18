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
    print("! TurkishStemmer not found. Please pip install TurkishStemmer")
    class IdentityStemmer:
        def stem(self, w): return w
    stemmer = IdentityStemmer()

# ---------- Config ----------
DATA_PATH = 'database/raw/Veri_Seti_Cleaned_v2.xlsx' # Use the cleaned file if you have it!
MODEL_OUT = 'model/merdan-modeller-train/weights/merdan_ultimate_ridge.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 1. STEMMED View Config (Focus: Topic)
# We only need unigrams here to know "this is about a phone" vs "this is about cargo"
STEM_MAX_FEATURES = 20000

# 2. RAW View Config (Focus: Sentiment & Phrase Context)
# We capture "hiç beğenmedim" etc. here
RAW_MAX_FEATURES = 50000
RAW_NGRAM_RANGE = (1, 4)

# 3. CHAR View Config (Focus: Morphology/Suffixes)
# Captures "-medi", "-miyor" even if words are unique
CHAR_MAX_FEATURES = 40000
CHAR_NGRAM_RANGE = (2, 8) # Expanded range to catch longer suffixes

NEGATION_WORDS = {'değil', 'yok', 'hiç', 'ama', 'fakat', 'asla', 'olmayan', 'hayır', 'ne'}

def detect_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower: return cols_lower[cand]
    return None

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ''
    text = unicodedata.normalize('NFKC', text).replace('İ', 'i').replace('I', 'ı').lower()
    text = re.sub(r'http\S+|www\.[^\s]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # Keep punctuation that might imply sentiment
    text = re.sub(r'[^a-zçğıöşü\s!:?\-\)\(\[\]\{\}\.,;"\'"\+\/\\]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_stemmed(text: str, stop_words_list=None) -> str:
    """ Aggressively stems everything. Good for Topic detection. """
    cleaned = clean_text(text)
    if not cleaned: return ''
    tokens = cleaned.split()
    final = []
    for t in tokens:
        # Don't remove negation words even in stemmed view
        if stop_words_list and t in stop_words_list and t not in NEGATION_WORDS:
            continue
        try:
            final.append(stemmer.stem(t))
        except:
            final.append(t)
    return ' '.join(final)

def preprocess_raw(text: str) -> str:
    """ Minimal cleaning. Preserves suffixes. Good for Sentiment detection. """
    # No stemming! No stopword removal!
    return clean_text(text)

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
        # Fallback to original if cleaned doesn't exist
        print(f"Cleaned data not found. Falling back to original.")
        f_path = 'database/raw/Veri_Seti.xlsx'
    else:
        f_path = DATA_PATH
        
    print(f"Loading data from {f_path}...")
    df = pd.read_excel(f_path)

    comment_col = detect_column(df, ['comment', 'yorum', 'text', 'content'])
    rating_col = detect_column(df, ['rating', 'puan', 'score', 'label'])

    df = df.dropna(subset=[comment_col])
    raw_comments = df[comment_col].astype(str).values 

    # Normalize Scores
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = float(numeric_ratings.min())
    num_max = float(numeric_ratings.max())

    df['score'] = df[rating_col].apply(lambda v: derive_score(v, num_min, num_max))
    df = df.dropna(subset=['score'])
    df['score'] = df['score'].astype(float)
    
    raw_comments = df[comment_col].astype(str).values

    print(f"Data loaded. Samples: {len(df)}")

    # Stopwords
    import nltk
    try: nltk.data.find('corpora/stopwords')
    except: nltk.download('stopwords')
    from nltk.corpus import stopwords as nltk_stopwords
    stop_words = list(nltk_stopwords.words('turkish'))

    # --- THE DUAL VIEW PIPELINE ---

    # 1. Stemmed Text (For general topic grouping)
    print("Preprocessing View 1: Stemmed Text...")
    X_stem = [preprocess_stemmed(t, stop_words_list=stop_words) for t in raw_comments]
    
    vec_stem = TfidfVectorizer(
        max_features=STEM_MAX_FEATURES,
        ngram_range=(1, 1), # Only unigrams for stemmed
        min_df=5,
        max_df=0.90,
        sublinear_tf=True
    )
    X_vec_stem = vec_stem.fit_transform(X_stem)

    # 2. Raw Text (For exact phrases like 'hiç beğenmedim')
    print("Preprocessing View 2: Raw Text (Preserving Suffixes)...")
    X_raw = [preprocess_raw(t) for t in raw_comments]
    
    vec_raw = TfidfVectorizer(
        max_features=RAW_MAX_FEATURES,
        ngram_range=RAW_NGRAM_RANGE, # Unigrams, Bigrams, Trigrams
        min_df=3,
        max_df=0.90,
        sublinear_tf=True
    )
    X_vec_raw = vec_raw.fit_transform(X_raw)

    # 3. Char N-Grams (For morphological hints)
    print("Preprocessing View 3: Char N-Grams...")
    vec_char = TfidfVectorizer(
        analyzer='char',
        ngram_range=CHAR_NGRAM_RANGE, # Expanded to (3,6)
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True
    )
    # We use X_raw here because char n-grams on stemmed text are useless
    X_vec_char = vec_char.fit_transform(X_raw)

    # 4. Engineered Features
    print("Extracting engineered features...")
    X_eng, eng_names = extract_engineered_features(raw_comments)
    X_eng_sparse = csr_matrix(X_eng)

    # STACK EVERYTHING
    print("Stacking features...")
    # This matrix is now very wide, covering every angle
    X = hstack([X_vec_stem, X_vec_raw, X_vec_char, X_eng_sparse], format='csr')
    y = df['score'].values

    print(f"Final Feature Matrix Shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Grid Search
    print("Tuning Ridge Regression (GridSearch)...")
    # 'auto' solver chooses the fastest based on data shape (likely 'sag' or 'sparse_cg')
    param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 'solver': ['auto']}
    grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"Best Alpha: {grid.best_params_['alpha']}")

    # Eval
    y_pred = best_model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0) # Clip results

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    print(f"\n--- ULTIMATE RIDGE RESULTS ---")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Pearson: {corr:.6f}")

    # Save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'vec_stem': vec_stem,
            'vec_raw': vec_raw,
            'vec_char': vec_char,
            'eng_names': eng_names
        }, f)
    print(f"Saved model to {MODEL_OUT}")

if __name__ == '__main__':
    main()