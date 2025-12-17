import re
import unicodedata
import numpy as np
import pandas as pd
import os
import pickle

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import hstack, csr_matrix

# NLTK imports (handled dynamically in main, but good to have ready)
import nltk

# ---------- Config ----------
DATA_PATH = 'database/raw/Veri_Seti.xlsx'
MODEL_OUT = 'model/merdan-modeller-train/weights/merdan_classic_model.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF tuning
TFIDF_MAX_WORD_FEATURES = 20000
TFIDF_MAX_CHAR_FEATURES = 5000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.95

NEGATION_WORDS = {'değil', 'yok', 'hiç', 'ama', 'fakat', 'asla', 'olmayan'}


def detect_column(df, candidates):
    """Auto-detects column names based on a list of candidates."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def clean_text(text: str) -> str:
    """Basic string cleaning/normalization."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text)
    # Turkish-aware lowercase: handle dotted/dotless I
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    # remove urls, emails, mentions, hashtags
    text = re.sub(r'http\S+|www\.[^\s]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    # remove punctuation and numbers (keep Turkish letters and basic punctuation)
    text = re.sub(r'[^a-zçğıöşü\s!:?\-\)\(\[\]\{\}\.,;"\'"\+\/\\]', ' ', text)
    # collapse repeated characters (e.g., cooool -> coool)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str, stop_words_list=None, stemmer=None) -> str:
    """Tokenizes and optionally stems/removes stop words."""
    cleaned = clean_text(text)
    if cleaned == '':
        return ''
    tokens = cleaned.split()
    if stop_words_list:
        tokens = [t for t in tokens if t not in stop_words_list]
    if stemmer:
        try:
            # stemmer may be a SnowballStemmer or a callable lemmatizer
            tokens = [stemmer.stem(t) if hasattr(stemmer, 'stem') else stemmer(t) for t in tokens]
        except Exception:
            pass
    return ' '.join(tokens)


def extract_engineered_features(texts):
    """Extracts meta-features like punctuation counts, emoji counts, etc."""
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
        neg_flag = int(any(w in s.split() for w in NEGATION_WORDS))
        
        emoticons = 0
        for emo in [':)', ':-)', ':(', ':-(', ':D', ':-D', ';)', ';-)', ':P', ':-P']:
            emoticons += s.count(emo)
            
        repeats = len([m.group(0) for m in re.finditer(r'(.)\1{2,}', s)])
        
        # Emoji detection (approximate unicode ranges)
        try:
            emoji_count = len([ch for ch in s if '\U0001F600' <= ch <= '\U0001F64F' or '\U0001F300' <= ch <= '\U0001F5FF'])
        except Exception:
            emoji_count = sum(s.count(c) for c in ['❤', '😂', '😊', '😢', '👍', '😡'])
            
        feat_list.append([char_len, token_count, exclam, question, upper_ratio, neg_flag, emoticons, repeats, emoji_count])
        
    return np.array(feat_list, dtype=float), names


def derive_score(val, num_min=None, num_max=None):
    """Normalizes scores to a 0.0 - 1.0 float range."""
    try:
        num = float(val)
        if num_min is None or num_max is None or num_max == num_min:
            return None
        # Min-Max Scaling
        return float((num - num_min) / (num_max - num_min))
    except Exception:
        s = str(val).strip().lower()
        if s in ['positive', 'pos', 'pozitif', '1']:
            return 1.0
        if s in ['negative', 'neg', 'negatif', '0']:
            return 0.0
        if s in ['neutral', 'nötr', 'nötral', 'orta']:
            return 0.5
        return None


def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)

    comment_col = detect_column(df, ['comment', 'yorum', 'text', 'content', 'message', 'review'])
    rating_col = detect_column(df, ['rating', 'puan', 'score', 'stars', 'label'])

    if comment_col is None:
        raise ValueError(f"Could not find a comment/text column in {list(df.columns)}")

    if rating_col is None:
        # Fallback if specific label column isn't found
        if 'label' in df.columns:
            rating_col = 'label'
        else:
            raise ValueError(f"Could not find a rating/score column in {list(df.columns)}")

    df = df.dropna(subset=[comment_col])
    # Keep original text for feature engineering before cleaning
    raw_comments = df[comment_col].astype(str).values 

    # 2. Normalize Scores
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = None if numeric_ratings.dropna().empty else float(numeric_ratings.min())
    num_max = None if numeric_ratings.dropna().empty else float(numeric_ratings.max())

    df['score'] = df[rating_col].apply(lambda v: derive_score(v, num_min=num_min, num_max=num_max))
    df = df.dropna(subset=['score'])
    df['score'] = df['score'].astype(float)
    
    # Update raw_comments to match the dropped rows
    raw_comments = df[comment_col].astype(str).values

    print(f"Data loaded. Samples: {len(df)}")

    # 3. Setup NLP Tools (Stopwords / Stemmer / Zemberek)
    stop_words = None
    stemmer = None
    zemberek_lemmatizer = None
    
    try:
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        from nltk.corpus import stopwords as nltk_stopwords
        stop_words = list(nltk_stopwords.words('turkish'))
        
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer('turkish')
    except Exception as e:
        print(f"NLTK setup warning (using fallbacks): {e}")
        stop_words = ['ve', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'mi', 'ne', 'ama', 'çok', 'gibi']

    # Optional Zemberek setup
    try:
        from zemberek import TurkishMorphology
        tm = TurkishMorphology.create_with_defaults()

        def zemberek_lemma_tokenize(s):
            try:
                parses = tm.analyze_sentence(s)
                lemmas = []
                for word_parses in parses:
                    if word_parses.analysis_results:
                        lemmas.append(word_parses.analysis_results[0].dictionary_item.lemma)
                    else:
                        lemmas.append(word_parses.surface)
                return ' '.join(lemmas)
            except Exception:
                return s

        zemberek_lemmatizer = zemberek_lemma_tokenize
        print("Zemberek loaded successfully.")
    except ImportError:
        print("Zemberek not found. skipping...")
        zemberek_lemmatizer = None

    # 4. Preprocess Text
    print("Preprocessing texts...")
    if zemberek_lemmatizer:
        X_texts_pre = [zemberek_lemmatizer(clean_text(t)) for t in raw_comments]
    else:
        X_texts_pre = [preprocess_text(t, stop_words_list=stop_words, stemmer=stemmer) for t in raw_comments]

    # 5. Vectorization (Word + Char)
    print("Vectorizing...")
    word_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_WORD_FEATURES, 
        stop_words=None, 
        ngram_range=(1, 2), 
        min_df=TFIDF_MIN_DF, 
        max_df=TFIDF_MAX_DF, 
        sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char', 
        ngram_range=(3, 5), 
        max_features=TFIDF_MAX_CHAR_FEATURES, 
        sublinear_tf=True
    )
    
    X_word = word_vectorizer.fit_transform(X_texts_pre)
    X_char = char_vectorizer.fit_transform(X_texts_pre)

    # 6. Engineered Features
    print("Extracting engineered features...")
    X_eng, eng_names = extract_engineered_features(raw_comments)
    X_eng_sparse = csr_matrix(X_eng)

    # 7. Stack Features
    X = hstack([X_word, X_char, X_eng_sparse], format='csr')
    y = df['score'].values

    # 8. Train/Test Split & Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Training Ridge Regression...")
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred_test = reg.predict(X_test)

    # 9. Evaluation
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    try:
        corr = np.corrcoef(y_test, y_pred_test)[0, 1]
    except Exception:
        corr = float('nan')

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Pearson correlation: {corr:.6f}")

    # 10. Save Outputs
    out_dir = os.path.dirname(MODEL_OUT) or '.'
    os.makedirs(out_dir, exist_ok=True)

    report_path = os.path.join(out_dir, 'merdan_classic_regression_report.txt')
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write(f"MSE: {mse:.6f}\n")
        rf.write(f"MAE: {mae:.6f}\n")
        rf.write(f"R2: {r2:.6f}\n")
        rf.write(f"Pearson correlation: {corr:.6f}\n\n")
        rf.write(f"Test size: {len(y_test)}\n")

    print(f"Saved regression report to {report_path}")

    # Save Pipeline Dictionary
    pipeline_out = {
        'model': reg,
        'word_vectorizer': word_vectorizer,
        'char_vectorizer': char_vectorizer,
        'eng_feature_names': eng_names,
        'stop_words': stop_words,
        'zemberek': bool(zemberek_lemmatizer)
    }
    
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(pipeline_out, f)

    print(f"Regression model and vectorizers saved as {MODEL_OUT}")

    # 11. Interpretability (Top Coefficients)
    try:
        coef = reg.coef_
        w_feats = list(word_vectorizer.get_feature_names_out())
        c_feats = list(char_vectorizer.get_feature_names_out())
        feat_names = w_feats + c_feats + eng_names
        
        # Sort coefficients
        top_idx = np.argsort(coef)[-20:][::-1] # Top positive
        bot_idx = np.argsort(coef)[:20]        # Top negative (lowest)

        fi_path = os.path.join(out_dir, 'merdan_classic_top_features.txt')
        with open(fi_path, 'w', encoding='utf-8') as ff:
            ff.write('Top positive features (Predicts Higher Score):\n')
            for i in top_idx:
                if i < len(feat_names):
                    ff.write(f"{feat_names[i]}\t{coef[i]:.6f}\n")
            
            ff.write('\nTop negative features (Predicts Lower Score):\n')
            for i in bot_idx:
                if i < len(feat_names):
                    ff.write(f"{feat_names[i]}\t{coef[i]:.6f}\n")
                    
        print(f"Saved top features to {fi_path}")
    except Exception as e:
        print(f"Could not save feature importance: {e}")


if __name__ == '__main__':
    main()