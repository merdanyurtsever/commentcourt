"""
This script trains a classical machine learning model (Ridge Regression) for sentiment analysis on Turkish text data.

It performs the following steps:
1.  Loads a dataset from an Excel file.
2.  Cleans and preprocesses the text data, with optional support for NLTK and Zemberek.
3.  Engineers features from the text (TF-IDF on words and characters, plus meta-features).
4.  Trains a Ridge Regression model on the engineered features.
5.  Evaluates the model and saves the performance report.
6.  Saves the trained model, vectorizers, and other pipeline components to a pickle file.
7.  Performs basic model interpretability by saving the most influential features.

Required libraries:
- pandas
- scikit-learn
- numpy
- nltk
- openpyxl
- zemberek-python (optional, for advanced lemmatization)
"""
import re
import unicodedata
import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import hstack, csr_matrix

# NLTK imports (handled dynamically)
import nltk

# ---------- File Paths (relative to this script's location) ----------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[2] # Assumes script is in model/merdan-modeller-train/
DATA_PATH = PROJECT_ROOT / 'database/raw/Veri_Seti.xlsx'
WEIGHTS_DIR = SCRIPT_DIR / 'weights'
MODEL_OUT = WEIGHTS_DIR / 'merdan_classic_model.pkl'
REPORT_PATH = WEIGHTS_DIR / 'merdan_classic_regression_report.txt'
FEATURES_PATH = WEIGHTS_DIR / 'merdan_classic_top_features.txt'


# ---------- Model Configuration ----------
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
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def clean_text(text: str) -> str:
    """Basic string cleaning/normalization for Turkish."""
    if not isinstance(text, str):
        return ''
    # Turkish-aware lowercase
    text = text.replace('I', 'ı').replace('İ', 'i').lower()
    text = unicodedata.normalize('NFKC', text)
    # Remove URLs, emails, mentions, hashtags
    text = re.sub(r'https://?[\w./?=%&-]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    # Remove punctuation and numbers, keeping essential Turkish chars
    text = re.sub(r'[^a-zçğıöşü\s]', ' ', text)
    # Collapse repeated characters (e.g., cooool -> cool)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str, stop_words_list=None, stemmer=None) -> str:
    """Tokenizes and optionally stems/removes stop words."""
    cleaned = clean_text(text)
    if not cleaned:
        return ''
    tokens = cleaned.split()
    if stop_words_list:
        tokens = [t for t in tokens if t not in stop_words_list]
    if stemmer:
        try:
            tokens = [stemmer.stem(t) for t in tokens]
        except (AttributeError, TypeError) as e:
            print(f"Warning: Stemmer failed for a token. Details: {e}")
            pass # Continue without stemming for this token
    return ' '.join(tokens)


def extract_engineered_features(texts):
    """Extracts meta-features like punctuation counts, upper case ratio, etc."""
    feat_list = []
    names = ['char_len', 'token_count', 'exclam_count', 'question_count', 'upper_ratio', 'negation_word_flag', 'emoticon_count', 'repeat_char_count']

    for t in texts:
        original_text = t if isinstance(t, str) else ''
        char_len = len(original_text)
        tokens = original_text.split()
        token_count = len(tokens)
        exclam = original_text.count('!')
        question = original_text.count('?')
        
        uppers = sum(1 for ch in original_text if ch.isupper())
        upper_ratio = (uppers / char_len) if char_len > 0 else 0.0
        
        neg_flag = int(any(w in NEGATION_WORDS for w in tokens))
        
        emoticons = sum(original_text.count(emo) for emo in [':)', ':(', ':D', ';)', ':P', '❤'])
            
        repeats = len(re.findall(r'(.)\1{2,}', original_text))
        
        feat_list.append([char_len, token_count, exclam, question, upper_ratio, neg_flag, emoticons, repeats])
        
    return np.array(feat_list, dtype=float), names


def derive_score(val, num_min=None, num_max=None):
    """Normalizes scores to a 0.0 - 1.0 float range from numeric or string input."""
    if pd.isna(val):
        return None
    # Try converting to float first
    try:
        num = float(val)
        if num_min is not None and num_max is not None and num_max > num_min:
            return max(0.0, min(1.0, (num - num_min) / (num_max - num_min)))
        return num # Return as is if scaling is not possible
    except (ValueError, TypeError):
        # Handle string labels
        s = str(val).strip().lower()
        if s in ['positive', 'pos', 'pozitif', '1', 'evet']:
            return 1.0
        if s in ['negative', 'neg', 'negatif', '0', 'hayır']:
            return 0.0
        if s in ['neutral', 'nötr', 'orta']:
            return 0.5
        return None


def main():
    # 1. Load Data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check the path.")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH, engine='openpyxl')

    comment_col = detect_column(df, ['comment', 'yorum', 'text', 'content', 'message', 'review'])
    rating_col = detect_column(df, ['rating', 'puan', 'score', 'stars', 'label'])

    if not comment_col:
        raise ValueError(f"Could not find a comment/text column in {list(df.columns)}")
    if not rating_col:
        raise ValueError(f"Could not find a rating/score column in {list(df.columns)}")

    # 2. Pre-filter and Normalize Scores
    df = df.dropna(subset=[comment_col, rating_col])
    
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = numeric_ratings.min()
    num_max = numeric_ratings.max()

    df['score'] = df[rating_col].apply(lambda v: derive_score(v, num_min=num_min, num_max=num_max))
    df = df.dropna(subset=['score'])
    df['score'] = df['score'].astype(float)
    
    raw_comments = df[comment_col].astype(str).values
    y = df['score'].values

    print(f"Data loaded and cleaned. Samples: {len(df)}")

    # 3. Setup NLP Tools (Stopwords / Stemmer / Zemberek)
    stop_words, stemmer, zemberek_lemmatizer = None, None, None
    
    try:
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        try:
            stop_words = list(stopwords.words('turkish'))
        except LookupError:
            print("NLTK 'stopwords' not found. Downloading...")
            nltk.download('stopwords')
            stop_words = list(stopwords.words('turkish'))
        stemmer = SnowballStemmer('turkish')
    except ImportError:
        print("NLTK not found. Using basic fallback stopwords. Stemming will be skipped.")
        stop_words = ['ve', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'mi', 'ne', 'ama', 'çok', 'gibi']

    # Optional: Zemberek setup for lemmatization
    try:
        from zemberek import TurkishMorphology
        tm = TurkishMorphology.create_with_defaults()

        def zemberek_lemma_tokenize(text):
            try:
                analysis = tm.analyze_sentence(text)
                lemmas = []
                for word_analysis in analysis:
                    best_analysis = word_analysis.analysis_results[0] if word_analysis.analysis_results else None
                    if best_analysis and best_analysis.dictionary_item.lemma != "UNK":
                        lemmas.append(best_analysis.dictionary_item.lemma)
                    else:
                        lemmas.append(word_analysis.surface) # Keep original if cannot be lemmatized
                return ' '.join(lemmas)
            except Exception as e:
                print(f"Warning: Zemberek lemmatization failed for a sentence. Returning original. Error: {e}")
                return text

        zemberek_lemmatizer = zemberek_lemma_tokenize
        print("Zemberek TurkishMorphology loaded for lemmatization.")
    except ImportError:
        print("Zemberek-python not found. Skipping lemmatization. Using NLTK stemmer if available.")
    
    # 4. Preprocess Text
    print("Preprocessing texts...")
    processor = zemberek_lemmatizer if zemberek_lemmatizer else lambda t: preprocess_text(t, stop_words, stemmer)
    X_texts_preprocessed = [processor(clean_text(t)) for t in raw_comments]

    # 5. Vectorization (Word + Char)
    print("Vectorizing text features...")
    word_vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_WORD_FEATURES, ngram_range=(1, 2), 
        min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF, sublinear_tf=True
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=(3, 5), 
        max_features=TFIDF_MAX_CHAR_FEATURES, sublinear_tf=True
    )
    
    X_word = word_vectorizer.fit_transform(X_texts_preprocessed)
    X_char = char_vectorizer.fit_transform(X_texts_preprocessed)

    # 6. Engineered Features
    print("Extracting engineered features...")
    X_eng, eng_names = extract_engineered_features(raw_comments)
    X_eng_sparse = csr_matrix(X_eng)

    # 7. Stack All Features
    X = hstack([X_word, X_char, X_eng_sparse], format='csr')

    # 8. Train/Test Split & Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Training Ridge Regression model...")
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # 9. Evaluation
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    try:
        corr = np.corrcoef(y_test, y_pred_test)[0, 1]
    except (ValueError, IndexError):
        corr = float('nan')

    print(f"\n--- Model Evaluation ---")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R-squared: {r2:.6f}")
    print(f"Pearson Correlation: {corr:.6f}")
    print(f"------------------------\n")

    # 10. Save Outputs
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Ridge Regression Model Evaluation Report\n")
        f.write("="*40 + "\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"R-squared: {r2:.6f}\n")
        f.write(f"Pearson Correlation: {corr:.6f}\n\n")
        f.write(f"Test set size: {len(y_test)}\n")
        f.write(f"Total samples: {len(df)}\n")

    print(f"Saved regression report to {REPORT_PATH}")

    pipeline_to_save = {
        'model': model,
        'word_vectorizer': word_vectorizer,
        'char_vectorizer': char_vectorizer,
        'engineered_feature_names': eng_names,
        'stop_words': stop_words, # For reference
        'lemmatizer_used': 'zemberek' if zemberek_lemmatizer else 'nltk_stemmer'
    }
    
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(pipeline_to_save, f)

    print(f"Saved model pipeline to {MODEL_OUT}")

    # 11. Interpretability (Top Coefficients)
    try:
        coef = model.coef_
        word_feats = word_vectorizer.get_feature_names_out()
        char_feats = char_vectorizer.get_feature_names_out()
        all_feat_names = list(word_feats) + list(char_feats) + eng_names
        
        if len(all_feat_names) != len(coef):
            raise ValueError("Mismatch between number of features and coefficients.")

        coef_df = pd.DataFrame({'feature': all_feat_names, 'coefficient': coef})
        coef_df = coef_df.sort_values(by='coefficient', ascending=False)

        with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
            f.write("Top 25 Positive Features (Predicts Higher Score):\n")
            f.write(coef_df.head(25).to_string(index=False))
            f.write("\n\n" + "="*40 + "\n\n")
            f.write("Top 25 Negative Features (Predicts Lower Score):\n")
            f.write(coef_df.tail(25).sort_values(by='coefficient').to_string(index=False))
            
        print(f"Saved top features to {FEATURES_PATH}")
    except Exception as e:
        print(f"Could not save feature importance: {e}")


if __name__ == '__main__':
    main()