import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os


# ---------- Config ----------
DATA_PATH = 'database/raw/Veri_Seti.xlsx'
MODEL_OUT = 'model/weights/merdan_classic_model.pkl'
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42


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
    # Turkish-aware lowercase: handle dotted/dotless I
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    # remove urls, emails, mentions, hashtags
    text = re.sub(r'http\S+|www\.[^\s]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    # remove punctuation and numbers (keep Turkish letters)
    text = re.sub(r'[^a-zçğıöşü\s]', ' ', text)
    # collapse repeated characters (e.g., cooool -> coool)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str, stop_words_list=None, stemmer=None) -> str:
    # Clean base text
    cleaned = clean_text(text)
    if cleaned == '':
        return ''
    # tokenise on whitespace
    tokens = cleaned.split()
    # remove stopwords
    if stop_words_list:
        tokens = [t for t in tokens if t not in stop_words_list]
    # apply stemmer if provided
    if stemmer:
        try:
            tokens = [stemmer.stem(t) for t in tokens]
        except Exception:
            pass
    return ' '.join(tokens)


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_excel(DATA_PATH)

# Detect comment and rating columns from common candidates
comment_col = detect_column(df, ['comment', 'yorum', 'text', 'content', 'message', 'review'])
rating_col = detect_column(df, ['rating', 'puan', 'score', 'stars', 'label'])

if comment_col is None:
    raise ValueError(f"Could not find a comment/text column in {list(df.columns)}")

# If rating not found, try to infer label column
if rating_col is None:
    # If there is an existing binary/text label column
    if 'label' in df.columns:
        rating_col = 'label'
    else:
        raise ValueError(f"Could not find a rating/score column in {list(df.columns)}")

# Prepare dataframe: drop rows without comment
df = df.dropna(subset=[comment_col])
df[comment_col] = df[comment_col].astype(str)

# Clean comments
df['clean_comment'] = df[comment_col].apply(clean_text)

# Drop very short comments
df = df[df['clean_comment'].str.len() >= 3]

# Handle rating -> binary label mapping
def derive_label(val):
    # numeric ratings: map >=7 -> positive (1), <=4 -> negative (0), else neutral (None)
    try:
        num = float(val)
        if num >= 7:
            return 1
        if num <= 4:
            return 0
        return None
    except Exception:
        # if categorical labels already present
        s = str(val).strip().lower()
        if s in ['positive', 'pos', 'pozitif', '1']:
            return 1
        if s in ['negative', 'neg', 'negatif', '0']:
            return 0
        return None


df['label'] = df[rating_col].apply(derive_label)
# Drop neutral/unknown labels
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

X_texts = df['clean_comment'].values
y = df['label'].values

# Prepare stop words: try NLTK Turkish stopwords, else fallback
stop_words = None
stemmer = None
try:
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        try:
            nltk.download('stopwords')
        except Exception:
            pass
    from nltk.corpus import stopwords as nltk_stopwords
    stop_words = list(nltk_stopwords.words('turkish'))
    # try to get Snowball stemmer for Turkish
    try:
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer('turkish')
    except Exception:
        stemmer = None
except Exception:
    # Fallback minimal Turkish stopword list
    stop_words = [
        've','bir','bu','da','de','ile','için','mi','ne','ama','çok','gibi',
        'olarak','ya','kadar','sonra','önce','eğer','çünkü','ben','sen','o',
        'biz','siz','onlar','her','hiç','daha','ile','var','yok','olan'
    ]

# Ensure stop_words is a list (sklearn expects list or None)
if isinstance(stop_words, set):
    stop_words = list(stop_words)

# Preprocess texts with token-level stopword removal and stemming
X_texts_pre = [preprocess_text(t, stop_words_list=stop_words, stemmer=stemmer) for t in X_texts]

# Vectorize
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words=None)
X = vectorizer.fit_transform(X_texts_pre)

# Train/test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Initial simple training to detect mismatches (optional weighting)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_train = nb_model.predict(X_train)

# Calculate sample weights: penalize samples mismatching the simple model
sample_weights = np.where(y_pred_train != y_train, 0.2, 1.0)

# Retrain with adjusted weights
nb_model_weighted = MultinomialNB()
nb_model_weighted.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate on test set
y_pred_test = nb_model_weighted.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and vectorizer together
with open(MODEL_OUT, 'wb') as f:
    pickle.dump({'model': nb_model_weighted, 'vectorizer': vectorizer}, f)

print(f"Model and vectorizer saved as {MODEL_OUT}")