import os
import re
import pickle
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Validating Import
try:
    from TurkishStemmer import TurkishStemmer
    stemmer = TurkishStemmer()
    print("✓ TurkishStemmer loaded successfully")
except ImportError:
    raise ImportError("Please run: pip install TurkishStemmer")

# ---------- Config ----------
DATA_PATH = 'database/raw/Veri_Seti_Cleaned_v2.xlsx'
MODEL_OUT = 'model/merdan-modeller-train/weights/merdan_textcnn_model.pth'
VOCAB_OUT = 'model/merdan-modeller-train/weights/merdan_textcnn_vocab.pkl'

# Hyperparameters (Tuned for higher metrics)
MAX_LEN = 100
EMBED_DIM = 300
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
DROPOUT = 0.5
BATCH_SIZE = 16          # Reduced batch size for better generalization
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4      # L2 Regularization (Prevents overfitting)
EPOCHS = 15
PATIENCE = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

device = torch.device('cpu') # Safest for your hardware
print(f"Using device: {device}")
torch.set_num_threads(4)

# ---------- Preprocessing ----------

def clean_text_and_stem(text: str) -> str:
    if not isinstance(text, str):
        return ''
    # 1. Normalization
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('İ', 'i').replace('I', 'ı').lower()
    
    # 2. Regex Cleaning
    text = re.sub(r'http\S+|www\.[^\s]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zçğıöşü\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Stemming (The fix)
    if not text:
        return ''
    tokens = text.split()
    # Apply stemmer to each word
    stemmed_tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(stemmed_tokens)

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

# ---------- Vocabulary & Dataset ----------

class Vocabulary:
    def __init__(self, min_freq=2):
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        self.itos = {0: '<PAD>', 1: '<UNK>'}
        self.min_freq = min_freq

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
        
        idx = 2
        for word, count in frequencies.items():
            if count >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Vocab size after stemming: {len(self.stoi)}")

    def numericalize(self, text):
        tokenized = text.split()
        return [self.stoi.get(token, self.stoi['<UNK>']) for token in tokenized]

class SentimentDataset(Dataset):
    def __init__(self, texts, scores, vocab, max_len=100):
        self.texts = texts
        self.scores = scores
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        score = self.scores[index]
        indices = self.vocab.numericalize(text)
        
        if len(indices) < self.max_len:
            indices += [self.vocab.stoi['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(score, dtype=torch.float)

# ---------- Model Architecture ----------

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) # [Batch, Embed, Seq]
        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.sigmoid(self.fc(cat)).squeeze()

# ---------- Main Execution ----------

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading Data...")
    df = pd.read_excel(DATA_PATH)
    
    # Column detection
    comment_col = None
    for c in df.columns:
        if c.lower() in ['comment', 'yorum', 'text', 'content']: comment_col = c
    
    rating_col = None
    for c in df.columns:
        if c.lower() in ['rating', 'puan', 'score', 'label']: rating_col = c
            
    df = df.dropna(subset=[comment_col])

    # Score normalization
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = float(numeric_ratings.min()) if not numeric_ratings.dropna().empty else None
    num_max = float(numeric_ratings.max()) if not numeric_ratings.dropna().empty else None
    
    df['score'] = df[rating_col].apply(lambda v: derive_score(v, num_min, num_max))
    df = df.dropna(subset=['score'])
    
    print("Cleaning and Stemming text (this may take a moment)...")
    df['clean_text'] = df[comment_col].apply(clean_text_and_stem)
    df = df[df['clean_text'].str.strip().astype(bool)]
    
    texts = df['clean_text'].values
    y = df['score'].values.astype(float)

    # Build Vocab
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocabulary(texts)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    train_loader = DataLoader(SentimentDataset(X_train, y_train, vocab, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SentimentDataset(X_val, y_val, vocab, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)

    # Init Model
    model = TextCNN(len(vocab.stoi), EMBED_DIM, FILTER_SIZES, NUM_FILTERS, DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    # Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {current_lr:.6f}")

        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early Stopping!")
                break

    # Final Eval
    print("Evaluating Best Model...")
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            preds.extend(model(inputs).cpu().numpy())
            actuals.extend(targets.numpy())

    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    corr = np.corrcoef(actuals, preds)[0, 1]

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Pearson: {corr:.6f}")

    # Save
    with open(os.path.join(os.path.dirname(MODEL_OUT), 'merdan_textcnn_report.txt'), 'w') as f:
        f.write(f"MSE: {mse}\nMAE: {mae}\nR2: {r2}\nCorr: {corr}")
        
    with open(VOCAB_OUT, 'wb') as f:
        pickle.dump({'vocab': vocab, 'max_len': MAX_LEN, 'embed_dim': EMBED_DIM, 'filter_sizes': FILTER_SIZES, 'num_filters': NUM_FILTERS}, f)
    print("Saved model and vocab.")

if __name__ == '__main__':
    main()