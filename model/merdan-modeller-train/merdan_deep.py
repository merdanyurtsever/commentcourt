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

# ---------- Config ----------
DATA_PATH = 'database/raw/Veri_Seti.xlsx'
MODEL_OUT = 'model/merdan-modeller-train/weights/merdan_textcnn_model.pth'
VOCAB_OUT = 'model/merdan-modeller-train/weights/merdan_textcnn_vocab.pkl'

# Hyperparameters
MAX_LEN = 100             # Max sequence length (pad/truncate to this)
EMBED_DIM = 300           # Dimension of word embeddings
FILTER_SIZES = [3, 4, 5]  # N-gram filter sizes
NUM_FILTERS = 100         # Number of filters per size
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Device Config
device = torch.device('cpu')
print(f"Using device: {device}")
torch.set_num_threads(4)

# ---------- Preprocessing ----------

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
    text = text.replace('İ', 'i').replace('I', 'ı').lower()
    text = re.sub(r'http\S+|www\.[^\s]+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'[^a-zçğıöşü\s!:?\-\)\(\[\]\{\}\.,;"\'"\+\/\\]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def derive_score(val, num_min=None, num_max=None):
    try:
        num = float(val)
        if num_min is None or num_max is None or num_max == num_min:
            return None
        return float((num - num_min) / (num_max - num_min))
    except Exception:
        s = str(val).strip().lower()
        if s in ['positive', 'pos', 'pozitif', '1']: return 1.0
        if s in ['negative', 'neg', 'negatif', '0']: return 0.0
        if s in ['neutral', 'nötr', 'nötral', 'orta']: return 0.5
        return None

# ---------- Vocabulary Builder ----------

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
        print(f"Vocab size: {len(self.stoi)}")

    def numericalize(self, text):
        tokenized = text.split()
        return [self.stoi.get(token, self.stoi['<UNK>']) for token in tokenized]

# ---------- PyTorch Dataset ----------

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
        
        # Convert text to integers
        indices = self.vocab.numericalize(text)
        
        # Pad or Truncate
        if len(indices) < self.max_len:
            indices += [self.vocab.stoi['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(score, dtype=torch.float)

# ---------- TextCNN Model ----------

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutional Layers (one for each filter size)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
        self.sigmoid = nn.Sigmoid() # Force output between 0 and 1 for regression

    def forward(self, x):
        # x: [Batch, Seq_Len]
        x = self.embedding(x)  # [Batch, Seq_Len, Embed_Dim]
        
        # Permute for Conv1d: PyTorch expects [Batch, Channels, Length]
        x = x.permute(0, 2, 1) # [Batch, Embed_Dim, Seq_Len]
        
        # Apply Conv -> ReLU -> MaxPool
        # Result of conv: [Batch, Num_Filters, Out_Len]
        # Result of max_pool: [Batch, Num_Filters]
        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate features from different kernel sizes
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        
        output = self.fc(cat)
        return self.sigmoid(output).squeeze()

# ---------- Main Execution ----------

def main():
    # 1. Load and Clean Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading Data...")
    df = pd.read_excel(DATA_PATH)
    
    comment_col = detect_column(df, ['comment', 'yorum', 'text', 'content'])
    rating_col = detect_column(df, ['rating', 'puan', 'score', 'label'])
    
    if not comment_col or not rating_col:
        raise ValueError("Columns not found.")

    df = df.dropna(subset=[comment_col])
    
    # Calculate score logic (0.0 - 1.0)
    numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
    num_min = float(numeric_ratings.min()) if not numeric_ratings.dropna().empty else None
    num_max = float(numeric_ratings.max()) if not numeric_ratings.dropna().empty else None

    df['score'] = df[rating_col].apply(lambda v: derive_score(v, num_min, num_max))
    df = df.dropna(subset=['score'])
    
    # Clean text
    print("Cleaning text...")
    df['clean_text'] = df[comment_col].apply(clean_text)
    # Remove empty strings after cleaning
    df = df[df['clean_text'].str.strip().astype(bool)]
    
    texts = df['clean_text'].values
    y = df['score'].values.astype(float)

    # 2. Build Vocabulary
    print("Building Vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocabulary(texts)

    # 3. Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_dataset = SentimentDataset(X_train, y_train, vocab, MAX_LEN)
    val_dataset = SentimentDataset(X_val, y_val, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    model = TextCNN(
        vocab_size=len(vocab.stoi),
        embed_dim=EMBED_DIM,
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.MSELoss() # Regression loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    print(f"Starting training on {device} for {EPOCHS} epochs...")
    best_loss = float('inf')

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
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Create output directory
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)

    # 6. Final Evaluation
    print("\nTraining Complete. Evaluating Best Model...")
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()
    
    final_preds = []
    final_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            final_preds.extend(outputs.cpu().numpy())
            final_targets.extend(targets.numpy())

    mse = mean_squared_error(final_targets, final_preds)
    mae = mean_absolute_error(final_targets, final_preds)
    r2 = r2_score(final_targets, final_preds)
    corr = np.corrcoef(final_targets, final_preds)[0, 1]

    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Pearson Correlation: {corr:.6f}")

    # Save Report
    report_path = os.path.join(os.path.dirname(MODEL_OUT), 'merdan_textcnn_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Model: TextCNN\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"Pearson: {corr:.6f}\n")
    
    print(f"Report saved to {report_path}")

    # Save Vocab and Hyperparams
    meta_data = {
        'vocab': vocab,
        'max_len': MAX_LEN,
        'embed_dim': EMBED_DIM,
        'filter_sizes': FILTER_SIZES,
        'num_filters': NUM_FILTERS
    }
    with open(VOCAB_OUT, 'wb') as f:
        pickle.dump(meta_data, f)
    print(f"Vocab saved to {VOCAB_OUT}")

if __name__ == '__main__':
    main()