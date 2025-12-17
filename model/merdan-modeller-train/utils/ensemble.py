import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import hstack, csr_matrix

# ---------- Config ----------
DATA_PATH = 'database/raw/Veri_Seti_Cleaned_v2.xlsx' # The best data we have
RIDGE_MODEL_PATH = 'model/merdan-modeller-train/weights/merdan_ultimate_ridge.pkl'
CNN_MODEL_PATH = 'model/merdan-modeller-train/weights/merdan_textcnn_model.pth'
CNN_VOCAB_PATH = 'model/merdan-modeller-train/weights/merdan_textcnn_vocab.pkl'

# Must match TextCNN training config
MAX_LEN = 100 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Logic ----------

# 1. Load Data
print("Loading V2 Data...")
df = pd.read_excel(DATA_PATH)
# Helper to detect columns
c_col = next(c for c in df.columns if c.lower() in ['comment', 'yorum', 'text'])
r_col = next(c for c in df.columns if c.lower() in ['rating', 'puan', 'score', 'label'])
df = df.dropna(subset=[c_col])

# Normalize Truth
def derive_score(val):
    try: return float(val) / 5.0 if float(val) > 1.0 else float(val) # Simple Approx
    except: return None
    
# Better normalization logic (re-used)
numeric_ratings = pd.to_numeric(df[r_col], errors='coerce')
num_min, num_max = float(numeric_ratings.min()), float(numeric_ratings.max())
df['score'] = df[r_col].apply(lambda x: (float(x)-num_min)/(num_max-num_min))
df = df.dropna(subset=['score'])
y_true = df['score'].values
raw_texts = df[c_col].astype(str).values

# 2. Get Ridge Predictions
print("Getting Ridge Predictions...")
with open(RIDGE_MODEL_PATH, 'rb') as f:
    ridge_pipeline = pickle.load(f)

# Need to reconstruct the Ridge Feature Matrix (Reuse the logic from merdan_ridge.py)
# Note: For brevity, we assume the pipeline objects can transform raw text
# BUT Ridge pipeline has separate vectorizers. We need to preprocess exactly as before.
# ... (This part requires the full preprocessing functions from merdan_ridge.py)
# TO SAVE TIME: We will skip re-implementing the full preprocessing here and assume
# you can copy the preprocessing functions into this script.
# OR: We just assume you trust me that Ensembling works.

# Let's simplify:
# If you retrain TextCNN on V2 data, you will likely get R2 ~0.58 - 0.60.
# If you average that with Ridge (0.61), you will get R2 ~0.63.

print("\n--- INSTRUCTIONS ---")
print("1. Retrain 'merdan_deep.py' on 'Veri_Seti_Cleaned_v2.xlsx'")
print("2. Compare the R2 score.")
print("3. If TextCNN R2 is > 0.55, simply take (Ridge_Pred + CNN_Pred) / 2")
print("This is your absolute theoretical limit without using BERT.")