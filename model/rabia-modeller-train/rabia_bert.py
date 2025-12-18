# ==============================
# 1. KÜTÜPHANELER
# ==============================
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import os
from google.colab import files

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 2. VERİYİ OKU
# ==============================
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

if file_name.endswith(".csv"):
    df = pd.read_csv(file_name)
elif file_name.endswith(".xlsx"):
    df = pd.read_excel(file_name)
else:
    raise ValueError("Desteklenmeyen dosya formatı")

# 🔴 SADECE İLK 1500 SATIRI KULLAN
df = df.iloc[:1500]

# 🔧 Elle etiketlenmiş sentiment sütunu
df = df.dropna(subset=["comment", "sentiment_manual"])

# ==============================
# 3. TRAIN / VALIDATION SPLIT
# ==============================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ==============================
# 4. DATASET SINIFI
# ==============================
class ReviewDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ==============================
# 5. TOKENIZER & MODEL
# ==============================
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression"
).to(DEVICE)

# ==============================
# 6. DATALOADER
# ==============================
train_ds = ReviewDataset(
    train_df["comment"].tolist(),
    train_df["sentiment_manual"].tolist(),
    tokenizer
)

val_ds = ReviewDataset(
    val_df["comment"].tolist(),
    val_df["sentiment_manual"].tolist(),
    tokenizer
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# ==============================
# 7. OPTIMIZER
# ==============================
optimizer = AdamW(model.parameters(), lr=2e-5)

# ==============================
# 8. TRAINING LOOP
# ==============================
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    train_preds, train_truths = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        preds = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        train_preds.extend(preds)
        train_truths.extend(labels.cpu().numpy())

    # 🔵 TRAIN METRICS
    train_preds = np.array(train_preds)
    train_truths = np.array(train_truths)

    train_acc = accuracy_score(
        (train_truths > 0.5).astype(int),
        (train_preds > 0.5).astype(int)
    )

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {np.mean(train_losses):.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")

    # ==============================
    # 9. VALIDATION METRICS
    # ==============================
    model.eval()
    val_preds, val_truths = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.sigmoid(outputs.logits).cpu().numpy()
            val_preds.extend(preds)
            val_truths.extend(labels.cpu().numpy())

    val_preds = np.array(val_preds)
    val_truths = np.array(val_truths)

    val_acc = accuracy_score(
        (val_truths > 0.5).astype(int),
        (val_preds > 0.5).astype(int)
    )

    val_f1 = f1_score(
        (val_truths > 0.5).astype(int),
        (val_preds > 0.5).astype(int)
    )

    try:
        val_auc = roc_auc_score(val_truths, val_preds)
    except:
        val_auc = np.nan

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation ROC-AUC: {val_auc}")

# ==============================
# 10. MODELİ KAYDET
# ==============================
SAVE_PATH = "sentiment_model_manual_son"
os.makedirs(SAVE_PATH, exist_ok=True)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\nModel ve tokenizer başarıyla kaydedildi.")