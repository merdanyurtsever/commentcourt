import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Load the dataset (adjust path if needed)
data_path = 'database/raw/Veri_Seti.xlsx'  # Assuming it's in database/ folder
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_excel(data_path)

# Assume columns: 'comment' (text) and 'rating' (numeric)
# Preprocess: Drop NaNs, lowercase comments
df = df.dropna(subset=['comment', 'rating'])
df['comment'] = df['comment'].str.lower()

# Derive initial labels from ratings (e.g., >5 = positive, else negative)
df['label'] = (df['rating'] > 5).astype(int)  # 1 = positive, 0 = negative

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # Adjust stop_words for Turkish if needed
X = vectorizer.fit_transform(df['comment'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial training to detect mismatches
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_train = nb_model.predict(X_train)

# Calculate weights: Lower weight for mismatches
sample_weights = []
for i, (pred, true) in enumerate(zip(y_pred_train, y_train)):
    if pred != true:
        sample_weights.append(0.1)  # Low weight for mismatches
    else:
        sample_weights.append(1.0)  # Full weight for matches

# Retrain with adjusted weights
nb_model_weighted = MultinomialNB()
nb_model_weighted.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate on test set
y_pred_test = nb_model_weighted.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and vectorizer
with open('merdan_classic_model.pkl', 'wb') as f:
    pickle.dump((nb_model_weighted, vectorizer), f)

print("Model saved as merdan_classic_model.pkl")