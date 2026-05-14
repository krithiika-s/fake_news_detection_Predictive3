import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Column names
columns = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true", "false", "half_true", "mostly_true",
    "pants_on_fire", "context"
]

# Load dataset
df = pd.read_csv("train.tsv", sep="\t", names=columns)

# Feature engineering
df["text"] = df["statement"].fillna("") + " " + df["subject"].fillna("")

X = df["text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=12000,
            ngram_range=(1, 2)
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=500,
            class_weight="balanced"
        )
    )
])

# Train
print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")
