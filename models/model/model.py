import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# -------------------------
# COLUMN NAMES
# -------------------------
columns = [
    "id", "label", "statement", "subject", "speaker",
    "job_title", "state_info", "party_affiliation",
    "barely_true", "false", "half_true", "mostly_true",
    "pants_on_fire", "context"
]


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("train.tsv", sep="\t", names=columns)


# -------------------------
# FEATURE ENGINEERING
# -------------------------
df["text"] = df["statement"].fillna("") + " " + df["subject"].fillna("")

X = df["text"]
y = df["label"]


# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# MODEL PIPELINE
# -------------------------
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


# -------------------------
# TRAIN MODEL
# -------------------------
print("Training model...")
model.fit(X_train, y_train)


# -------------------------
# EVALUATION
# -------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump(model, "liar_model.pkl")

print("\nModel saved as liar_model.pkl")
