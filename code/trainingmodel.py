import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("cleanedfinaldataset.csv")
df = df[['Complaint', 'Category']]
df.dropna(inplace=True)

print("Total complaints:", len(df))
print(df['Category'].value_counts())


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_complaint'] = df['Complaint'].apply(clean_text)


# -----------------------------
# SAFE Synonym Enrichment
# (adds context, does NOT replace)
# -----------------------------
synonym_map = {
    "crash": "malfunction",
    "broken": "malfunction",
    "bug": "malfunction",
    "error": "malfunction",

    "charged": "financial",
    "debited": "financial",
    "invoice": "billingdoc",
    "refund": "reversal",
    "cancelled": "termination",

    "login": "access",
    "profile": "identity",

    "shipment": "logistics",
    "courier": "logistics",
    "delayed": "lateness",
    "late": "lateness",

    "damaged": "defect",
    "wrong": "incorrect",

    "policy": "rulebased",
    "system": "automated",
    "hacked": "security privacy",
    "password": "security account",
    "unauthorized": "security alert",
    "suspicious": "security warning"
}


def enrich_synonyms(text):
    for word, extra in synonym_map.items():
        text = re.sub(
            rf'\b{word}\b',
            f'{word} {extra}',
            text
        )
    return text

df['clean_complaint'] = df['clean_complaint'].apply(enrich_synonyms)

df.drop_duplicates(inplace=True)
# -----------------------------
# Features & Labels
# -----------------------------
X = df['clean_complaint']
y = df['Category']


# -----------------------------
# TF-IDF Vectorizer (Optimized)
# -----------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    max_features=12000,
    sublinear_tf=True,
    stop_words='english'
)

X_vec = vectorizer.fit_transform(X)


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# SVM Model (Optimized)
# -----------------------------
model = SVC(
    kernel='linear',
    C=1.0,
    class_weight='balanced'
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Model Accuracy:", accuracy)

# -----------------------------
# Save Model & Vectorizer
# (UNCHANGED pattern)
# -----------------------------
joblib.dump(model, "svm_model.pltk")
joblib.dump(vectorizer, "vectorizer.pltk")

print("Model and vectorizer saved successfully.")
