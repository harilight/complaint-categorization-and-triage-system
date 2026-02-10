import pandas as pd
import torch
import joblib
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

df = pd.read_csv("cleanedfinaldataset.csv")
df = df[['Complaint', 'Category']].dropna()

texts = df['Complaint'].astype(str).tolist()
labels = df['Category']

MODEL_NAME = "bert-base-uncased"
BERT_SAVE_PATH = "bert_model_local"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)
bert_model.eval()

def get_cls_embeddings(texts, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = bert_model(**encoded)
        cls_embeddings = output.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings)
    return torch.cat(embeddings).numpy()

X_embeddings = get_cls_embeddings(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

clf = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    n_jobs=-1
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred) * 100)

bert_model.save_pretrained(BERT_SAVE_PATH)
tokenizer.save_pretrained(BERT_SAVE_PATH)

joblib.dump(clf, "bert_logistic_model.pkl")

print("BERT model, tokenizer, and classifier saved successfully")
