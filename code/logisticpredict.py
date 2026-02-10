import pandas as pd
import torch
import joblib
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

BERT_PATH = "bert_model_local"
CLASSIFIER_PATH = "bert_logistic_model.pkl"

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert_model = BertModel.from_pretrained(BERT_PATH)
bert_model.eval()

clf = joblib.load(CLASSIFIER_PATH)

def get_cls_embedding(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
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

def predict_single(text):
    text = str(text)
    embedding = get_cls_embedding([text])
    return clf.predict(embedding)[0]

def predict_from_csv(input_csv, output_csv="predictions.csv"):
    df = pd.read_csv(input_csv)
    if "Complaint" not in df.columns:
        raise ValueError("CSV must contain a 'Complaint' column")

    texts = df["Complaint"].astype(str).tolist()
    embeddings = get_cls_embedding(texts)
    predictions = clf.predict(embeddings)

    df["Predicted_Category"] = predictions
    df.to_csv(output_csv, index=False)
    return output_csv

if __name__ == "__main__":
    mode = input("Enter mode (single / file): ").strip().lower()

    if mode == "single":
        complaint = input("Enter Complaint: ")
        result = predict_single(complaint)
        print("Predicted Category:", result)

    elif mode == "file":
        input_csv = input("Enter input CSV file path: ")
        output_csv = input("Enter output CSV file name (default: predictions.csv): ").strip()
        if not output_csv:
            output_csv = "predictions.csv"
        saved_file = predict_from_csv(input_csv, output_csv)
        print("Predictions saved to:", saved_file)

    else:
        print("Invalid mode. Use 'single' or 'file'")
