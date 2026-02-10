# 🛡️ Customer Complaint Categorization: using SVM(TF-IDF + SVM) and BERT(bert + logistic regression)

An automated NLP pipeline designed to classify customer complaints into specific business categories. This project explores the trade-off between traditional feature engineering (**TF-IDF + SVM**) and modern deep learning embeddings (**BERT + Logistic Regression**).



## 📌 Project Overview
Manual categorization of customer feedback is time-consuming and prone to human error. This system automates the process by taking raw text input and predicting the most relevant category.

### **Key Features**
* **Dual-Model Support:** Compare results between SVM and BERT.
* **Batch Processing:** Predict categories for entire CSV files.
* **Real-time Inference:** A command-line interface for single-text testing.
* **GPU Optimized:** Built with PyTorch to leverage CUDA on hardware like the NVIDIA RTX 3050 Ti.

---

## 🔬 Approaches Compared

### **Approach 1: TF-IDF + Support Vector Machine (SVM)**
The traditional approach focusing on word frequency and importance.
* **How it works:** Text is cleaned and converted into a numerical matrix using **TF-IDF**. An **SVM** classifier draws a hyperplane to separate the data.
* **Pros:** Extremely fast, simple, and works well on smaller datasets.
* **Cons:** Doesn't understand deep semantic meaning; can be confused by wording changes.

### **Approach 2: BERT + Logistic Regression**
The deep learning-based approach focusing on semantic meaning.
* **How it works:** Uses pretrained **`bert-base-uncased`** to extract the **[CLS] token** embedding for each complaint. A **Logistic Regression** model classifies these embeddings.
* **Pros:** Understands context, synonyms, and handles overlapping categories better.
* **Cons:** Slower and needs more compute (GPU recommended).

---

## 📊 Performance Benchmarks
| Metric | TF-IDF + SVM | BERT + Logistic |
| :--- | :--- | :--- |
| **Accuracy (Approx)** | ~98% | **~99.5%** |
| **Training Speed** | Ultra Fast | Slower (Requires GPU) |
| **Context Awareness** | Basic | High |

---

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** Scikit-learn, PyTorch
* **Transformers:** HuggingFace Transformers (BERT)
* **Techniques:** TF-IDF, SVM, Logistic Regression

---

## 🚀 How to Run

### **1. Installation**

pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install transformers scikit-learn pandas joblib tqdm


## 📸 Screenshots

### Terminal Interface
![Main Menu](assets/terminal_ui.png)

### Spending Visualization
![Spending Chart](assets/spending_graph.png)
