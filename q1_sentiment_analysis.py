import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


DATA_PATH = "/Users/karthik/Downloads/financial_news_headlines_sentiment.csv"


OUT_DIR = "/Users/karthik/Downloads/"

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    try:
        df = pd.read_csv(path, header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(path, header=None, encoding='latin1')
    if df.shape[1] < 2:
        raise ValueError("Expected two columns: label, text")
    df = df.iloc[:, :2]
    df.columns = ['label', 'text']
    df = df.dropna()
    return df

def preprocess_text(df):
    df['text'] = df['text'].astype(str).str.lower()
    return df

def vectorize_features(X_train, X_test, method="count"):
    if method == "count":
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
    else:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    return Xtr, Xte, vectorizer

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
    return acc, precision, recall, f1, preds

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

def main():
    df = load_data()
    df = preprocess_text(df)

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    feature_methods = ["count", "tfidf"]
    results = []

    for fm in feature_methods:
        Xtr, Xte, vec = vectorize_features(X_train, X_test, fm)
        for model_name, model in models.items():
            acc, prec, rec, f1, preds = evaluate_model(model, Xtr, y_train, Xte, y_test)
            results.append([fm, model_name, acc, prec, rec, f1])
            plot_confusion(
                y_test, preds,
                f"Confusion Matrix: {model_name} ({fm})",
                f"confusion_{fm}_{model_name}.png"
            )

    results_df = pd.DataFrame(results, columns=["Features", "Model", "Accuracy", "Precision", "Recall", "F1"])
    results_df.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)

    best_row = results_df.sort_values("F1", ascending=False).iloc[0]
    summary_text = f"""Assignment Summary

Results Summary:
{results_df.to_string(index=False)}

Best Method:
Feature Type: {best_row['Features']}
Model: {best_row['Model']}
Weighted F1 Score: {best_row['F1']:.4f}

Explanation:
This model achieved the highest weighted F1 score, balancing precision and recall across classes and producing overall superior sentiment classification performance.
"""

    with open(os.path.join(OUT_DIR, "assignment_submission.txt"), "w") as f:
        f.write(summary_text)

if __name__ == "__main__":
    main()
