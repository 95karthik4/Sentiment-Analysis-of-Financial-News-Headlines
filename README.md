# 📈 Financial News Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange) ![Status](https://img.shields.io/badge/Status-Completed-success)

### 📢 Project Overview
In the fast-paced world of finance, news drives the market. A single headline can cause stocks to plummet or soar. 

This project builds a Natural Language Processing (NLP) pipeline to automatically classify financial news headlines into three sentiment categories: **Positive, Negative, or Neutral**. 

By comparing multiple feature extraction techniques and classification algorithms, I identified the optimal model for predicting market sentiment from text data.

---

### 🏆 Key Results
I benchmarked two vectorization strategies (*Count Vectorizer vs. TF-IDF*) against two classifiers (*Naive Bayes vs. Logistic Regression*).

**The Winning Combination: Logistic Regression + Count Vectorizer**

| Feature Type | Model | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **Count (Bag of Words)** | **Logistic Regression** | **75.2%** 🏆 | **0.738** |
| Count (Bag of Words) | Multinomial NB | 71.5% | 0.702 |
| TF-IDF | Logistic Regression | 71.8% | 0.692 |
| TF-IDF | Multinomial NB | 67.5% | 0.605 |

> **Analysis:** Logistic Regression with simple Count Vectorization achieved the highest weighted F1 score (0.738), proving it was the most balanced at maximizing both precision and recall across all sentiment classes.

---

### ⚙️ The NLP Pipeline

#### 1. Data Processing
* **Input:** A dataset of financial news headlines labeled by sentiment.
* **Cleaning:** Text was normalized to lowercase to reduce vocabulary size and improve matching.

#### 2. Feature Engineering
We tested two methods to convert text into machine-readable numbers:
* **CountVectorizer:** Counts the frequency of words (n-grams range 1-2).
* **TfidfVectorizer:** Weighs words based on uniqueness (penalizing common words like "the").

#### 3. Classification & Evaluation
* **Algorithms:** Trained **Multinomial Naive Bayes** and **Logistic Regression** models.
* **Metrics:** Evaluated using Accuracy, Precision, Recall, and Weighted F1-Score.
* **Visualization:** Generated Confusion Matrices to visualize where the models made errors.


---

### 💻 Technologies Used
* **Python:** Core programming language.
* **Scikit-Learn:** Used for vectorization (`CountVectorizer`, `TfidfVectorizer`), modeling, and metrics.
* **Pandas & NumPy:** For data manipulation.
* **Matplotlib & Seaborn:** For generating the heatmap visualizations.

---

### 👨‍💻 About the Author
**Karthik Kunnamkumarath**
*Aerospace Engineer | Project Management Professional (PMP) | AI Solutions Developer*

I combine engineering precision with data science to solve complex problems.
* 📍 Toronto, ON
* 💼 [LinkedIn Profile](https://linkedin.com/in/4karthik95)
* 📧 Aero13027@gmail.com
