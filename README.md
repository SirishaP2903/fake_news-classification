# 📰 Fake News Detection using Natural Language Processing

## 📌 Project Overview

This project demonstrates an end-to-end **Natural Language Processing (NLP) pipeline for Fake News Detection** using classical Machine Learning techniques.

The objective is to understand how textual information can be transformed into numerical features and subsequently used to train a machine learning model to classify news content as either **REAL** or **FAKE**.

The project has been intentionally designed as a **classroom-friendly NLP walkthrough**, using a small, hand-built dataset so that every stage of the process can be inspected and understood.

The same pipeline can later be extended to larger, real-world datasets such as the Kaggle Fake News dataset without fundamentally changing the overall workflow.

---

## 🎯 Objectives

The main objectives of this project are to:

* Understand why Fake News Detection is a **text classification problem**
* Understand the basic NLP preprocessing pipeline
* Clean and normalize raw text
* Perform tokenization
* Remove stopwords
* Perform lemmatization
* Understand **Bag of Words (BoW)**
* Understand **TF-IDF**
* Convert text into numerical feature vectors
* Split data into training and testing sets
* Train classical machine learning classification models
* Compare **Naive Bayes** and **Logistic Regression**
* Evaluate classification performance
* Understand the confusion matrix
* Make predictions on previously unseen text
* Understand the importance of applying the same preprocessing and vectorization pipeline to new data

---

## 🔄 End-to-End NLP Pipeline

The project follows the following workflow:

```text
Raw News Text
      │
      ▼
Text Cleaning
      │
      ├── Lowercase
      ├── Remove punctuation
      ├── Remove numbers
      └── Remove extra spaces
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
Lemmatization
      │
      ▼
Cleaned Text
      │
      ▼
Text Vectorization
      │
      ├── Bag of Words
      └── TF-IDF
      │
      ▼
Numerical Feature Matrix
      │
      ▼
Train / Test Split
      │
      ▼
Machine Learning Models
      │
      ├── Multinomial Naive Bayes
      └── Logistic Regression
      │
      ▼
Model Evaluation
      │
      ├── Accuracy
      ├── Precision
      ├── Recall
      ├── F1 Score
      └── Confusion Matrix
      │
      ▼
Prediction on New Text
```

This complete pipeline is demonstrated in the project notebook.

---

## 📊 Dataset

For this introductory project, a small **dummy dataset containing 20 news-style text samples** is used.

The dataset contains:

* **10 REAL news examples**
* **10 FAKE news examples**

The labels are represented as:

| Label | Meaning |
| ----- | ------- |
| `1`   | REAL    |
| `0`   | FAKE    |

The dataset is deliberately small so that students can inspect individual examples and understand how the complete NLP pipeline works.

### Example

**REAL**

> Scientists at the university published a peer reviewed study on renewable energy storage

**FAKE**

> Scientists confirm the moon is actually a hologram projected by secret government agents

---

# 🧹 NLP Preprocessing

Raw text cannot be directly supplied to most traditional machine learning algorithms.

Therefore, the project demonstrates several preprocessing techniques.

### 1. Lowercasing

Converts all text to lowercase.

```text
"Scientists Confirm the Moon"
```

becomes:

```text
"scientists confirm the moon"
```

This prevents the model from treating `"Moon"` and `"moon"` as separate features.

### 2. Removing Punctuation and Numbers

Unnecessary punctuation and numeric characters are removed to simplify the text representation.

### 3. Tokenization

The sentence is split into individual words, known as **tokens**.

Example:

```text
"scientists confirm the moon"
```

becomes:

```text
["scientists", "confirm", "the", "moon"]
```

### 4. Stopword Removal

Common words such as:

```text
the
is
a
by
```

are removed because they generally provide limited discriminatory information in this particular classification task.

### 5. Lemmatization

Words are converted into their base/dictionary form.

Example:

```text
agents → agent
studies → study
```

## The notebook combines these operations into a reusable `preprocess()` function.

# 🔢 Converting Text into Numbers

Machine learning models cannot directly understand words.

Therefore, textual information must be converted into numerical features.

This project demonstrates two important approaches.

## 1. Bag of Words

**Bag of Words (BoW)** represents each document using word frequencies.

For example:

```text
Doc A: "fake news spreads fast"
Doc B: "real news is checked"
Doc C: "fake news is fast news"
```

A document-term matrix is created where each column represents a word and each row represents a document.

The project demonstrates this using `CountVectorizer`.

### Limitation

Bag of Words treats words primarily according to their occurrence/frequency and does not consider the importance of a word across the entire collection of documents.

---

# 2. TF-IDF

The project then introduces **TF-IDF — Term Frequency-Inverse Document Frequency**.

The basic idea is:

```text
TF-IDF = Term Frequency × Inverse Document Frequency
```

TF-IDF gives greater importance to words that are relatively distinctive within a document and less emphasis to words that occur across many documents.

The notebook also includes a small hand-calculated TF-IDF example before applying `TfidfVectorizer` to the complete dataset.

The final dataset produces a feature matrix of:

```text
20 documents × 100 features
```

using:

```python
TfidfVectorizer(max_features=100)
```

---

# 🤖 Machine Learning Models

Two classical machine learning algorithms are trained.

## Multinomial Naive Bayes

Naive Bayes is a popular baseline algorithm for text classification.

It works particularly well with word-frequency and TF-IDF based representations.

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
```

---

## Logistic Regression

Logistic Regression is another strong baseline for binary text classification.

In addition to classification, its learned coefficients can be inspected to understand which words contribute toward particular classes.

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
```

Both models are trained using the TF-IDF representation.

---

# 📈 Model Evaluation

The project evaluates the models using multiple classification metrics.

### Accuracy

Measures the proportion of correctly classified examples.

### Precision

Measures how many examples predicted as a particular class were actually from that class.

### Recall

Measures how many actual examples of a class were correctly identified.

### F1 Score

Provides a balanced measure based on precision and recall.

### Confusion Matrix

Provides a detailed view of:

* True Positives
* True Negatives
* False Positives
* False Negatives

The notebook emphasizes that accuracy alone may not always provide sufficient information, particularly when dealing with imbalanced datasets.

---

# 🏆 Example Results

Using the small dummy dataset, the notebook produced the following illustrative results:

| Model                   | Accuracy | Precision | Recall | F1 Score |
| ----------------------- | -------: | --------: | -----: | -------: |
| Multinomial Naive Bayes |     0.67 |      0.60 |   1.00 |     0.75 |
| Logistic Regression     |     1.00 |      1.00 |   1.00 |     1.00 |

These results should **not** be interpreted as production-level performance because the dataset contains only 20 manually constructed examples.

The results are intended to demonstrate how model training and evaluation work.

---

# 🔍 Prediction on New Text

After training, the Logistic Regression model is tested on previously unseen sentences.

For example:

```text
The government announced a new tax policy effective from next fiscal year
```

and:

```text
Secret underground city of lizard people discovered beneath the capital say insiders
```

The model generates both:

* Predicted class
* Prediction confidence

The project demonstrates that new text must pass through the **same preprocessing and TF-IDF transformation pipeline** used during training.

---

# 🛠️ Technologies Used

The project is implemented using Python and the following libraries:

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| Python           | Programming language                   |
| Pandas           | Data manipulation                      |
| NumPy            | Numerical operations                   |
| NLTK             | NLP preprocessing                      |
| Scikit-learn     | Vectorization, modeling and evaluation |
| Matplotlib       | Visualization                          |
| Seaborn          | Confusion matrix visualization         |
| Jupyter Notebook | Interactive development and teaching   |

The notebook imports NLTK, Pandas, NumPy, Matplotlib, Seaborn and Scikit-learn components for these tasks.

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
fake-news-detection-nlp/
│
├── README.md
│
├── notebooks/
│   └── fake_news_nlp_basics.ipynb
│
├── data/
│   └── fake_news_dummy.csv
│
├── requirements.txt
│
└── .gitignore
```

For the current classroom version, the dataset is created directly inside the notebook. If you later move the dataset into a separate CSV file, it can be placed inside the `data/` directory.

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fake-news-detection-nlp.git
```

## 2. Navigate to the project

```bash
cd fake-news-detection-nlp
```

## 3. Create a virtual environment

```bash
python -m venv nlp_env
```

Activate it on Windows:

```bash
nlp_env\Scripts\activate
```

## 4. Install dependencies

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn jupyter
```

## 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/fake_news_nlp_basics.ipynb
```

and execute the cells sequentially.

---

# 🎓 Learning Outcomes

After completing this project, learners should be able to explain:

1. What NLP is
2. Why Fake News Detection can be treated as a binary classification problem
3. What text preprocessing means
4. Why tokenization is required
5. What stopwords are
6. Stemming vs. lemmatization
7. What Bag of Words is
8. How a document-term matrix is created
9. What TF-IDF means
10. How text is converted into numerical features
11. How train/test splitting works
12. How Naive Bayes can be used for text classification
13. How Logistic Regression can be used for text classification
14. Accuracy vs. precision vs. recall vs. F1 score
15. How to interpret a confusion matrix
16. How to predict on unseen text
17. Why the training preprocessing/vectorization pipeline must be reused during prediction

---

# 🧪 Exercises and Future Improvements

The notebook includes several exercises that can be used to extend the project.

### Exercise 1 — Expand the Dataset

Add additional REAL and FAKE examples and investigate how model performance changes.

### Exercise 2 — Compare BoW and TF-IDF

Train both models using:

```python
CountVectorizer()
```

instead of TF-IDF and compare the results.

### Exercise 3 — Inspect Important Words

Investigate Logistic Regression coefficients to identify words that strongly influence the prediction toward REAL or FAKE.

### Exercise 4 — Use N-Grams

Experiment with:

```python
TfidfVectorizer(ngram_range=(1, 2))
```

to incorporate both individual words and two-word phrases.

### Exercise 5 — Stress Test the Model

Create ambiguous or deliberately misleading sentences and investigate where the model fails.

These exercises are included in the original notebook as extensions for learners.

---

# 🚀 Possible Future Enhancements

This introductory project can be developed into a more realistic NLP application by:

* Replacing the dummy dataset with a larger real-world dataset
* Using a separate training and testing dataset
* Performing exploratory data analysis
* Handling missing text
* Handling duplicate articles
* Comparing multiple ML algorithms
* Performing hyperparameter tuning
* Using n-gram features
* Handling class imbalance
* Implementing cross-validation
* Building an NLP pipeline using Scikit-learn `Pipeline`
* Saving the trained model using `joblib`
* Creating a Streamlit web application
* Deploying the application
* Experimenting with Word2Vec or other embeddings
* Comparing classical ML approaches with modern transformer-based NLP models

---

# ⚠️ Important Disclaimer

This repository is primarily an **educational demonstration of an NLP text-classification workflow**.

The current model is trained on a very small, manually created dummy dataset of 20 examples. Therefore, the reported performance should **not** be considered evidence that the model can reliably determine whether real-world news is true or false.

Real-world fake-news detection requires substantially larger and more diverse datasets, careful labeling, robust validation, source/context analysis, and consideration of misinformation, satire, bias, and changing language patterns.

---

# 👩‍💻 Author

**Dr. Sirisha P**

Data Science | Artificial Intelligence | Machine Learning | NLP

This repository is created as an educational resource for understanding the fundamentals of Natural Language Processing and Machine Learning-based text classification.

---

# ⭐ If You Find This Project Useful

If this project helps you understand NLP or Machine Learning concepts, consider giving the repository a ⭐ on GitHub.

You are also welcome to fork the repository, experiment with the notebook, expand the dataset, and build your own NLP applications.

---

## 📚 Key Concepts Covered

```text
NLP
│
├── Text Cleaning
├── Tokenization
├── Stopword Removal
├── Lemmatization
│
├── Text Representation
│   ├── Bag of Words
│   └── TF-IDF
│
├── Machine Learning
│   ├── Multinomial Naive Bayes
│   └── Logistic Regression
│
├── Evaluation
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   ├── F1 Score
│   └── Confusion Matrix
│
└── Prediction
    └── Unseen Text Classification
```

**End-to-end workflow:**

> **Text → Preprocessing → Vectorization → Machine Learning → Evaluation → Prediction**
