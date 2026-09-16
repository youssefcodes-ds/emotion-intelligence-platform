# Customer Sentiment & Emotion Intelligence Platform

An end-to-end NLP platform that analyzes customer messages and classifies them into six emotions. The project combines data engineering, SQL analytics, classical machine learning, deep learning, and a Streamlit application.

> Graduation Project — TechTrek Advanced Data Science & AI

---

## Team 236

| Member | Responsibility |
|---|---|
| **Youssef Tarek** | Data Engineering, SQL & EDA |
| **Menna El-Sayed** | Classical ML & Deployment |
| **Malak Ahmed** | Project Development |
| **Mawada Karam** | Deep Learning & Advanced NLP |

---

## Project Overview

Customer support teams receive large amounts of unstructured text that can be difficult to analyze manually.

This project provides an automated emotion intelligence platform that:

- Cleans and analyzes customer text
- Classifies messages into six emotions
- Compares classical NLP and deep learning approaches
- Provides SQL-based analytical insights
- Supports single-message and batch prediction
- Provides an interactive Streamlit dashboard
- Includes testing and Docker support

> **Note:** Emotion predictions are not clinical or mental-health assessments.

---

## Dataset

The project uses the **dair-ai/emotion** dataset.

The dataset contains six emotion classes:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

### Dataset Splits

| Split | Records |
|---|---:|
| Train | 15,983 |
| Validation | 2,000 |
| Test | 2,000 |
| **Total** | **19,983** |

The original raw dataset is not committed to the repository. Processed datasets are provided for reproducibility.

---

## Data Engineering & EDA

The data pipeline performs:

- Data ingestion
- Text cleaning and normalization
- Duplicate removal
- Cross-split leakage detection
- Feature generation
- Class distribution analysis
- Text length analysis
- Negation analysis
- SQLite data warehouse creation
- Analytical SQL queries
- Data dictionary generation

### Important EDA Findings

- The dataset contains a significant class imbalance.
- **Joy** is the largest class.
- **Surprise** is the smallest class.
- The largest-to-smallest class ratio is approximately **9.4:1**.
- The median message length is approximately **17 words**.
- Approximately **22.7%** of messages contain negation.

Because of the imbalance, macro F1 and per-class recall are important evaluation metrics.

---

## Machine Learning

### Classical NLP

The project evaluates:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Unigram features
- Bigram features
- Trigram features
- Class imbalance handling
- Probability calibration
- Threshold tuning
- Feature importance and error analysis

### Best Classical Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Linear SVM — Unigrams | 0.8355 | 0.7752 | 0.8339 |
| Logistic Regression — Unigrams | 0.8205 | 0.7788 | 0.8217 |
| Linear SVM — Bigrams | 0.8220 | 0.7373 | 0.8177 |

---

## Deep Learning

The project also includes an embedding-based neural network and an LSTM model.

### LSTM Architecture

```text
Text
 ↓
Text Vectorization
 ↓
Embedding
 ↓
LSTM
 ↓
Dense Layer
 ↓
Dropout
 ↓
6-Class Softmax
