# Fake News Detector

This is a web application built with **Streamlit** to detect fake news headlines and articles using a machine learning model. It uses a **Logistic Regression** classifier trained on TF-IDF features.

---

## Features

- Predict fake or true news from a single headline or article.
- Batch predict news from CSV files with columns `title` and/or `text`.
- View model insights and top contributing words.
- Download prediction results as CSV.
- Upload your own trained model and vectorizer.

---

## How it works

- User inputs or uploads news data.
- Text is cleaned and preprocessed.
- The TF-IDF vectorizer transforms the text into features.
- The Logistic Regression model predicts if news is *Fake* or *True*.
- Confidence scores and word clouds help interpret the results.

---

## Setup & Run

1. Clone the repo:
   ```bash
   git clone https://github.com/abdelrhman-m2/fake-news-detector.git
   cd fake-news-detector

## Files

- app.py — Main Streamlit app.

- fake_news_model.pkl — Pretrained Logistic Regression model.

- tfidf_vectorizer.pkl — Pretrained TF-IDF vectorizer.

- requirements.txt — Python dependencies.

- Fake_News_Model.ipynb — Model


## Install dependencies:

```bash
   pip install -r requirements.txt


## About
This project helps detect fake news using a machine learning approach to support media literacy and combat misinformation.
