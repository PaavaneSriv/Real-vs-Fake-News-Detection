# Real vs Fake News Detection

## Project Overview
This project is a machine learning-powered web application that detects whether a news article is real or fake. It aims to support fact-checking efforts and minimize the spread of misinformation by classifying news based on textual content using NLP techniques. Implemenetd four ML models and one LSTM-based model, and evaluated their performance on two different datasets.

The web app is built using **Streamlit**.

## Problem Statement
In today’s digital world, the spread of fake news has become a significant issue, particularly on social media platforms. The main challenge lies in distinguishing between legitimate and deceptive news articles. This project addresses the problem by leveraging natural language processing and machine learning to detect fake news automatically.

## Objectives

- Train and evaluate ML and DL models for fake news classification.
- Compare their performance across different datasets.
- Build user-friendly Streamlit apps for real-time testing.

## Data Sources

- **ML Dataset**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **LSTM Dataset**: Sourced from various internet repositories and cleaned for consistency.

## Methodology

### 1. **ML Models**
- Models used: Naive Bayes, Logistic Regression, Random Forest, Linear SVM.
- Vectorization: TF-IDF
- Evaluation: Accuracy, precision, recall, F1-score
- Best Model: Linear SVM (accuracy: **81.80%**)

### 2. **LSTM Model**
- Embedding layer + LSTM architecture
- Tokenization and padding applied
- Evaluated on both datasets
- Despite accuracy being slightly lower (**~79.86%**), the LSTM showed better generalization and stability across datasets

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- NLTK
- Streamlit

## Results
- While **Linear SVM** scored highest accuracy on the Kaggle dataset, the **LSTM** model demonstrated more reliable performance on both datasets.
- This suggests the LSTM's strength in capturing deeper text dependencies that traditional models might miss.

## 🌐 Live Apps

- **ML Models Web App**: Streamlit ML App
- **LSTM Model Web App**: Streamlit LSTM App

  
## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/Real-vs-Fake-News-Detection.git
cd Real-vs-Fake-News-Detection

# Install dependencies
pip install -r requirements.txt

# Run ML app
cd ML models app
streamlit run app.py

# Run LSTM app
cd LSTM app
streamlit run LSTMapp.py

#Compatibility Warning
#TensorFlow Compatibility Note
#TensorFlow requires a 64-bit version of Python and is compatible with Python 3.8 to 3.11. Make sure your environment meets these requirements. TensorFlow will not install or run properly on 32-bit systems or unsupported Python versions.
