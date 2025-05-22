# Real vs Fake News Detection

## 📌 Project Overview
This project is a machine learning-powered web application that detects whether a news article is real or fake. It aims to support fact-checking efforts and minimize the spread of misinformation by classifying news based on textual content using NLP techniques.

The web app is built using **Streamlit** and deployed online for public access.

🔗 **[Click here to open the app](https://your-username-your-app-name.streamlit.app/)**

---

## ❓ Problem Statement
In today’s digital world, the spread of fake news has become a significant issue, particularly on social media platforms. The main challenge lies in distinguishing between legitimate and deceptive news articles. This project addresses the problem by leveraging natural language processing and machine learning to detect fake news automatically.

---

## 🎯 Objectives
- To build machine learning models that can classify news as real or fake.
- To process and clean raw text data for effective model performance.
- To provide an interactive web interface for users to test the classifier.
- To deploy the solution as a web application using Streamlit.

---

## 🗂️ Data Source
The dataset used for this project was sourced from **Kaggle**:  
🔗 [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/razanaqvi14/real-and-fake-news)

It contains two separate files: `Fake.csv` and `True.csv`, representing fake and real news articles respectively.

---

## ⚙️ Methodology
1. **Data Collection**: Downloaded from Kaggle.
2. **Data Preprocessing**:
   - Removing stop words and punctuation
   - Tokenization and text normalization
3. **Model Building**:
   - TF-IDF Vectorization
   - Logistic Regression / Naive Bayes classifier
4. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1 Score
5. **Deployment**:
   - Built interactive UI using Streamlit
   - Deployed on Streamlit Cloud for public access

---

## 🧠 Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- NLTK
- Streamlit

---

## 📁 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/Real-vs-Fake-News-Detection.git
cd Real-vs-Fake-News-Detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
