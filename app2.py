import streamlit as st
import joblib
import json


# Load model and vectorizer
model = joblib.load('model.pkl')
#model = pickle.load(open('model.pkl', 'rb'))
vectorizer = joblib.load(open('vectorizer.pkl', 'rb'))

# UI
st.title("Real vs Fake News Detector")
text_input = st.text_area("Enter news text here:")

if st.button("Classify"):
    if text_input:
        transformed = vectorizer.transform([text_input])
        prediction = model.predict(transformed)[0]
        st.success("This news is: " + ("Real ✅" if prediction == 1 else "Fake ❌"))
    else:
        st.warning("Please enter some text.")