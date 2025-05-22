import streamlit as st
#import joblib
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import h5py



# Load tokenizer
with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
    tokenizer_json_str = json.dumps(tokenizer_data)
    tokenizer = tokenizer_from_json(tokenizer_json_str)

# # If tokenizer_data is a dict, convert it to a JSON string
# tokenizer_json_str = json.dumps(tokenizer_data)
# tokenizer = tokenizer_from_json(tokenizer_json_str)

# ------------------------------------------------------------ 
with h5py.File("lstm_model.h5", 'r') as f:
    print("Keys:", list(f.keys()))

# Load model
model = tf.keras.models.load_model("lstm_model.h5")

# Set max sequence length (must match training)
max_len = 200

def predict_news(news_text):
    seq = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    return "Real ✅" if prediction >= 0.5 else "Fake ❌"

st.title("Real vs Fake News Detector (LSTM)")

news_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if news_input.strip() != "":
        result = predict_news(news_input)
        st.success(f"This news is: {result}")
    else:
        st.warning("Please enter some news text.")


# # Load model and vectorizer
# model = joblib.load('model.pkl')
# #model = pickle.load(open('model.pkl', 'rb'))
# vectorizer = joblib.load(open('vectorizer.pkl', 'rb'))

# # UI
# st.title("Real vs Fake News Detector")
# text_input = st.text_area("Enter news text here:")

# if st.button("Classify"):
#     if text_input:
#         transformed = vectorizer.transform([text_input])
#         prediction = model.predict(transformed)[0]
#         st.success("This news is: " + ("Real ✅" if prediction == 1 else "Fake ❌"))
#     else:
#         st.warning("Please enter some text.")
