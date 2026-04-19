import streamlit as st
import pickle

st.title("Spam Detection using Deep Learning")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

text = st.text_input("Enter message")

if text:
    data = vectorizer.transform([text])
    pred = model.predict(data)[0]

    if pred == 1:
        st.error("Spam Message 🚫")
    else:
        st.success("Not Spam ✅")
