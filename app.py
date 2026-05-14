
import streamlit as st
import joblib

model = joblib.load("model/liar_model.pkl")

st.title("LIAR Fake News Detection")

text = st.text_area("Enter statement")

if st.button("Predict"):
    if text:
        pred = model.predict([text])[0]
        st.success(pred)
