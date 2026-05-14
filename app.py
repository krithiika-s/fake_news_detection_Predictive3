import streamlit as st
import joblib

model = joblib.load("model/liar_model.pkl")

st.title("📰 LIAR Fake News Detection App")

user_input = st.text_area("Enter a statement")

if st.button("Predict"):
    if user_input:
        pred = model.predict([user_input])[0]
        st.success(f"Prediction: {pred}")
    else:
        st.warning("Please enter text")
