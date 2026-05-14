import streamlit as st
import joblib

# Load trained model
model = joblib.load("models/model.pkl")

# App title
st.title("📰 Fake News Detection App")

st.write("Enter a news statement to classify it")

# User input
user_input = st.text_area("Enter statement")

# Prediction button
if st.button("Predict"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter a statement")
