import pickle
import numpy as np
import streamlit as st

# ------------------------------
# Load model & vectorizer
# ------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="Real Time Fake News Detection System",
    layout="centered"
)

st.markdown("""
<div style="text-align:center; background: linear-gradient(135deg, #ffb3b3, #ff8080); padding:30px; border-radius:12px;">
    <h1 style="color:white;">Real Time Fake News Detection System</h1>
    <p style="color:white;">Enter any news article content and get real-time prediction.</p>
</div>
""", unsafe_allow_html=True)

text = st.text_area("Enter news content here:")

if st.button("Predict Now"):
    if len(text.strip()) < 20:
        st.warning("Please enter a larger news text for analysis.")
    else:
        input_data = vectorizer.transform([text])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        fake_prob = float(probability[0])
        real_prob = float(probability[1])

        if prediction == 1:
            st.success("REAL NEWS")
        else:
            st.error("FAKE NEWS")

        confidence = max(fake_prob, real_prob) * 100
        st.write(f"Confidence Score: {confidence:.2f}%")
