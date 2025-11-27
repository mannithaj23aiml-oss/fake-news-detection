import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Real Time Fake News Detection System")

# ------------------ PAGE STYLING ------------------
st.markdown("""
<style>
body {
    background-color: #f2f2f2;
}
.stApp {
    background-color: #f2f2f2 !important;
}
.header-box {
    background: #0a2e6c;
    padding: 25px;
    border-radius: 8px;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 30px;
}
.input-box {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    margin-bottom: 30px;
}
.result-box {
    font-weight: bold;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------

# Header
st.markdown("<div class='header-box'>Real Time Fake News Detection System</div>", unsafe_allow_html=True)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.markdown("<div class='input-box'>", unsafe_allow_html=True)
text = st.text_area("Enter news content here:")
st.markdown("</div>", unsafe_allow_html=True)

if st.button("Predict Now"):
    if len(text.strip()) < 20:
        st.warning("Please enter a larger news text for analysis.")
    else:
        input_data = vectorizer.transform([text])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        fake_prob = probability[0]
        real_prob = probability[1]

        if prediction == 1:
            st.markdown("<div class='result-box' style='background:#27ae60;color:white;'>REAL NEWS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box' style='background:#e74c3c;color:white;'>FAKE NEWS</div>", unsafe_allow_html=True)

        confidence = max(fake_prob, real_prob) * 100
        st.write(f"Confidence Score: {confidence:.2f}%")
