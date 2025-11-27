import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Real Time Fake News Detection System")

# ------------------ BACKGROUND ------------------
# GitHub raw image link
BG_URL = "https://raw.githubusercontent.com/mannithaj23aiml-oss/fake-news-detection/main/Screenshot%202025-11-27%20163912.jpg"

st.markdown(f"""
     <style>
     .stApp {{
         background-image: url("{BG_URL}");
         background-size: cover;
         background-position: center;
         background-repeat: no-repeat;
     }}
     .header-box {{
         background: rgba(10,46,108,0.90);
         padding: 25px;
         border-radius: 8px;
         text-align: center;
         color: white;
         font-size: 28px;
         font-weight: bold;
         margin-bottom: 30px;
         backdrop-filter: blur(4px);
     }}
     .textbox {{
         background: rgba(255,255,255,0.85) !important;
         border-radius: 10px;
         padding: 10px;
         backdrop-filter: blur(4px);
     }}
     .result-box {{
         font-weight: bold;
         padding: 12px;
         border-radius: 8px;
         text-align: center;
         font-size: 20px;
     }}
     </style>
""", unsafe_allow_html=True)
# --------------------------------------------------

# Header
st.markdown("<div class='header-box'>Real Time Fake News Detection System</div>", unsafe_allow_html=True)


# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Input
text = st.text_area("Enter news content here:", key="textbox", height=200)

# Predict
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
