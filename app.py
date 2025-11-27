import pickle
import numpy as np
import streamlit as st
import base64

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
)

# ➤ Set Background Image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpg;base64,{encoded}");
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
         }}
         .main {{
             background-color: rgba(255,255,255,0.83) !important;
             padding: 20px;
             border-radius: 10px;
         }}
         textarea {{
             background-color: rgba(255,255,255,0.85) !important;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

set_background("newspaper.jpg")

# ============ TITLE BOX ============
st.markdown("""
<div style="text-align:center;
     background:rgba(255,255,255,0.85);
     padding:18px;
     border-radius:12px;
     border:2px solid #222;
     font-family:'Georgia';
     font-size:30px;
     font-weight: bold;">
📰 Real Time Fake News Detection System
</div>
""", unsafe_allow_html=True)


# ============ INPUT FIELD ============
text = st.text_area("Enter news content below:", height=200)


# ============ PREDICT BUTTON ============
if st.button("Analyze News"):
    if len(text.strip()) < 20:
        st.warning("Please enter longer text for meaningful analysis.")
    else:
        input_data = vectorizer.transform([text])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        fake_prob = probability[0]
        real_prob = probability[1]

        if prediction == 1:
            st.success("✔ REAL NEWS")
        else:
            st.error("✖ FAKE NEWS")

        confidence = max(fake_prob, real_prob) * 100
        st.write(f"Confidence Score: **{confidence:.2f}%**")
