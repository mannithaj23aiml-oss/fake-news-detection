import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Real Time Fake News Detection System")

# ------------------ BACKGROUND ------------------
def set_background_url(url):
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
         }}
         textarea {{
             background-color: rgba(255,255,255,0.85) !important;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

set_background_url("https://i.imgur.com/qzQjL1z.jpeg")
# -------------------------------------------------

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.markdown("""
<div style="text-align:center; background: rgba(0,0,0,0.6); padding:20px; border-radius:10px;">
    <h1 style="color:white;">Real Time Fake News Detection System</h1>
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

        fake_prob = probability[0]
        real_prob = probability[1]

        if prediction == 1:
            st.success(f"REAL NEWS")
        else:
            st.error(f"FAKE NEWS")

        confidence = max(fake_prob, real_prob) * 100
        st.write(f"Confidence Score: {confidence:.2f}%")
