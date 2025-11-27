import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Real Time Fake News Detection System", layout="centered")

# ------------------ PAGE STYLING ------------------
st.markdown("""
<style>

.stApp {
    background-image: url("https://wallpaperaccess.com/full/1567665.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.overlay-box {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    padding: 25px;
    border-radius: 12px;
    margin: auto;
    width: 70%;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

.title-text {
    text-align: center;
    font-size: 30px;
    padding-bottom: 10px;
    font-weight: 700;
    color: #0a2e6c;
}

textarea {
    background-color: rgba(255,255,255,0.9) !important;
    border-radius: 10px !important;
    border: 1px solid #cccccc !important;
}

button {
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# UI Layout Container
with st.container():
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("<div class='overlay-box'>", unsafe_allow_html=True)

    st.markdown("<div class='title-text'>Real Time Fake News Detection System</div>", unsafe_allow_html=True)

    text = st.text_area("Enter news content here:")

    if st.button("Predict Now"):
        if len(text.strip()) < 20:
            st.warning("Please enter a longer text sample.")
        else:
            input_data = vectorizer.transform([text])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            fake_prob = probability[0]
            real_prob = probability[1]
            confidence = max(fake_prob, real_prob) * 100

            if prediction == 1:
                st.success(f"REAL NEWS")
            else:
                st.error(f"FAKE NEWS")

            st.write(f"Confidence Score: {confidence:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)
