# 📰 Fake News Detection

An NLP-powered web app that classifies news articles as **Real or Fake** using a trained ML model served via **FastAPI**.

---

## 🧠 How It Works

1. News text is submitted via POST request
2. Text is vectorized using **TF-IDF** (`vectorizer.pkl`)
3. Passed to a trained **ML Classifier** (`model.pkl`)
4. Returns a **Real / Fake** prediction as JSON

---

## ⚙️ Setup & Run

```bash
git clone https://github.com/mannithaj23aiml-oss/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
uvicorn backend:app --host 0.0.0.0 --port 8000
```

---

## 🛠️ Tech Stack
`Python` `Scikit-learn` `TF-IDF` `FastAPI` `Uvicorn` `Google Colab`
