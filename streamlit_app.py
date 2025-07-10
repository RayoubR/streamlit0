import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Leasing Q&A Assistant", page_icon="💬", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    body {
        background-color: #f6f9fc;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
        padding: 10px;
    }
    .card {
        background-color: #fff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    .title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #333;
    }
    .similarity {
        display: inline-block;
        margin-top: 10px;
        padding: 0.4rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: white;
        border-radius: 25px;
    }
    .high { background-color: #198754; }
    .medium { background-color: #ffc107; color: #000; }
    .low { background-color: #dc3545; }
    .footer {
        margin-top: 3rem;
        font-size: 0.85rem;
        color: #888;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/RayoubR/streamlit0/refs/heads/master/Classeur1.csv")
    df.dropna(subset=["text"], inplace=True)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
    return df

df = load_data()
questions_df = df[df["inbound"] == True]
responses_df = df[df["inbound"] == False]

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>Leasing Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask your leasing-related question. We'll find the most relevant answer based on existing Q&A data.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- INPUT ---
user_question = st.text_input("🔍 What’s your question about leasing?")

if user_question:
    # TF-IDF vectorization
    tfidf = TfidfVectorizer()
    question_texts = questions_df["text"].tolist()
    vectors = tfidf.fit_transform(question_texts + [user_question])

    # Cosine similarity
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_idx = cosine_similarities.argmax()
    best_match_score = cosine_similarities[0, best_match_idx]
    matched_question = questions_df.iloc[best_match_idx]
    response_row = responses_df[responses_df["in_response_to_tweet_id"] == matched_question["tweet_id"]]

    # Confidence label
    if best_match_score >= 0.70:
        conf_class = "high"
        conf_label = "High Confidence"
    elif best_match_score >= 0.30:
        conf_class = "medium"
        conf_label = "Moderate Confidence"
    else:
        conf_class = "low"
        conf_label = "Low Confidence"

    # --- DISPLAY RESULTS ---
    st.markdown(f"""
    <div class="card">
        <div class="title">🧠 Most Similar Question</div>
        <p>{matched_question['text']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div class="title">💬 Answer</div>
        <p>{response_row.iloc[0]['text'] if not response_row.empty else "No response found for this question."}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div class="title">📊 Similarity Score</div>
        <span class="similarity {conf_class}">{conf_label} — {best_match_score:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

# --- RAW DATASET ---
with st.expander("📄 View Full Dataset (Raw CSV)"):
    st.dataframe(df)

# --- FOOTER ---
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & scikit-learn • Version 0.1.4</div>", unsafe_allow_html=True)
