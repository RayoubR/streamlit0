import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Leasing Q&A Bot", page_icon="💬", layout="wide")

# --- STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 2rem;
    }
    .question-box, .answer-box {
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .question-box {
        background-color: #e0f0ff;
        border-left: 5px solid #1f77b4;
    }
    .answer-box {
        background-color: #e8ffe8;
        border-left: 5px solid #2ca02c;
    }
    .similarity-bar > div {
        height: 24px;
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

# --- TITLE ---
st.title("💡 Leasing Q&A Assistant (v0.1.4)")
st.markdown("Ask a leasing-related question and get an instant answer powered by text similarity.")

# --- INPUT ---
user_question = st.text_input("🔎 Type your question here:")

# --- PROCESSING ---
if user_question:
    tfidf = TfidfVectorizer()
    question_texts = questions_df["text"].tolist()
    vectors = tfidf.fit_transform(question_texts + [user_question])
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_idx = cosine_similarities.argmax()
    best_match_score = cosine_similarities[0, best_match_idx]
    matched_tweet = questions_df.iloc[best_match_idx]
    response_row = responses_df[responses_df["in_response_to_tweet_id"] == matched_tweet["tweet_id"]]

    # --- DISPLAY MATCH ---
    st.markdown('<div class="question-box"><strong>🧠 Closest Match:</strong><br>' +
                matched_tweet["text"] + '</div>', unsafe_allow_html=True)

    # --- DISPLAY ANSWER ---
    st.markdown('<div class="answer-box"><strong>✅ Answer:</strong><br>', unsafe_allow_html=True)
    if best_match_score >= 0.30:
        if not response_row.empty:
            st.markdown(response_row.iloc[0]["text"], unsafe_allow_html=True)
        else:
            st.warning("⚠️ No response found for the matched question.")
    else:
        st.markdown("🤖 Please contact a human assistant for more information.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- SIMILARITY SCORE ---
    st.markdown("### 🔢 Similarity Score")
    st.progress(best_match_score)

# --- OPTIONAL DATASET VIEW ---
with st.expander("📊 View Raw Dataset"):
    st.dataframe(df)
