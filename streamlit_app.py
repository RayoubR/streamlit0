import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Leasing Q&A System", page_icon="🔍", layout="centered")

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        margin-top: 1rem;
        border-radius: 10px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #222;
    }
    .similarity-score {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        font-size: 0.9rem;
        border-radius: 20px;
        margin-top: 0.5rem;
        font-weight: 500;
        color: white;
    }
    .high { background-color: #4CAF50; }
    .medium { background-color: #FFC107; color: black; }
    .low { background-color: #f44336; }
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
st.title("Leasing Question Answering System")
st.markdown("Type your leasing-related question below. The system will try to match it with existing Q&A data.")

# --- USER INPUT ---
user_question = st.text_input("Enter your question:")

if user_question:
    tfidf = TfidfVectorizer()
    question_texts = questions_df["text"].tolist()
    vectors = tfidf.fit_transform(question_texts + [user_question])
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_idx = cosine_similarities.argmax()
    best_match_score = cosine_similarities[0, best_match_idx]
    matched_tweet = questions_df.iloc[best_match_idx]
    response_row = responses_df[responses_df["in_response_to_tweet_id"] == matched_tweet["tweet_id"]]

    # --- SIMILARITY CLASS ---
    if best_match_score >= 0.70:
        similarity_class = "high"
        confidence_label = "High Confidence"
    elif best_match_score >= 0.30:
        similarity_class = "medium"
        confidence_label = "Moderate Confidence"
    else:
        similarity_class = "low"
        confidence_label = "Low Confidence"

    # --- DISPLAY MATCHED QUESTION ---
    st.markdown(f"""
    <div class="card">
        <div class="section-title">Matched Question:</div>
        {matched_tweet["text"]}
    </div>
    """, unsafe_allow_html=True)

    # --- DISPLAY RESPONSE ---
    st.markdown(f"""
    <div class="card">
        <div class="section-title">Answer:</div>
        {"<i>No answer found.</i>" if response_row.empty else response_row.iloc[0]['text']}
    </div>
    """, unsafe_allow_html=True)

    # --- DISPLAY SIMILARITY ---
    st.markdown(f"""
    <div class="card">
        <div class="section-title">Similarity Score:</div>
        <span class="similarity-score {similarity_class}">{confidence_label}: {best_match_score:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

# --- OPTIONAL: SHOW DATASET ---
with st.expander("View Full Dataset"):
    st.dataframe(df)
