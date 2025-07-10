import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Leasing Q&A", page_icon="💬", layout="centered")

# --- STYLING ---
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .score {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #fff;
    }
    .high { background-color: #198754; }
    .medium { background-color: #ffc107; color: black; }
    .low { background-color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/RayoubR/streamlit0/refs/heads/master/Classeur1.csv")
    df.dropna(subset=["text"], inplace=True)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
    return df

df = load_data()
questions_df = df[df["inbound"] == True].reset_index(drop=True)
responses_df = df[df["inbound"] == False]

# --- EMBEDDINGS ---
@st.cache_data
def embed_questions(text_list):
    return model.encode(text_list, convert_to_tensor=True)

question_texts = questions_df["text"].tolist()
question_embeddings = embed_questions(question_texts)

# --- INTERFACE ---
st.title("💡 Leasing Q&A Assistant")
st.markdown("Type your leasing question below. We'll find the most relevant match.")

user_question = st.text_input("🔍 Ask a question:")

if user_question:
    # Encode input question
    query_embedding = model.encode(user_question, convert_to_tensor=True)

    # Find most similar question
    cosine_scores = util.cos_sim(query_embedding, question_embeddings)
    top_k = torch.topk(cosine_scores, k=3)

    for rank, (score, idx) in enumerate(zip(top_k[0][0], top_k[1][0])):
        matched_q = questions_df.iloc[idx]
        response_row = responses_df[responses_df["in_response_to_tweet_id"] == matched_q["tweet_id"]]
        answer = response_row.iloc[0]["text"] if not response_row.empty else "*No response found.*"

        # Confidence class
        score_val = score.item()
        if score_val >= 0.7:
            conf_class = "high"
            label = "High Confidence"
        elif score_val >= 0.3:
            conf_class = "medium"
            label = "Moderate Confidence"
        else:
            conf_class = "low"
            label = "Low Confidence"

        st.markdown(f"""
        <div class="card">
            <div class="title">Matched Question #{rank+1}</div>
            <p>{matched_q['text']}</p>
            <div class="title">Answer:</div>
            <p>{answer}</p>
            <span class="score {conf_class}">{label}: {score_val:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

# --- RAW DATA EXPANDER ---
with st.expander("📄 View Full Dataset"):
    st.dataframe(df)
