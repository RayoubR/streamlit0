import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Clustered Q&A Assistant", page_icon="💬", layout="centered")

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

# --- LOAD EMBEDDING MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --- EMBED QUESTIONS ---
@st.cache_data
def embed_questions(questions):
    return model.encode(questions, convert_to_tensor=False)

question_texts = questions_df["text"].tolist()
question_embeddings = embed_questions(question_texts)

# --- CLUSTERING (KMeans) ---
NUM_CLUSTERS = 10  # Change depending on dataset size

@st.cache_data
def cluster_questions(embeddings, k=NUM_CLUSTERS):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels

kmeans_model, cluster_labels = cluster_questions(question_embeddings, NUM_CLUSTERS)
questions_df["cluster"] = cluster_labels

# --- UI ---
st.title("💬 Clustered Leasing Q&A Assistant")
st.markdown("Ask your leasing question. We'll match it to a cluster of related questions and return the best answer.")

user_question = st.text_input("🔍 Your question:")

if user_question:
    # Step 1: Embed user question
    user_embedding = model.encode([user_question])[0]

    # Step 2: Predict closest cluster
    cluster_idx = kmeans_model.predict([user_embedding])[0]

    # Step 3: Retrieve all Qs from that cluster
    cluster_questions = questions_df[questions_df["cluster"] == cluster_idx]
    cluster_embeddings = embed_questions(cluster_questions["text"].tolist())

    # Step 4: Find most similar question in that cluster
    similarities = cosine_similarity([user_embedding], cluster_embeddings)[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_question_row = cluster_questions.iloc[best_idx]

    # Step 5: Get response
    response_row = responses_df[responses_df["in_response_to_tweet_id"] == best_question_row["tweet_id"]]
    answer = response_row.iloc[0]["text"] if not response_row.empty else "No response found."

    # Confidence level
    if best_score >= 0.7:
        conf_color, conf_label = "#28a745", "High"
    elif best_score >= 0.3:
        conf_color, conf_label = "#ffc107", "Medium"
    else:
        conf_color, conf_label = "#dc3545", "Low"

    # --- DISPLAY ---
    st.markdown(f"### 🧠 Closest Matched Question (Cluster #{cluster_idx})")
    st.info(best_question_row["text"])

    st.markdown("### ✅ Answer")
    st.success(answer)

    st.markdown(f"""
    <div style='margin-top: 1rem; padding: 0.5rem 1rem; border-radius: 8px;
                background-color: {conf_color}; color: white; display: inline-block;'>
        Confidence: {conf_label} ({best_score:.2f})
    </div>
    """, unsafe_allow_html=True)

# --- OPTIONAL: Dataset Preview ---
with st.expander("📄 View Full Dataset"):
    st.dataframe(df)
