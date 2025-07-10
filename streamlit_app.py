import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- PAGE CONFIG ---
st.set_page_config(page_title="Clustered Q&A Assistant", page_icon="💬", layout="centered")

# --- STYLING ---
st.markdown("""
    <style>
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
    return SentenceTransformer("all-MiniLM-L6-v2")

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

# --- EMBED QUESTIONS ---
@st.cache_data
def embed_questions(questions):
    return model.encode(questions, convert_to_tensor=False)

question_texts = questions_df["text"].tolist()
question_embeddings = embed_questions(question_texts)

# --- CLUSTERING (KMeans) ---
NUM_CLUSTERS = 10  # Tune this based on dataset size

@st.cache_data
def cluster_questions(embeddings, k=NUM_CLUSTERS):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels

kmeans_model, cluster_labels = cluster_questions(question_embeddings, NUM_CLUSTERS)
questions_df["cluster"] = cluster_labels

# --- UI ---
st.title("💬 Clustered Leasing Q&A Assistant")
st.markdown("Ask your leasing question. We’ll match it to a cluster of related questions and return the best answer.")

user_question = st.text_input("🔍 Your question:")

if user_question:
    # Step 1: Embed the user's question
    user_embedding = model.encode([user_question])[0]

    # ✅ Step 2: Predict the closest cluster (with correct shape)
    cluster_idx = kmeans_model.predict(np.array(user_embedding).reshape(1, -1))[0]

    # Step 3: Get all questions in the same cluster
    cluster_questions = questions_df[questions_df["cluster"] == cluster_idx]
    cluster_embeddings = embed_questions(cluster_questions["text"].tolist())

    # Step 4: Find the most similar question within that cluster
    similarities = cosine_similarity([user_embedding], cluster_embeddings)[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_question_row = cluster_questions.iloc[best_idx]

    # Step 5: Get the corresponding answer
    response_row = responses_df[responses_df["in_response_to_tweet_id"] == best_question_row["tweet_id"]]
    answer = response_row.iloc[0]["text"] if not response_row.empty else "*No response found.*"

    # Confidence display
    if best_score >= 0.7:
        conf_class, conf_label = "high", "High"
    elif best_score >= 0.3:
        conf_class, conf_label = "medium", "Medium"
    else:
        conf_class, conf_label = "low", "Low"

    # --- DISPLAY RESULTS ---
    st.markdown(f"""
    <div class="card">
        <div class="title">Matched Question (Cluster #{cluster_idx})</div>
        <p>{best_question_row['text']}</p>
        <div class="title">Answer</div>
        <p>{answer}</p>
        <span class="score {conf_class}">Confidence: {conf_label} ({best_score:.2f})</span>
    </div>
    """, unsafe_allow_html=True)

# --- OPTIONAL: VIEW RAW DATASET ---
with st.expander("📄 View Raw Dataset"):
    st.dataframe(df)
