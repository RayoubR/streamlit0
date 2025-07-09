import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("Smart QA System - NLP Matching :)")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("sample.csv")
    df.dropna(subset=["Question", "Answer"], inplace=True)
    return df

df = load_data()

# Input question
user_question = st.text_input("Ask your question here:")

if user_question:
    # Combine user's question with dataset questions
    questions = df["Question"].tolist()
    questions.append(user_question)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)

    # Compute cosine similarity between user's question and dataset questions
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])

    # Find the most similar question
    most_similar_idx = cosine_similarities.argmax()
    similarity_score = cosine_similarities[0, most_similar_idx]

    # Get matched question and answer
    matched_question = df.iloc[most_similar_idx]["Question"]
    matched_answer = df.iloc[most_similar_idx]["Answer"]

    # Display results
    st.subheader("Matched Question:")
    st.write(matched_question)

    st.subheader("Answer:")
    st.write(matched_answer)

    st.info(f"Similarity score: {similarity_score:.2f}")

# Optional: Show dataset
with st.expander("Show dataset"):
    st.dataframe(df)
