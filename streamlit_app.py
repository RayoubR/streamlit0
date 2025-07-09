import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("sample.csv")
    df.dropna(subset=["text"], inplace=True)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"
    return df

df = load_data()

# Separate inbound (questions) and outbound (responses)
questions_df = df[df["inbound"] == True]
responses_df = df[df["inbound"] == False]

# User input
st.title("Tweet-Based Question Answering System")
user_question = st.text_input("Enter your question:")

if user_question:
    # Build TF-IDF vectors for the inbound questions
    tfidf = TfidfVectorizer()
    question_texts = questions_df["text"].tolist()
    vectors = tfidf.fit_transform(question_texts + [user_question])

    # Compute similarity
    cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
    best_match_idx = cosine_similarities.argmax()
    best_match_score = cosine_similarities[0, best_match_idx]
    matched_tweet = questions_df.iloc[best_match_idx]

    # Find response to matched tweet
    response_row = responses_df[responses_df["in_response_to_tweet_id"] == matched_tweet["tweet_id"]]

    # Display results
    st.subheader("Most Similar Question:")
    st.write(matched_tweet["text"])

    st.subheader("Answer:")
    if best_match_score >= 0.80:
        if not response_row.empty:
            st.write(response_row.iloc[0]["text"])
        else:
            st.warning("No response found for the matched question.")
    else:
        st.write("Can you call the assistant for more info")

    st.info(f"Similarity Score: {best_match_score:.2f}")

# Optional: Expand to show full dataset
with st.expander("See raw dataset"):
    st.dataframe(df)
