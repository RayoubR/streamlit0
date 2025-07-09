import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("🤖 Leasing AI Assistant :°)")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/RayoubR/streamlit0/refs/heads/master/sample.csv")

df = load_data()

# Clean the question text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['clean_question'] = df['question'].astype(str).apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_question'])

# Find the best match using cosine similarity
def find_best_match(user_input):
    user_input_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    similarities = cosine_similarity(user_vec, X)
    index = similarities.argmax()
    return df.iloc[index]['answer'], df.iloc[index]['question']

# User input
st.subheader("💬 Ask a question about leasing:")
user_input = st.text_input("Type your question here")

# Answer section
if user_input:
    answer, matched_q = find_best_match(user_input)
    st.markdown("### 🔍 Matched Question:")
    st.info(matched_q)
    st.markdown("### 💡 Answer:")
    st.success(answer)

# Optional: Preview data
with st.expander("🔎 Preview Dataset"):
    st.dataframe(df[['question', 'answer']].head(10))

