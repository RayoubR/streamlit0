import streamlit as st
import pandas as pd 
st.title('test chatbot')
df = pd.read_csv('https://raw.githubusercontent.com/RayoubR/streamlit0/refs/heads/master/sample.csv')
st.write('Hello world!')
df['response_tweet_id'] = df['response_tweet_id'].fillna(1)
df.tail(5)
