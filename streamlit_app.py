import streamlit as st
import pandas as pd 
st.title('test chatbot')
df = pd.read_csv('https://github.com/RayoubR/streamlit0/blob/master/sample.csv')
st.write('Hello world!')
df
