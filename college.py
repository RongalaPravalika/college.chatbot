import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn

st.setpage_config(page_title="Svecw College Chatbot", layout="centered")

csv_url="svcew.details.csv"
try:
  df=pd.read_csv(csv_url)
except Exception as e:
  st.error(f"Failed to load the CSV file. Error: {e}")
  st.stop()
df=df.fillna("")
df['Question']=df['Question'].str.lower()
df=['Answer']=df['Answer'].str.lower()
vectorizer=TfidfVectorizer()
question_vectors=vectorizer.fit_transform(df['Question'])
