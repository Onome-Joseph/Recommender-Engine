import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    return pd.read_csv("final_data.csv")

@st.cache_data
def load_embeddings():
    return np.load("embeddings.npy")  # embeddings should be saved beforehand locally

def get_recommendations(query, embeddings, model, data, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return data.iloc[top_indices]


st.title("Movie Recommender System")

query = st.text_input("Enter your interest (e.g. 'comedy shows about school')")

if query:
    model = load_model()
    data = load_data()
    embeddings = load_embeddings()

    recommendations = get_recommendations(query, embeddings, model, data, top_n=10)
    st.subheader("Top Recommendations:")
    st.dataframe(recommendations)
