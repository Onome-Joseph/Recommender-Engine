import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model once and cache it
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load your dataset (cached)
@st.cache_data
def load_data():
    data = pd.read_csv("final_data.csv")
    data['full_description'] = data['full_description'].fillna('')
    return data

# Compute embeddings dynamically without caching input
def compute_embeddings(text_list, model):
    return model.encode(text_list, show_progress_bar=True)

# Recommendation function
def get_recommendations(query, embeddings, model, data, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

# Streamlit UI
st.title("🎬 Movie Recommender System")

query = st.text_input("Enter your interest (e.g., 'comedy shows about school')")

if query:
    with st.spinner("Loading model and data..."):
        model = load_model()
        data = load_data()

    with st.spinner("Generating embeddings..."):
        embeddings = compute_embeddings(data['full_description'].tolist(), model)

    with st.spinner("Fetching recommendations..."):
        recommendations = get_recommendations(query, embeddings, model, data, top_n=10)

    st.success("Here are your top recommendations:")
    st.dataframe(recommendations)
