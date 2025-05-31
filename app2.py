import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv("https://github.com/Onome-Joseph/Recommender-Engine/blob/main/recommender_dataset.csv")  # Replace with your dataset path

# Load sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute sentence embeddings
@st.cache_data
def compute_embeddings(data, model):
    return model.encode(data['full_description'].tolist(), show_progress_bar=True)

def get_recommendations(query, model, embeddings, final_data, top_n=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return final_data.iloc[top_indices]

# === Streamlit UI ===
st.set_page_config(page_title="Recommender System", layout="wide")
st.title("🎬 Content Recommendation Engine")

# Load resources
data = load_data()
model = load_model()
embeddings = compute_embeddings(data, model)

query = st.text_input("Enter a description or keywords of what you're looking for:")
top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations") and query:
    recommendations = get_recommendations(query, model, embeddings, data, top_n = top_n)
    
    st.subheader("Top Recommendations:")
    for i, row in recommendations.iterrows():
        st.markdown(f"**🎬 {row['title']}**")
        st.markdown(f"**Type:** {row['type']}  |  **Year:** {row['release_year']}")
        st.markdown(f"**Plot:** {row['full_description']}")
        st.markdown(f"**Rating:** {row['rating']}")
        st.markdown("---")
