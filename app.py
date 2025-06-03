from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data and model
final_data = pd.read_csv("final_data.csv")
embeddings = np.load("embeddings.npy")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Flask app
app = Flask(__name__)

def get_recommendations(query, embeddings, model, data, top_n=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    query = ""
    top_n = 10  # default

    if request.method == 'POST':
        query = request.form['query']
        try:
            top_n = int(request.form.get('top_n'))
        except ValueError:
            top_n = 10

        recommendations = get_recommendations(query, embeddings, model, final_data, top_n=top_n)
    return render_template('front6.html', recommendations=recommendations, query=query, top_n=top_n)

if __name__ == '__main__':
    import os
    os.environ['FLASK_ENV'] = 'development'
    app.run(debug=True, use_reloader=False)

