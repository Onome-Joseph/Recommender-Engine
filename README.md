#  Netflix Movie & TV Show Recommender System

A content-based recommender system built using up-to-date data from Netflix movies and TV shows. The system leverages semantic similarity using Sentence-BERT embeddings to provide intelligent, personalized recommendations based on user input. It is lightweight, scalable, and can be adapted to fit any entertainment or content-based recommendation use case.

##  Features

-  Real-time semantic search using [Sentence-BERT](https://www.sbert.net/)
-  Customizable number of recommendations
-  Simple and responsive web interface using Flask + Bootstrap
-  Easily extendable to other platforms like Hulu, Prime Video, YouTube, Spotify, or eCommerce

![App Preview](https://github.com/Onome-Joseph/Recommender-Engine/blob/main/App%20preview/Screenshot%20(41).png)


##  Applications
- Enhance user engagement with smarter search and discovery
- Personalize recommendations for new users (cold start problem)
- Recommend similar products based on description
- Suggest similar tracks based on mood, genre, or descriptions
- Recommend courses, videos, or documents based on user queries


## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Onome-Joseph/Recommender-Engine.git
```
### 2. Install required packages
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```
### 4. Run the jupyter notebook to get the `embeddings.npy`
```bash
Recommend_engine.ipynb
```
### 3. Run the Flask App
```bash
app.ipynb or app.py
```
### How It Works
- User enters a query like:
"Funny high school TV shows"
- The model converts the query into an embedding using Sentence-BERT.
- It compares this with embeddings of all movie/show descriptions. 
- The most similar results are shown as recommendations
