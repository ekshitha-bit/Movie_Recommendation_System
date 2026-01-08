import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie you like and get similar movie recommendations.")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies = load_data()

# -------------------------------
# Vectorization & similarity
# -------------------------------
@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["overview"].fillna(""))
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity = compute_similarity(movies)

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie_name):
    try:
        index = movies[movies["title"] == movie_name].index[0]
        distances = similarity[index]

        movie_list = sorted(
