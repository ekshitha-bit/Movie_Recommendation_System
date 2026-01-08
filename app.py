import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies = load_data()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["overview"].fillna(""))

# Cosine Similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function with rating filter
def recommend(movie, min_rating):
    index = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i in distances[1:]:
        movie_data = movies.iloc[i[0]]
        if movie_data["vote_average"] >= min_rating:
            recommended_movies.append(
                (movie_data["title"], movie_data["vote_average"])
            )
        if len(recommended_movies) == 5:
            break

    return recommended_movies

# Streamlit UI
st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and minimum rating to get recommendations")

selected_movie = st.selectbox(
    "Select a movie you like",
    movies["title"].values
)

min_rating = st.slider(
    "Minimum IMDb Rating",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.5
)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie, min_rating)
    st.subheader("Recommended Movies:")
    
    if recommendations:
        for movie, rating in recommendations:
            st.write(f"👉 **{movie}** — ⭐ {rating}")
    else:
        st.warning("No movies found with the selected rating.")
