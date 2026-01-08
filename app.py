import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

movies = load_data()

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["overview"].fillna(""))

# Compute similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(movie_name):
    try:
        index = movies[movies["title"] == movie_name].index[0]
        distances = similarity[index]

        movie_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        return [movies.iloc[i[0]]["title"] for i in movie_list]
    except:
        return []

# Streamlit UI
st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie to get similar recommendations")

selected_movie = st.selectbox(
    "Select a movie you like",
    movies["title"].values
)

if st.button("Show Recommendations"):
    recommendations = recommend(selected_movie)

    if recommendations:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write("👉", movie)
    else:
        st.warning("No recommendations found.")
