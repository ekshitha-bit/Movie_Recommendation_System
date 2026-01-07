import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['title', 'overview', 'vote_average']]
    df.dropna(inplace=True)
    return df

movies = load_data()

# -----------------------------
# VECTORIZE TEXT
# -----------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

similarity = cosine_similarity(tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(movie_title, min_rating):
    idx = indices[movie_title]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in sim_scores[1:]:
        if movies.iloc[i[0]]['vote_average'] >= min_rating:
            recommendations.append(movies.iloc[i[0]])
        if len(recommendations) == 5:
            break

    return recommendations

# -----------------------------
# UI INPUTS
# -----------------------------
selected_movie = st.selectbox(
    "Select a movie you like:",
    movies['title'].values
)

min_rating = st.slider(
    "Minimum Rating ⭐",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.5
)

# -----------------------------
# BUTTON + OUTPUT
# -----------------------------
if st.button("Show Recommendations"):
    results = recommend(selected_movie, min_rating)

    st.subheader("You may also like:")
    for movie in results:
        st.write(f"🎥 **{movie['title']}** — ⭐ {movie['vote_average']}")
