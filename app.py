import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

st.title("🎬 Movie Recommendation System")

# Movie selection dropdown
movie_list = movies['title'].tolist()
selected_movie = st.selectbox("Select a movie you like:", movie_list)

# Minimum rating slider
min_rating = st.slider("Minimum Rating ⭐", 0.0, 10.0, 6.0)

# Show recommendations button
if st.button("Show Recommendations"):
    # TF-IDF on 'overview'
    tfidf = TfidfVectorizer(stop_words='english')
    movies['overview'] = movies['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get index of selected movie
    idx = movies[movies['title'] == selected_movie].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar movies (excluding the selected one)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    # Filter by minimum rating
    recommended_movies = movies.iloc[movie_indices]
    recommended_movies = recommended_movies[recommended_movies['vote_average'] >= min_rating]
    
    st.subheader("You may also like:")
    for i, row in recommended_movies.iterrows():
        st.write(f"🎥 {row['title']} — ⭐ {row['vote_average']}")
