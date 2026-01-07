# Movie Recommendation System
# Copy this entire code into app.py

import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Program started...")  # Check if Python is running

# ---------------------------
# Step 1: Load dataset
# ---------------------------
movies = pd.read_csv("movies.csv")  # Make sure your CSV is named movies.csv

# Keep only necessary columns
movies = movies[['title', 'overview', 'genres']]

# Fill missing values
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')

# ---------------------------
# Step 2: Convert genres to text
# ---------------------------
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return " ".join(L)

movies['genres'] = movies['genres'].apply(convert)

# ---------------------------
# Step 3: Create tags
# ---------------------------
movies['tags'] = movies['overview'] + " " + movies['genres']
movies = movies[['title', 'tags']]

# ---------------------------
# Step 4: Vectorize text
# ---------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# ---------------------------
# Step 5: Compute similarity
# ---------------------------
similarity = cosine_similarity(vectors)

# ---------------------------
# Step 6: Recommendation function
# ---------------------------
def recommend(movie_name):
    if movie_name not in movies['title'].values:
        print(f"Sorry! '{movie_name}' not found in the dataset.")
        return
    
    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\nMovies similar to '{movie_name}':")
    for i in movies_list:
        print(movies.iloc[i[0]].title)

# ---------------------------
# Step 7: Interactive movie input
# ---------------------------
print("\nYou can now type any movie title from the dataset.")
print("Here are some sample movies to try:")
print(movies['title'].head(10))  # Shows first 10 movie titles

movie_name = input("\nEnter a movie name exactly as shown above: ")
recommend(movie_name)
