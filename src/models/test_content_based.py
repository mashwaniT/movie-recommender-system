import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_DATA_PATH = '../../data/processed/'

# Load the preprocessed data
processed_movies_file = os.path.join(PROCESSED_DATA_PATH, "processed_movies.csv")
movies_df = pd.read_csv(processed_movies_file)

# Load the trained model
model_file = os.path.join(PROCESSED_DATA_PATH, "nearest_neighbors_model.pkl")
with open(model_file, "rb") as f:
    model = pickle.load(f)

# Generate the TF-IDF matrix for the combined features
feature_columns = ['genres', 'original_language']
combined_features = movies_df[feature_columns].apply(lambda x: " ".join(x.astype(str)), axis=1)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_features)

def get_movie_recommendations(movie_title, k=10):
    movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[movie_idx], n_neighbors=k + 1)
    recommended_movie_indices = indices.flatten()[1:]
    return movies_df.iloc[recommended_movie_indices]

# Test the recommendation system with an example movie
example_movie = "The Dark Knight"
print(f"Top recommendations for '{example_movie}':")
recommended_movies = get_movie_recommendations(example_movie)
print(recommended_movies[['title', 'genres', 'original_language']])
