import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np

def create_combined_features(df, feature_columns):
    combined_features = df[feature_columns].apply(lambda x: " ".join(x.astype(str)), axis=1)
    return combined_features

def train_nearest_neighbors_model(tfidf_matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    return model

def get_similar_movies(model, tfidf_matrix, movie_index, movies_df, k=10):
    distances, indices = model.kneighbors(tfidf_matrix[movie_index], n_neighbors=k+1)
    similar_movies = movies_df.iloc[indices.flatten()]
    similar_movies = similar_movies[similar_movies.index != movie_index]
    return similar_movies


def main():
    PROCESSED_DATA_PATH = '../../data/processed/'

    # Load the preprocessed data
    processed_movies_file = os.path.join(PROCESSED_DATA_PATH, "processed_movies.csv")
    movies_df = pd.read_csv(processed_movies_file)

    # Select the features to use for content-based filtering
    feature_columns = ['genres', 'original_language']

    # Create combined features
    combined_features = create_combined_features(movies_df, feature_columns)

    # Generate the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_features)

    # Train the NearestNeighbors model
    model = train_nearest_neighbors_model(tfidf_matrix)

    # Save the trained model
    model_file = os.path.join(PROCESSED_DATA_PATH, "nearest_neighbors_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    np.save("tfidf_matrix.npy", tfidf_matrix.toarray())

    # Test the model: Find the 10 most similar movies for a given movie_title
    movie_title = "Toy Story"

    try:
        movie_index = movies_df.index[movies_df['title'].str.contains(movie_title)].tolist()[0]
        similar_movies = get_similar_movies(model, tfidf_matrix, movie_index, movies_df, k=10)
        print(f"Movies similar to {movies_df.iloc[movie_index]['title']}:")
        print(similar_movies[['title', 'genres']])
    except IndexError:
        print(f"Movie title '{movie_title}' not found in the preprocessed data.")

if __name__ == "__main__":
    main()
