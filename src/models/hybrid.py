import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import SVD
from surprise import Dataset, Reader
import pickle

def hybrid_recommendations(user_id, movie_title, movies_df, svd_model, nearest_neighbors_model, tfidf_matrix, k=10, alpha=0.5):
    # Get movie_index from movie_title
    movie_index = movies_df.index[movies_df['title'] == movie_title].tolist()[0]

    # Step 1: Get content-based similarity scores
    distances, indices = nearest_neighbors_model.kneighbors(tfidf_matrix[movie_index].reshape(1, -1), n_neighbors=k+1)
    indices = indices.flatten()[1:]  # Exclude the movie itself from the indices

    content_based_similar_movies = movies_df.iloc[indices].copy()
    content_based_similar_movies.loc[:, 'cosine_similarity'] = 1 - distances.flatten()[1:]  # Calculate the cosine similarity from the distance

    # Step 2: Predict collaborative filtering scores
    cf_predictions = []
    for _, row in content_based_similar_movies.iterrows():
        prediction = svd_model.predict(user_id, row["movieId"]).est
        cf_predictions.append(prediction)
    content_based_similar_movies.loc[:, "cf_predictions"] = cf_predictions

    # Step 3: Combine both sets of scores
    content_based_similar_movies.loc[:, "hybrid_score"] = (
        alpha * content_based_similar_movies["cf_predictions"] +
        (1 - alpha) * content_based_similar_movies["cosine_similarity"]
    )

    # Step 4: Sort the combined scores and select the top recommendations
    recommendations = content_based_similar_movies.sort_values(by="hybrid_score", ascending=False).head(k)

    return recommendations



# Load data, models, and matrix
movies_df = pd.read_csv("..\\..\\data\\processed\\processed_movies.csv")
with open("..\\..\\data\\processed\\nearest_neighbors_model.pkl", "rb") as f:
    nearest_neighbors_model = pickle.load(f)
tfidf_matrix = np.load("tfidf_matrix.npy")
reader = Reader(rating_scale=(0, 5))
ratings_df = pd.read_csv("..\\..\\data\\interim\\cleaned_ratings.csv")
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd_model = SVD()
svd_model.fit(trainset)

# Example usage
# user_id = 1
# movie_title = "Toy Story"
# recommended_movies = hybrid_recommendations(user_id, movie_title, movies_df, svd_model, nearest_neighbors_model, tfidf_matrix)
# print(recommended_movies[['title', 'genres', 'cosine_similarity']])
