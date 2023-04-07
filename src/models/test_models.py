import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from surprise import Dataset, Reader, dump
from hybrid import hybrid_recommendations
from tensorflow.keras.models import load_model
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(predictions, threshold, k):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    f1_scores = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in user_ratings[:k])
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = k

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        f1_scores[uid] = 2 * (precisions[uid] * recalls[uid]) / (precisions[uid] + recalls[uid]) if (precisions[uid] + recalls[uid]) != 0 else 0

    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)
    f1 = sum(f1_scores.values()) / len(f1_scores)

    return precision, recall, f1


# Load data and models
movies_df = pd.read_csv("..\\..\\data\\processed\\processed_movies.csv")
with open("..\\..\\data\\processed\\nearest_neighbors_model.pkl", "rb") as f:
    nearest_neighbors_model = pickle.load(f)
tfidf_matrix = np.load("tfidf_matrix.npy")

reader = Reader(rating_scale=(0, 5))
ratings_df = pd.read_csv("..\\..\\data\\interim\\cleaned_ratings.csv")
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

_, svd_model = dump.load("svd_model.dump")

matrix_fact_model = load_model("best_matrix_factorization_model.h5")

# Test models
movie_titles_list = movies_df["original_title"].tolist()
valid_movie = False

user_id = input("Enter user ID (pick any number between 1-50): ")
while not valid_movie:
    movie_title=input("Enter a movie title: ")
    if movie_title in movie_titles_list:
        valid_movie = True
    else:
        print("Invalid movie title. Please try again.")

# Content-Based Filtering
_, indices = nearest_neighbors_model.kneighbors(tfidf_matrix[movies_df.index[movies_df['title'] == movie_title].tolist()[0]].reshape(1, -1), n_neighbors=11)
content_based_recommendations = movies_df.iloc[indices.flatten()]
content_based_recommendations = content_based_recommendations[content_based_recommendations['title'] != movie_title]


# print("Content-Based Filtering Recommendations:")
# print(content_based_recommendations)
print("Content-Based Filtering Recommendations:")
for title in content_based_recommendations["original_title"].head(10):
    print(title)


# Collaborative Filtering
movies_not_watched = movies_df[~movies_df["movieId"].isin(ratings_df[ratings_df["userId"] == user_id]["movieId"])]
cf_predictions = []
for _, row in movies_not_watched.iterrows():
    prediction = svd_model.predict(user_id, row["movieId"]).est
    cf_predictions.append(prediction)
movies_not_watched["predicted_rating"] = cf_predictions
collaborative_filtering_recommendations = movies_not_watched.sort_values(by="predicted_rating", ascending=False).head(10)

# print("Collaborative Filtering Recommendations:")
# print(collaborative_filtering_recommendations)

print("\nCollaborative Filtering Recommendations:")
for title in collaborative_filtering_recommendations["original_title"].head(10):
    print(title)


# Hybrid
hybrid_recommendations = hybrid_recommendations(user_id, movie_title, movies_df, svd_model, nearest_neighbors_model, tfidf_matrix)

# print("Hybrid Recommendations:")
# print(hybrid_recommendations)

print("\nHybrid Recommendations:")
for title in hybrid_recommendations["original_title"].head(10):
    print(title)


# Calculate precision, recall, and F1-score for the content-based method
# NOTE: Content-based filtering doesn't use collaborative filtering predictions, so we can't directly use the compute_metrics function here.

# Calculate precision, recall, and F1-score for the collaborative filtering method
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
predictions = svd_model.test(testset)
threshold = 3
k = 10
precision, recall, f1 = compute_metrics(predictions, threshold, k)

print("Collaborative Filtering Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Calculate precision, recall, and F1-score for the hybrid method
# As the hybrid method uses the SVD model for collaborative filtering, we can use the same predictions.
precision, recall, f1 = compute_metrics(predictions, threshold, k)

print("Hybrid Method Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
