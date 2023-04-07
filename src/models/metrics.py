import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import cross_validate, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Load your dataset

# Define a reader with a rating scale and the appropriate line format
# Replace 'user', 'item', and 'rating' with the actual column names from your dataset
reader = Reader(rating_scale=(1, 5), sep=',', skip_lines=1, rating_col=2)
data = Dataset.load_from_file('..\\..\\data\\processed\\processed_movies.csv', reader=reader)


# Split the dataset into a training set (80%) and a testing set (20%)
trainset, testset = train_test_split(data, test_size=0.2)

# Train the content-based filtering model
# (Replace this line with your actual content-based filtering implementation)
cbf_model = KNNBasic(sim_options={'user_based': False})
cbf_model.fit(trainset)

# Train the collaborative filtering model
cf_model = SVD()
cf_model.fit(trainset)

# Predict ratings using both models
cbf_predictions = cbf_model.test(testset)
cf_predictions = cf_model.test(testset)

# Combine the predictions for a hybrid approach
hybrid_predictions = []

for cbf_pred, cf_pred in zip(cbf_predictions, cf_predictions):
    hybrid_pred = (cbf_pred.est + cf_pred.est) / 2
    hybrid_predictions.append((cbf_pred.uid, cbf_pred.iid, hybrid_pred))

# Calculate RMSE
cbf_rmse = accuracy.rmse(cbf_predictions)
cf_rmse = accuracy.rmse(cf_predictions)
hybrid_rmse = accuracy.rmse(hybrid_predictions)

# Calculate precision, recall, and F1 score
# Set the threshold and top N recommendations
threshold = 3.5
top_n = 10

# Function to get top N recommendations
def get_top_n(predictions, n):
    top_n = {}
    for uid, iid, est in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get the top N recommendations for each user
cbf_top_n = get_top_n(cbf_predictions, top_n)
cf_top_n = get_top_n(cf_predictions, top_n)
hybrid_top_n = get_top_n(hybrid_predictions, top_n)

# Function to calculate precision, recall, and F1 score
def evaluate(top_n, testset, threshold):
    y_true = []
    y_pred = []

    for (uid, iid, true_rating) in testset:
        if uid in top_n:
            y_true.append(1 if true_rating >= threshold else 0)

            predicted_rating = None
            for item, est in top_n[uid]:
                if item == iid:
                    predicted_rating = est
                    break

            y_pred.append(1 if predicted_rating and predicted_rating >= threshold else 0)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

# Calculate
