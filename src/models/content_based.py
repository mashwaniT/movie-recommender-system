import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def create_combined_features(df, feature_columns):
    combined_features = df[feature_columns].apply(lambda x: " ".join(x.astype(str)), axis=1)
    return combined_features

def train_nearest_neighbors_model(tfidf_matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    return model

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
import pickle
model_file = os.path.join(PROCESSED_DATA_PATH, "nearest_neighbors_model.pkl")
with open(model_file, "wb") as f:
    pickle.dump(model, f)
