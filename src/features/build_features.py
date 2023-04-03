import os
import pandas as pd

INTERIM_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'interim')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed')

# Read cleaned data
cleaned_movies_file = os.path.join(INTERIM_DATA_PATH, "cleaned_movies_metadata.csv")

cleaned_ratings_file = os.path.join(INTERIM_DATA_PATH, "cleaned_ratings.csv")

movies_df = pd.read_csv(cleaned_movies_file)
ratings_df = pd.read_csv(cleaned_ratings_file)

# Feature Engineering

# Calculate the average rating for each movie
average_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
average_ratings.columns = ['movieId', 'average_rating']

# Calculate the number of ratings for each movie
rating_counts = ratings_df.groupby('movieId')['rating'].count().reset_index()
rating_counts.columns = ['movieId', 'num_ratings']

# print(movies_df.columns)

# Merge the average ratings and rating counts with the movies_df
movies_df = pd.merge(movies_df, average_ratings, left_on='id', right_on='movieId', how='left')

movies_df = pd.merge(movies_df, rating_counts, on='movieId', how='left')

# Fill missing values with zeros (for movies with no ratings)
movies_df['average_rating'].fillna(0, inplace=True)
movies_df['num_ratings'].fillna(0, inplace=True)

# Save the feature engineered data
# Save the feature engineered data
features_file = os.path.join(PROCESSED_DATA_PATH, "features.csv")
movies_df.to_csv(features_file, index=False)


print("Feature engineering completed.")
