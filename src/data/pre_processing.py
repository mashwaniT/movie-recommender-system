import pandas as pd
import numpy as np
import os

# Read raw data
RAW_DATA_PATH = "..\\..\\data\\raw"
metadata_file = os.path.join(RAW_DATA_PATH, "movies_metadata.csv")
metadata_df = pd.read_csv(metadata_file, low_memory=False)

# Drop unnecessary columns
columns_to_drop = ['adult', 'belongs_to_collection', 'homepage', 'imdb_id', 'poster_path', 'video']
metadata_df.drop(columns_to_drop, axis=1, inplace=True)

# Convert 'budget' column to numeric
metadata_df['budget'] = pd.to_numeric(metadata_df['budget'], errors='coerce')

# Replace empty strings in 'original_language' with NaN
metadata_df['original_language'] = metadata_df['original_language'].replace('', np.nan)

# Convert 'release_date' column to datetime
metadata_df['release_date'] = pd.to_datetime(metadata_df['release_date'], errors='coerce')

# Remove duplicates
metadata_df.drop_duplicates(subset='id', keep='first', inplace=True)

# Drop rows with missing 'title' or 'id'
metadata_df.dropna(subset=['title', 'id'], inplace=True)

# Save the cleaned data
INTERIM_DATA_PATH = "../../data/interim"
cleaned_metadata_file = os.path.join(INTERIM_DATA_PATH, "cleaned_movies_metadata.csv")
metadata_df.to_csv(cleaned_metadata_file, index=False)

print("Data pre-processing completed.")


# Read raw data
ratings_file = os.path.join(RAW_DATA_PATH, "ratings_small.csv")
ratings_df = pd.read_csv(ratings_file)

# Convert 'timestamp' column to datetime
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

# Drop duplicates
ratings_df.drop_duplicates(subset=['userId', 'movieId'], keep='first', inplace=True)

# Remove any rows with missing values
ratings_df.dropna(inplace=True)

# Save the cleaned data
cleaned_ratings_file = os.path.join(INTERIM_DATA_PATH, "cleaned_ratings.csv")
ratings_df.to_csv(cleaned_ratings_file, index=False)

print("Ratings data pre-processing completed.")
