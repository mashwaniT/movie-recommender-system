import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
from surprise import dump

def main():
    # Paths
    INTERIM_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'interim')
    cleaned_ratings_file = os.path.join(INTERIM_DATA_PATH, "cleaned_ratings.csv")

    # Load the cleaned ratings data
    ratings_df = pd.read_csv(cleaned_ratings_file)

    # Define the reader
    reader = Reader(rating_scale=(0, 5))

    # Create the dataset
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    # Split the data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train the SVD model
    algorithm = SVD()
    algorithm.fit(trainset)

    # Saving the trained model
    dump.dump('svd_model.dump', algo=algorithm)

    # Load trained model
    _, loaded_algorithm = dump.load('svd_model.dump')

    # Test the model
    predictions = loaded_algorithm.test(testset)

    # Calculate the RMSE
    rmse = accuracy.rmse(predictions)

    print("RMSE:", rmse)


if __name__ == "__main__":
    main()
