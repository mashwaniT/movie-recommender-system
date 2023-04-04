import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def build_model(num_users, num_movies):
    n_latent_factors = 40
    regularizer = 0.0001

    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, n_latent_factors, embeddings_regularizer=l2(regularizer))(user_input)
    user_flatten = Flatten()(user_embedding)

    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(num_movies, n_latent_factors, embeddings_regularizer=l2(regularizer))(movie_input)
    movie_flatten = Flatten()(movie_embedding)

    dot_product = Dot(axes=1)([user_flatten, movie_flatten])

    model = Model(inputs=[user_input, movie_input], outputs=dot_product)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=["mae"])

    return model

def load_and_preprocess_data():
    ratings_small = pd.read_csv("..\\..\\data\\interim\\cleaned_ratings.csv")
    user_enc = LabelEncoder()
    ratings_small['user_id'] = user_enc.fit_transform(ratings_small['userId'].values)
    num_users = ratings_small['user_id'].nunique()

    movie_enc = LabelEncoder()
    ratings_small['movie_id'] = movie_enc.fit_transform(ratings_small['movieId'].values)
    num_movies = ratings_small['movie_id'].nunique()

    return ratings_small, num_users, num_movies

def train_and_evaluate_model(model, user_id_train, movie_id_train, rating_train, user_id_test, movie_id_test, rating_test):
    model.fit([user_id_train, movie_id_train], rating_train, batch_size=64, epochs=10, validation_split=0.1)
    test_mae = model.evaluate([user_id_test, movie_id_test], rating_test, batch_size=64)[1]
    print(f"Test MAE: {test_mae:.4f}")

def evaluate_model_performance(model, user_id_test, movie_id_test, rating_test):
    predicted_ratings = model.predict([user_id_test, movie_id_test]).flatten()
    mae = mean_absolute_error(rating_test, predicted_ratings)
    print(f"Mean Absolute Error: {mae:.4f}")

    mse = mean_squared_error(rating_test, predicted_ratings)
    print(f"Mean Squared Error: {mse:.4f}")

    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse:.4f}")

def main():
    ratings_small, num_users, num_movies = load_and_preprocess_data()
    user_id_train, user_id_test, movie_id_train, movie_id_test, rating_train, rating_test = train_test_split(
        ratings_small['user_id'], ratings_small['movie_id'], ratings_small['rating'], test_size=0.2, random_state=42
    )

    best_model = build_model(num_users, num_movies)
    train_and_evaluate_model(best_model, user_id_train, movie_id_train, rating_train, user_id_test, movie_id_test, rating_test)

    # Evaluate the performance on the test set
    evaluate_model_performance(best_model, user_id_test, movie_id_test, rating_test)

    # Save the model
    best_model.save("best_matrix_factorization_model.h5")

if __name__ == "__main__":
    main()
