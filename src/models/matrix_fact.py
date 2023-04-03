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

# Load and preprocess the data
ratings_small = pd.read_csv("..\\..\\data\\interim\\cleaned_ratings.csv")
user_enc = LabelEncoder()
ratings_small['user_id'] = user_enc.fit_transform(ratings_small['userId'].values)
num_users = ratings_small['user_id'].nunique()

movie_enc = LabelEncoder()
ratings_small['movie_id'] = movie_enc.fit_transform(ratings_small['movieId'].values)
num_movies = ratings_small['movie_id'].nunique()

# Split the data into train and test sets
user_id_train, user_id_test, movie_id_train, movie_id_test, rating_train, rating_test = train_test_split(
    ratings_small['user_id'], ratings_small['movie_id'], ratings_small['rating'], test_size=0.2, random_state=42
)

# Build the model with the best hyperparameters and train it
best_model = build_model(num_users, num_movies)
best_model.fit([user_id_train, movie_id_train], rating_train, batch_size=64, epochs=10, validation_split=0.1)

# Evaluate the performance on the test set
test_mae = best_model.evaluate([user_id_test, movie_id_test], rating_test, batch_size=64)[1]
print(f"Test MAE: {test_mae:.4f}")

# saving the trained model
best_model.save = ("best_matrix_factorization_model.h5")
