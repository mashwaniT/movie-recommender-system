import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


ratings = pd.read_csv('..\\..\\data\\interim\\cleaned_ratings.csv')
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
print(train.columns)

user_enc = LabelEncoder()
ratings['user_id'] = user_enc.fit_transform(ratings['userId'].values)
n_users = ratings['user_id'].nunique()

item_enc = LabelEncoder()
ratings['item_id'] = item_enc.fit_transform(ratings['movieId'].values)
n_items = ratings['item_id'].nunique()

train['user_id'] = user_enc.transform(train['userId'].values)
train['item_id'] = item_enc.transform(train['movieId'].values)
test['user_id'] = user_enc.transform(test['userId'].values)
test['item_id'] = item_enc.transform(test['movieId'].values)

def create_model(n_users, n_items, embedding_size=50, hidden_layers=[10], dropout_rate=0.2, l2_reg=1e-5):
    user_input = Input(shape=(1,))
    user_embedding = Embedding(n_users, embedding_size, embeddings_regularizer=l2(l2_reg))(user_input)
    user_flatten = Flatten()(user_embedding)

    item_input = Input(shape=(1,))
    item_embedding = Embedding(n_items, embedding_size, embeddings_regularizer=l2(l2_reg))(item_input)
    item_flatten = Flatten()(item_embedding)

    dot_product = Dot(axes=1)([user_flatten, item_flatten])

    x = dot_product
    for layer_size in hidden_layers:
        x = Dense(layer_size, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    output = Dense(1)(x)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

    return model

model = create_model(n_users, n_items)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    [train['user_id'].values, train['item_id'].values],
    train['rating'].values,
    batch_size=64,
    epochs=50,
    verbose=1,
    validation_split=0.1,
    callbacks=[early_stopping]
)

test_loss, test_mae = model.evaluate(
    [test['user_id'].values, test['item_id'].values],
    test['rating'].values,
    verbose=1
)

print(f"Test MAE: {test_mae:.4f}")
