import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data.pre_processing import get_preprocessed_data

def build_custom_cnn(input_shape, num_classes):
   model = models.Sequential()
   model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(num_classes, activation='sigmoid'))
   return model


def build_transfer_learning_model(base_model, num_classes):
   model = models.Sequential()
   model.add(base_model)
   model.add(layers.Flatten())
   model.add(layers.Dense(256, activation='relu'))
   model.add(layers.Dense(num_classes, activation='sigmoid'))
   return model


# define input shape and num of classes load and preprocess images and labels
input_shape = (224,224,1)
input_shape_resnet = (224,224, 3)
num_classes = len(mlb.classes_)

# create models
# Custom CNN
custom_cnn = build_custom_cnn(input_shape, num_classes)
# VGG16
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model_vgg16.trainable = False
vgg16_model = build_transfer_learning_model(base_model_vgg16, num_classes)

# ResNet
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape_resnet)
base_model_resnet.trainable = False
resnet_model = build_transfer_learning_model(base_model_resnet, num_classes)

# compile and train models
# Custom CNN
custom_cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
custom_cnn.fit(train_datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=10)

# VGG16
vgg16_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
vgg16_model.fit(train_datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=10)

# ResNet
resnet_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
resnet_model.fit(train_datagen.flow(X_train_resnet, y_train, batch_size=32), validation_data=(X_val_resnet, y_val), epochs=10)

# evaluate and print metrics
def evaluate_and_print_metrics(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

# Evaluate models
evaluate_and_print_metrics(custom_cnn, X_test, y_test, "Custom CNN")
evaluate_and_print_metrics(vgg16_model, X_test, y_test, "VGG16")
evaluate_and_print_metrics(resnet_model, X_test, y_test, "ResNet")
