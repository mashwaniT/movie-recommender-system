import sys
sys.path.append("C:\\Users\\MasterLaptop\\Documents\\GitHub\\final-project-190728860\\src")
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data.pre_processing import get_preprocessed_data
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is running on GPU")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("TensorFlow is running on CPU")

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

train_df, val_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test, mlb, train_datagen, val_datagen, X_train_rgb, X_val_rgb, X_test_rgb = get_preprocessed_data()

# assigning class weights to underrepresented classes
sample_weights = compute_sample_weight("balanced", y_train) # computing class weights



# define input shape and num of classes load and preprocess images and labels
input_shape = (224,224,1)
input_shape_resnet = (224,224, 3)
num_classes = len(mlb.classes_)

# create models
# Custom CNN
custom_cnn = build_custom_cnn(input_shape, num_classes)
# VGG16
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_resnet)
base_model_vgg16.trainable = False
vgg16_model = build_transfer_learning_model(base_model_vgg16, num_classes)

# ResNet
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape_resnet)
base_model_resnet.trainable = False
resnet_model = build_transfer_learning_model(base_model_resnet, num_classes)

# compile and train models
epochs = 10
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
# Custom CNN
custom_cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history_custom_cnn = custom_cnn.fit(
   train_datagen.flow(X_train, y_train, batch_size=32),
   steps_per_epoch=len(X_train) // 32,
   epochs=epochs,
   validation_data=val_datagen.flow(X_val, y_val, batch_size=32),
   validation_steps=len(X_val) // 32,
   callbacks=[early_stopping_callback]
)

# VGG16
vgg16_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
vgg16_model.fit(
   train_datagen.flow(X_train_rgb, y_train, batch_size=32, sample_weight=sample_weights),
   validation_data=(X_val_rgb, y_val),
   epochs=epochs)

# ResNet
resnet_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
resnet_model.fit(
   train_datagen.flow(X_train_rgb, y_train, batch_size=32, sample_weight=sample_weights),
   validation_data=(X_val_rgb, y_val),
   epochs=epochs)

# evaluate and print metrics
def evaluate_and_print_metrics(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, np.round(y_pred))
    precision = precision_score(y_test, np.round(y_pred), average='weighted')
    recall = recall_score(y_test, np.round(y_pred), average='weighted')
    f1 = f1_score(y_test, np.round(y_pred), average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred, average="weighted")

    print(f"{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}\n")

# Evaluate models
evaluate_and_print_metrics(custom_cnn, X_test, y_test, "Custom CNN")
evaluate_and_print_metrics(vgg16_model, X_test_rgb, y_test, "VGG16")
evaluate_and_print_metrics(resnet_model, X_test_rgb, y_test, "ResNet")
