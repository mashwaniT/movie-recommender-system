
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_preprocessed_data():
   def load_and_preprocess_images(df, image_dir, image_size=(224, 224)):
    images = []
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row['Image Index'])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, image_size)
        img_normalized = img_resized / 255.0
        images.append(img_normalized)

    return np.array(images)

   def load_and_preprocess_images_rgb(df, image_dir, image_size=(224, 224)):
    images = []
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row['Image Index'])
        img = cv2.imread(image_path)  # No conversion to grayscale
        img_resized = cv2.resize(img, image_size)
        img_normalized = img_resized / 255.0
        images.append(img_normalized)

    return np.array(images)

   csv_file_path = os.path.join("..", "..", "data", "raw", "sample_labels.csv")
   df = pd.read_csv(csv_file_path)

   class_counts = df['Finding Labels'].value_counts()
   classes_to_keep = class_counts[class_counts >= 2].index
   df = df[df['Finding Labels'].isin(classes_to_keep)]

   # creating train, validation and test sets
   train_df, test_df = train_test_split(
   df,
   test_size=0.2,
   random_state=42,
   stratify=df["Finding Labels"])

   train_df, val_df = train_test_split(
   train_df,
   test_size=0.25,
   random_state=42,
   stratify=train_df["Finding Labels"])

   # Load and preprocess images
   image_dir = os.path.join("..", "..", "data", "raw", "images")
   X_train = load_and_preprocess_images(train_df, image_dir)
   X_train = np.expand_dims(X_train, axis=-1) # add a channel dim to the end of the array
   X_val = load_and_preprocess_images(val_df, image_dir)
   X_val = np.expand_dims(X_val, axis=-1)
   X_test = load_and_preprocess_images(test_df, image_dir)
   X_test = np.expand_dims(X_test, axis=-1)

   X_train_rgb = load_and_preprocess_images_rgb(train_df, image_dir, image_size=(224, 224))
   X_val_rgb = load_and_preprocess_images_rgb(val_df, image_dir, image_size=(224, 224))
   X_test_rgb = load_and_preprocess_images_rgb(test_df, image_dir, image_size=(224, 224))

   # encoding labels using one-hot encoding
   mlb = MultiLabelBinarizer()
   y_train = mlb.fit_transform(train_df["Finding Labels"])
   y_val = mlb.transform(val_df["Finding Labels"])
   y_test = mlb.transform(test_df["Finding Labels"])

      # data augmentation with rotation, flipping and zooming
   train_datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=True,
      fill_mode="nearest")

   val_datagen = ImageDataGenerator()

   # multi-label stratified kfold
   n_splits = 5
   mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

   # Find the first fold to use as the train and test sets
   for train_index, test_index in mskf.split(df, mlb.transform(df["Finding Labels"])):
      train_df = df.iloc[train_index]
      test_df = df.iloc[test_index]
      break


   return train_df, val_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test, mlb, train_datagen, val_datagen, X_train_rgb, X_val_rgb, X_test_rgb



