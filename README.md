[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/YCTbQ0qx)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10597506&assignment_repo_type=AssignmentRepo)
Project Instructions
==============================

This repo contains the instructions for a machine learning project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── preprocessing data           <- Scripts to download or generate data and pre-process the data
       │   └── make_dataset.py -> removed.
       │   └── pre-processing.py
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py -> removed
       │   └── train_model.py -> removed
           |__ collab_filtering.py
           |__ content_based.py
           |__ hybrid.py (not functional, tried to combine collab and content based filtering)
           |__ matrix_fact.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py


# Movie Recommendation System Using the Movies Dataset

## Abstract

For my CP322 final project, I decided to build a machine learning algorithm to come up with a list of movie recommendations to help people find something to watch on movie nights. More times it takes longer to look for a movie than watch one.

There is metadata on over 45,000 movies along with 26 million ratings from over 270,000 users. I will be using the smaller sample ratings.csv to train models because I only have access to limited resources.

## Methodology
The movie recommendation system implemented in this project is based on two primary models: collaborative filtering and content-based filtering. Collaborative filtering is a technique that analyzes user-item interactions to find similar users and make recommendations based on their interests. Content-based filtering, on the other hand, relies on the attributes of the items being recommended to find similar items.

The collaborative filtering model used in this project is based on matrix factorization, specifically Singular Value Decomposition (SVD). This model takes the user-item matrix, where each row represents a user and each column represents an item, and decomposes it into three matrices: the user matrix, the item matrix, and the singular value matrix. This allows the model to represent each user and each item in a lower-dimensional space, making it easier to find similar users and items.

The content-based filtering model used in this project is based on the cosine similarity between the item attributes. In this case, the attributes are the plot summary and genre of each movie. This model creates a vector representation of each movie based on its attributes and calculates the cosine similarity between these vectors to find similar movies.

The hybrid model combines these two models to improve the accuracy of the recommendations. The hybrid model takes into account both the user-item interactions and the attributes of the items being recommended. This allows the model to make more accurate recommendations by finding movies that are both similar to the user's previous preferences and have similar attributes to movies the user has enjoyed in the past.

The methodology behind this movie recommendation system involves several steps. First, I preprocess the data by cleaning and transforming it into a format that can be used by the models. This involves removing duplicates, handling missing data, and creating feature vectors for each movie based on its plot summary and genre.

Next, I train the collaborative filtering model using Singular Value Decomposition (SVD) on the user-item matrix. This model uses the interactions between users and movies to find similar users and make recommendations based on their interests.

I also train the content-based filtering model by calculating the cosine similarity between the feature vectors of each movie. This model finds movies with similar attributes, such as plot summary and genre, to make recommendations.

Finally, I combine these two models to create a hybrid model that takes into account both user-item interactions and item attributes. This hybrid model uses a weighted average of the predicted ratings from the collaborative filtering and content-based models to make final recommendations.

To evaluate the performance of the recommendation system, I use a holdout set of ratings to measure the accuracy of the model's predictions. I measure the performance using metrics such as Root Mean Squared Error (RMSE) and Mean Average Precision (MAP).

In conclusion, the methodology behind this movie recommendation system involves preprocessing the data, training collaborative and content-based filtering models, and combining these models into a hybrid model. This allows the model to make more accurate recommendations by taking into account both user-item interactions and item attributes. The performance of the recommendation system is evaluated using metrics such as RMSE and MAP to measure the accuracy of the model's predictions.