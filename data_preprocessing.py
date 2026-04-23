# Data cleaning and feature engineering
# This file will handle preprocessing pipeline
# src/data_preprocessing.py
"""
Module for data cleaning, preprocessing, and feature engineering.
This file prepares the raw data for model training.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_clean_data(file_path='../data/house_prices.csv'):
    """
    Load the raw CSV and perform basic cleaning.
    Returns cleaned DataFrame.
    """
    df = pd.read_csv(file_path)

    # Basic cleaning
    df = df.drop_duplicates()  # remove duplicate rows if any
    df = df.dropna(subset=['Price'])  # drop rows where target is missing

    # Optional: Drop Property_ID if it's just an identifier
    if 'Property_ID' in df.columns:
        df = df.drop('Property_ID', axis=1)

    print("Data loaded and cleaned. Shape:", df.shape)
    return df


def create_preprocessor():
    """
    Create the full preprocessing pipeline:
    - Scale numeric features
    - One-hot encode categorical features
    """
    numeric_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    categorical_features = ['Location', 'Property_Type']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # drop any other columns not listed
    )

    return preprocessor


def preprocess_data(df):
    """
    Apply full preprocessing to the DataFrame.
    Returns X (features), y (target), and the fitted preprocessor.
    """
    # Separate features and target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("Preprocessing complete. Processed shape:", X_processed.shape)
    return X_processed, y, preprocessor


def transform_new_data(new_df, fitted_preprocessor):
    """
    Transform new/unseen data using the already fitted preprocessor.
    Used for inference.
    """
    X_new = new_df  # assume new_df has same columns as training X
    X_new_processed = fitted_preprocessor.transform(X_new)
    return X_new_processed