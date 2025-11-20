"""
Data preprocessing functions for ML pipeline.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath):
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(filepath)


def clean_data(df):
    """
    Perform basic data cleaning.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(method='ffill')
    
    return df


def encode_categorical_features(df, categorical_columns):
    """
    Encode categorical features using label encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical column names
        
    Returns:
        pd.DataFrame: Dataframe with encoded features
    """
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    
    return df_encoded, label_encoders


def scale_features(df, feature_columns):
    """
    Scale numerical features using standard scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): List of numerical column names
        
    Returns:
        pd.DataFrame: Dataframe with scaled features
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df_scaled, scaler