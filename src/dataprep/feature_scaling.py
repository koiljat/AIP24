import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
numerical_features = ['Age', 'Fee', 'Quantity', 'VideoAmt', 'PhotoAmt']

class ScaleFeature(BaseEstimator, TransformerMixin):
    """
    A transformer class for scaling numerical features using MinMaxScaler.

    Parameters:
    -----------
    features : list
        List of column names of the numerical features to be scaled.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer on the training data.

    transform(X)
        Transform the input data by scaling the specified numerical features.

    Returns:
    --------
    X : pandas DataFrame
        Transformed data with scaled numerical features.
    """

    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Scaling numerical features...")
        scaler = MinMaxScaler()
        numerical_data = X[self.features]
        scaled_numerical_data = scaler.fit_transform(numerical_data)
        X[self.features] = scaled_numerical_data
        return X
    
scaler = ScaleFeature(numerical_features)

scale_data = Pipeline(steps=[
    ('scaling', scaler)
]
)