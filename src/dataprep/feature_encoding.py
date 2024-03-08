import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from config import CATEGORICAL_TRAINING_ENCODE

class EncodeFeatures(BaseEstimator, TransformerMixin):
    """
    EncodeFeatures is a transformer class that encodes categorical features using one-hot encoding.

    Parameters:
    -----------
    features : list
        List of column names representing the categorical features to be encoded.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer on the input data.

    transform(X)
        Transform the input data by encoding the categorical features.

    """

    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        """
        Fit the transformer on the input data.

        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        return self

    def transform(self, X):
        """
        Transform the input data by encoding the categorical features.

        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        encoded_df : DataFrame
            The transformed DataFrame with encoded categorical features.
        """
        encoded_df = pd.get_dummies(X, columns=self.features, drop_first=True)
        return encoded_df

encoder = EncodeFeatures(CATEGORICAL_TRAINING_ENCODE)

encode_features = Pipeline(steps=[
    ('encoding', encoder)
]
)