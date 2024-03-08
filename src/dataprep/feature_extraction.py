from sklearn.base import BaseEstimator, TransformerMixin
import logging
from config import TRAINING_VAR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExtractColumns(BaseEstimator, TransformerMixin):
    """
    A transformer class to select specified columns from a DataFrame.

    Parameters:
    -----------
    training_data : list
        A list of column names to be extracted from the DataFrame.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data.

    transform(X):
        Transform the data by extracting the specified columns.

    Returns:
    --------
    DataFrame
        The transformed DataFrame with the specified columns extracted.
    """

    def __init__(self, training_data):
        self.training_data = training_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logging.info("Extracting relevant columns from the dataset...")
        return X[self.training_data]

select_features = ExtractColumns(TRAINING_VAR)