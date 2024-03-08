import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sklearn.preprocessing import OrdinalEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def categorize_sentiment(text):
    """
    Categorizes the sentiment of the given text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    str: The sentiment category ('Positive', 'Negative', or 'Neutral').

    Raises:
    None
    """
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Neutral'

def classify_language(text):
    """
    Classify the language of the given text.

    Parameters:
    text (str): The text to be classified.

    Returns:
    str: The detected language of the text, or 'unknown' if the language cannot be detected.
    """
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'
    
class AddDescSentiment(BaseEstimator, TransformerMixin):
    """
    Transformer class to add sentiment features to the dataset based on the 'Description' column.

    This class implements the fit and transform methods required by scikit-learn's TransformerMixin
    and BaseEstimator interfaces.

    Parameters:
    ----------
    None

    Methods:
    -------
    fit(X, y=None):
        Fit the transformer on the input data.

    transform(X, y=None):
        Transform the input data by adding sentiment features.

    Returns:
    -------
    X : pandas.DataFrame
        Transformed dataset with added sentiment features.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit the transformer on the input data.

        Parameters:
        ----------
        X : pandas.DataFrame
            Input data.

        y : array-like, default=None
            Target values.

        Returns:
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X, y=None):
        """
        Transform the input data by adding sentiment features.

        Parameters:
        ----------
        X : pandas.DataFrame
            Input data.

        y : array-like, default=None
            Target values.

        Returns:
        -------
        X : pandas.DataFrame
            Transformed dataset with added sentiment features.
        """
        logging.info("Adding sentiment feature to the dataset...")
        X['DescriptionSentiment'] = X['Description'].apply(categorize_sentiment)
        categories = [['Negative', 'Neutral', 'Positive']]
        ordinal_encoder = OrdinalEncoder(categories=categories)
        X['DescriptionSentimentEncoded'] = ordinal_encoder.fit_transform(X[['DescriptionSentiment']])
        X = X.drop(columns=['DescriptionSentiment'])
        return X
    
class AddDescLanguage(BaseEstimator, TransformerMixin):
    """
    Transformer class to add language feature to the dataset based on the 'Description' column.

    This class implements the fit and transform methods required by scikit-learn's TransformerMixin.
    The fit method does not perform any training and simply returns self.
    The transform method adds a new column 'DescriptionLanguage' to the input DataFrame X,
    which contains the language classification of each description.
    The language classification is obtained using the classify_language function.
    The transform method also performs one-hot encoding on the 'DescriptionLanguage' column.

    Parameters:
    ----------
    None

    Returns:
    -------
    X : pandas DataFrame
        Transformed DataFrame with the 'DescriptionLanguage' column added and one-hot encoded.

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit method required by scikit-learn's BaseEstimator.

        Parameters:
        ----------
        X : pandas DataFrame
            Input DataFrame.

        y : array-like, default=None
            Target values (ignored).

        Returns:
        -------
        self : object
            Returns self.

        """
        return self

    def transform(self, X, y=None):
        """
        Transform method required by scikit-learn's TransformerMixin.

        Parameters:
        ----------
        X : pandas DataFrame
            Input DataFrame.

        y : array-like, default=None
            Target values (ignored).

        Returns:
        -------
        X : pandas DataFrame
            Transformed DataFrame with the 'DescriptionLanguage' column added and one-hot encoded.

        """
        logging.info("Adding language feature to the dataset...")
        X['DescriptionLanguage'] = X['Description'].fillna('').apply(classify_language)
        X = pd.get_dummies(X, columns=['DescriptionLanguage'], drop_first=True)
        return X

add_desc_sentiment = AddDescSentiment()
add_desc_language = AddDescLanguage()

add_new_feature = Pipeline(steps=[
    ('sentiment', add_desc_sentiment),
    ('language', add_desc_language)
])