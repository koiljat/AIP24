from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandleMissingName(BaseEstimator, TransformerMixin):
    """
    A transformer class to handle missing values in the 'Name' column.

    This class fills missing values in the 'Name' column with 'No Name' and updates the 'NameorNO' column accordingly.

    Parameters:
    ----------
    None

    Methods:
    -------
    fit(X, y=None):
        Fit the transformer on the input data.

    transform(X, y=None):
        Transform the input data by filling missing values in the 'Name' column and updating the 'NameorNO' column.

    Returns:
    -------
    X : pandas.DataFrame
        Transformed input data with missing values filled in the 'Name' column and 'NameorNO' column updated.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logging.info("Handling missing names in the dataset...")
        X['Name'] = X['Name'].fillna('No Name')
        X.loc[X['NameorNO'] != 'No Name', 'NameorNO'] = "N"
        X['NameorNO'] = X['NameorNO'].map({'Y': 1, 'N': 0})

        return X
    
class HandleMissngDesc(BaseEstimator, TransformerMixin):
    """
    A transformer class to handle missing values in the 'Description' column of a DataFrame.
    
    This class fills the missing values in the 'Description' column with the string 'No Description'.
    
    Parameters:
    ----------
    None
    
    Methods:
    -------
    fit(X, y=None):
        Fit the transformer on the input data.
        
        Parameters:
        ----------
        X : pandas DataFrame
            The input data.
        y : None
            The target variable (not used in this transformer).
        
        Returns:
        -------
        self : HandleMissngDesc
            The fitted transformer object.
    
    transform(X, y=None):
        Transform the input data by filling missing values in the 'Description' column.
        
        Parameters:
        ----------
        X : pandas DataFrame
            The input data.
        y : None
            The target variable (not used in this transformer).
        
        Returns:
        -------
        X : pandas DataFrame
            The transformed data with missing values in the 'Description' column filled.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        logging.info("Handling missing descriptions in the dataset...")
        X['Description'] = X['Description'].fillna('No Description')
        return X
    
class HandleBreedMislabel(BaseEstimator, TransformerMixin):
    """
    Transformer class to handle breed mislabeling in the dataset.
    
    This class implements the fit and transform methods required by scikit-learn's TransformerMixin.
    It replaces mislabeled breed values in the dataset with correct values based on certain conditions.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Parameters:
        - X: Input data.
        - y: Target labels (default=None).
        
        Returns:
        - self: Returns the instance itself.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data by replacing mislabeled breed values.
        
        Parameters:
        - X: Input data.
        - y: Target labels (default=None).
        
        Returns:
        - X: Transformed data with corrected breed values.
        """
        logging.info("Handling breed mislabeling in the dataset...")
        X.loc[(X['Breed1'] == 307) & (X['Type'] == 1), 'Breed1'] = 307
        X.loc[(X['Breed2'] == 307) & (X['Type'] == 1), 'Breed2'] = 307
        X.loc[(X['Breed2'] == 307) & (X['Type'] == 1), 'BreedName'] = "Mixed Breed Dog"
        X.loc[(X['Breed1'] == 307) & (X['Type'] == 2), 'Breed1'] = 308
        X.loc[(X['Breed2'] == 307) & (X['Type'] == 2), 'Breed2'] = 308
        X.loc[(X['Breed2'] == 307) & (X['Type'] == 2), 'BreedName'] = "Mixed Breed Cat"
        X.loc[X['Breed2'] == X['Breed1'], ['Breed2', 'BreedPure']] = 0, "Y"
        X.loc[X['Breed1'] == 0, 'BreedPure'] =  "Y"
        X.loc[X['Breed1'] == 0, 'Breed1'] = X.loc[X['Breed1'] == 0, 'Breed2']
        X['BreedPure'] = X['BreedPure'].map({'Y': 1, 'N': 0})
        return X
    
class HandleFurNameMislabel(BaseEstimator, TransformerMixin):
    """
    Transformer class to handle mislabeled values in the 'FurLengthName' column.
    Replaces 'Small' with 'Short' in the 'FurLengthName' column of the input DataFrame.
    """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X.loc[X['FurLengthName'] == 'Small', 'FurLengthName'] = 'Short'
        return X
    
class HandleTypeMislabel(BaseEstimator, TransformerMixin):
    """
    A transformer class to handle mislabeled 'Type' values in the dataset.
    
    This class implements the BaseEstimator and TransformerMixin interfaces, allowing it to be used in scikit-learn pipelines.
    It identifies mislabeled 'Type' values based on the 'Breed1' column and corrects them.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data by correcting mislabeled 'Type' values.
        
        Parameters:
        - X: The input data.
        - y: The target labels (optional).
        
        Returns:
        - X: The transformed data with corrected 'Type' values.
        """
        logging.info("Handling mislabeled 'Type' values in the dataset...")
        X.loc[X['Breed1'] < 241, 'Type'] = 1
        return X
    
    
class HandleOutliers(BaseEstimator, TransformerMixin):
    """
    A class to handle outliers in the dataset.

    This class implements the BaseEstimator and TransformerMixin interfaces
    to be compatible with scikit-learn pipelines.

    Methods:
    - fit(X, y=None): Fit the transformer on the training data.
    - transform(X, y=None): Transform the input data by clipping outliers.

    Attributes:
    None
    """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data by clipping outliers.

        Parameters:
        - X: The input features.
        - y: The target variable (default=None).

        Returns:
        - X: The transformed data with outliers clipped.
        """
        logging.info("Handling outliers in the dataset...")
        cap_age = X['Age'].quantile(0.95)
        cap_fee = X['Fee'].quantile(0.95)

        X['Age'] = X['Age'].clip(upper=cap_age)
        X['Fee'] = X['Fee'].clip(upper=cap_fee)
        return X
    
handle_missing_name = HandleMissingName()
handle_missing_desc = HandleMissngDesc()
handle_breed_mislabel = HandleBreedMislabel()
handle_fur_name_mislabel = HandleFurNameMislabel()
handle_type_mislabel = HandleTypeMislabel()
handle_outliers = HandleOutliers()

clean_data = Pipeline(steps=[
    ('handle_missing_name', handle_missing_name),
    ('handle_missing_desc', handle_missing_desc),
    ('handle_breed_mislabel', handle_breed_mislabel),
    ('handle_fur_name_mislabel', handle_fur_name_mislabel),
    ('handle_type_mislabel', handle_type_mislabel),
    ('handle_outliers', handle_outliers)
    ])

