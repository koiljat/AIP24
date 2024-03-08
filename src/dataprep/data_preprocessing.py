from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dataprep.data_cleaning import clean_data
from dataprep.feature_engineer import add_new_feature
from dataprep.feature_extraction import select_features
from dataprep.feature_encoding import encode_features
from dataprep.feature_scaling import scale_data


def process_data_training(df):
    """
    Preprocesses the input dataframe using a pipeline of data preprocessing steps.

    Args:
        df (pandas.DataFrame): The input dataframe to be preprocessed.

    Returns:
        X_preprocessed (pandas.DataFrame): The preprocessed data as a pandas dataframe.
    """

    preprocess_pipeline = Pipeline(steps=[
        ('clean_data', clean_data),
        ('add_features', add_new_feature),
        ('select_features', select_features),
        ('encoding', encode_features),
        ])
    
    y = df["Adopted"]
    X = preprocess_pipeline.fit_transform(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = scale_data.fit_transform(X_train)
    X_test = scale_data.fit_transform(X_test)
    
    return X_train, X_test, y_train, y_test