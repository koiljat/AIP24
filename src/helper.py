from dataprep.data_preprocessing import *
from config import *
import pandas as pd 

df = pd.read_csv(TRAINING_DATA)

X_train, X_test, y_train, y_test = process_data_training(df)

columns = X_train.columns

def process_data(df):
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
        ('scaling', scale_data)
        ])
    
    if "Adopted" in df.columns:
        df = df.drop(columns=["Adopted"])
    
    X_preprocessed = preprocess_pipeline.fit_transform(df)
    
    missing_cols = set(columns) - set(X_preprocessed.columns)

    for col in missing_cols:
        X_preprocessed[col] = 0
        
    X_preprocessed = X_preprocessed[X_train.columns]

    return X_preprocessed