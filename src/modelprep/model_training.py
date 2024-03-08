import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import logging
from sklearn.svm import SVC


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_and_save_models(models, X_train, y_train):
    """
    Trains the given models on the provided training data and saves the trained models to disk.

    Args:
        models (list): A list of tuples containing the name and classifier for each model.
        X_train (array-like): The training data.
        y_train (array-like): The target labels.

    Returns:
        list: A list of tuples containing the name and trained classifier for each model.
    """
    logging.info("Training models...")
    trained_models = []
    for name, classifier in models:
        classifier.fit(X_train, y_train)
        trained_models.append((name, classifier))
        joblib.dump(classifier, f'./models/{name}_model.pkl')
    return trained_models

def train_model(X_train, y_train):
    """
    Trains multiple machine learning models using the given training data.

    Parameters:
    - X_train (array-like): The input features for training.
    - y_train (array-like): The target labels for training.

    Returns:
    - trained_models (list): A list of trained models.

    """

    lgc = LogisticRegression(random_state=42)
    rfc = RandomForestClassifier(random_state=42)
    xgbc = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    svc = SVC(random_state=42)

    models = [
        ('Logistic Regression', lgc),
        ('Random Forest', rfc),
        ('Support Vector Classification', svc),
        ('XGBoost', xgbc)
    ]

    trained_models = train_and_save_models(models, X_train, y_train)
    logging.info("Models trained and saved successfully.")

    return trained_models
