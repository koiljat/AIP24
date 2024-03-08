import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tuning(X, y):
    """
    Perform hyperparameter tuning for different models using RandomizedSearchCV.

    Parameters:
    X (array-like): The input features.
    y (array-like): The target variable.

    Returns:
    tuple: A tuple containing the best parameters found for each model.
    """
    logging.info("Tuning parameters. This will take awhile. Please wait")
    lr_param_grid = {'C': [0.1, 1, 10]}
    rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
    xgb_param_grid = {'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [3, 5, 7]}
    svc_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

    # Create the models
    lr_model = LogisticRegression()
    rf_model = RandomForestClassifier()
    xgb_model = XGBClassifier()
    svc_model = SVC()

    lr_random_search = RandomizedSearchCV(estimator=lr_model, param_distributions=lr_param_grid, n_iter=3, cv=5, random_state=42)
    rf_random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_param_grid, n_iter=9, cv=5, random_state=42)
    xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_grid, n_iter=9, cv=5, random_state=42)
    svc_random_search = RandomizedSearchCV(estimator=svc_model, param_distributions=svc_param_grid, n_iter=6, cv=5, random_state=42)

    # Fit the models to the data
    lr_random_search.fit(X, y)
    rf_random_search.fit(X, y)
    xgb_random_search.fit(X, y)
    svc_random_search.fit(X, y)

    lr_random_search.best_params_
    rf_random_search.best_params_
    xgb_random_search.best_params_
    svc_random_search.best_params_
    
    return (lr_random_search.best_params_, rf_random_search.best_params_, xgb_random_search.best_params_, svc_random_search.best_params_)