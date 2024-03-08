import pandas as pd
import logging
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_results(name, classifier, X_test, y_test, X_train, y_train):
    """
    Print evaluation results for a given classifier.

    Parameters:
    - name (str): The name of the classifier.
    - classifier: The trained classifier object.
    - X_test: The test data.
    - y_test: The true labels for the test data.
    """

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    score = classifier.score(X_test, y_test)
    
    # Cross validate the model
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    mean_cv_score = cv_scores.mean()

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)

    # Confusion Matrix
    confusion_matrix_str = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n{'='*40}\n{name}\n{'-'*40}")
    print(f"Test score: {score}")
    print(f"Mean cross-validation score: {mean_cv_score}")
    print(f"\nClassification Report:\n{report_str}")

    # Convert classification report to a table for better visualization
    report_table = [[label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']]
                    for label, metrics in report.items() if label.isdigit()]
    report_headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    print(tabulate(report_table, headers=report_headers, tablefmt="fancy_grid"))
    print(f"\nConfusion Matrix:\n{tabulate(confusion_matrix_str, tablefmt='fancy_grid')}\n")

def evaluate_model(models, X_test, y_test, X_train, y_train):
    """
    Evaluates the performance of multiple models.

    Args:
        models (list): A list of tuples containing the name and classifier for each model.

    Returns:
        None
    """
    for name, classifier in models:    
        logging.info("Getting results for %s. This may take a while. Please be patient.", name)
        print_results(name, classifier, X_test, y_test, X_train, y_train)