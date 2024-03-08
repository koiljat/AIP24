from sklearn.ensemble import StackingClassifier, VotingClassifier
from config import *
from modelprep.model_evaluation import *
from modelprep.model_training import *
from dataprep.data_preprocessing import *
from modelprep.hyperparameters_tuning import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def main():
    df = pd.read_csv(TRAINING_DATA)
    X = df.drop(columns=["Adopted"])
    y = df["Adopted"]
    
    X_train, X_test, y_train, y_test = process_data_training(df)
    
    trained_models  = train_model(X_train, y_train)
    evaluate_model(trained_models, X_test, y_test, X_train, y_train)
    models_params = tuning(X_train, y_train)
    
    new_models = [("new_lgc", LogisticRegression(**models_params[0], random_state=42)), 
                  ("new_rfc", RandomForestClassifier(**models_params[1], random_state=42)),
                  ("new_xgb", XGBClassifier(**models_params[2], use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
                  ("new_svc", SVC(**models_params[3], random_state=42))]
    
    new_trained_models = train_and_save_models(new_models, X_train, y_train)
    evaluate_model(new_trained_models, X_test, y_test, X_train, y_train)
    
    voting_classifier = VotingClassifier(estimators=new_models, voting='hard')
    meta_classifier = SVC(random_state=42)
    stacking_classifier = StackingClassifier(estimators=new_models, final_estimator=meta_classifier)
    
    ensembled_models = [("voting_classifier", voting_classifier), ("stacking_classifier", stacking_classifier)]
    ensembled_trained_models = train_and_save_models(ensembled_models, X_train, y_train)
    evaluate_model(ensembled_trained_models, X_test, y_test, X_train, y_train)
    
    logging.info("End of the pipeline.")

if __name__ == "__main__":
    main()
    
    