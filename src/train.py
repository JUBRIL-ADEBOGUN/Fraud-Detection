import mlflow # type: ignore #pylint [ignore]
import mlflow.sklearn # type: ignore
from mlflow.models import infer_signature # type: ignore
import xgboost as xgb # type: ignore
from sklearn.pipeline import Pipeline #pyright: ignore
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score #pyright: ignore
from features import XentePreprocessor, load_and_split_xente
from sklearn.metrics import ConfusionMatrixDisplay # pyright: ignore
import matplotlib.pyplot as plt # pyright: ignore

DATA_PATH = "./data/raw/train.csv"
EXPERIMENT_NAME = "Xente_Fraud_Detection"

def train_and_log():
    # Change logging directory
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("Loading Data...")
    X_train, X_test, y_train, y_test = load_and_split_xente(DATA_PATH)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 1200,
        'random_state': 42,
        'enable_categorical': True,
        # 'early_stopping_rounds': 20
    }
    

    with mlflow.start_run(run_name="XGB_Xente_Candidate") as run:
        # Pre-fit preprocessor to handle eval_set transformation
        preprocessor = XentePreprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Create pipeline and fit with transformed eval_set
        pipeline = Pipeline([
            ('preprocessor', XentePreprocessor()),  # Will be fit on X_train
            ('classifier', xgb.XGBClassifier(**params))
        ])
        
        print("Training pipeline...")
        pipeline.fit(X_train, y_train, 
                     classifier__eval_set=[(X_test_transformed, y_test)],
                     classifier__verbose=False)
        
        print("Evaluating...")
        preds = pipeline.predict(X_test)
        preds_proba = pipeline.predict_proba(X_test)[:, 1]
        
        auprc = average_precision_score(y_test, preds_proba)

        # plot confusion matrix and log it as an artifact
        disp = ConfusionMatrixDisplay.from_predictions(y_test, preds, normalize='true')
        plt.title("Normalized Confusion Matrix")
        plt.savefig("./reports/confusion_matrix.png")

        mlflow.log_artifact("./reports/confusion_matrix.png")
        # mlflow.log_artifact("./reports/shap_values.png")

        print(f"AUPRC: {auprc:.4f}")
        mlflow.log_params(params)
        mlflow.log_metric("auprc", auprc) # pyright: ignore

        # 1. Create a Model Signature (defines structure and data types)
        predictions = pipeline.predict(X_test)
        signature = infer_signature(X_test, predictions)

        # 2. Extract a real input example slice (e.g., first 3 rows) to bundle as an artifact
        input_example = X_test.iloc[0:3]

        # Plot Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, preds_proba)
        plt.figure(figsize=(8, 5))
        plt.plot(recall_vals, precision_vals, marker='.', label=f'XGBoost (AUPRC = {auprc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig("./reports/precision_recall_curve.png")
        mlflow.log_artifact("./reports/precision_recall_curve.png")

        
        mlflow.sklearn.log_model(pipeline, name="fraud_pipeline",  # pyright: ignore
                        signature=signature, input_example=input_example)
        print(f"Model saved! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_and_log()