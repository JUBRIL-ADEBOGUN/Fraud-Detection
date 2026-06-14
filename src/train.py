import json
import mlflow  # type: ignore #pylint [ignore]
import mlflow.sklearn  # type: ignore
from mlflow.models import infer_signature  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.pipeline import Pipeline  # pyright: ignore
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)  # pyright: ignore
from features import XentePreprocessor, load_and_split_xente
from sklearn.metrics import ConfusionMatrixDisplay  # pyright: ignore
import matplotlib.pyplot as plt  # pyright: ignore

DATA_PATH = "./data/raw/train.csv"
EXPERIMENT_NAME = "Xente_Fraud_Detection"
PARAMS_PATH = "./notebooks/best_params.json"


def load_params(params_path):
    """Load parameters from JSON file."""
    with open(params_path, "r") as f:
        return json.load(f)


def train_and_log():
    # Change logging directory
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading Data...")
    X_train, X_test, y_train, y_test = load_and_split_xente(DATA_PATH)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Load parameters from notebook
    print(f"Loading parameters from {PARAMS_PATH}...")
    params = load_params(PARAMS_PATH)
    # Override scale_pos_weight with current data if needed
    params["scale_pos_weight"] = scale_pos_weight
    # start MLflow run
    with mlflow.start_run(run_name="XGB_Xente_Candidate") as run:
        # Create pipeline with preprocessor and classifier
        xgb_model = xgb.XGBClassifier(**params)
        pipeline = Pipeline(
            [
                ("preprocessor", XentePreprocessor()),
                ("classifier", xgb_model),
            ]
        )

        print("Training pipeline...")
        # Transform X_test using the pipeline preprocessor (fit on X_train)
        preprocessor = pipeline.named_steps["preprocessor"].fit(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        # Fit pipeline with early stopping enabled
        pipeline.fit(
            X_train,
            y_train,
            classifier__eval_set=[(X_test_transformed, y_test)],
            # classifier__eval_metric=eval_metric,
            # classifier__early_stopping_rounds=early_stopping_rounds,
            classifier__verbose=False,
        )

        print("Evaluating...")
        preds = pipeline.predict(X_test)
        preds_proba = pipeline.predict_proba(X_test)[:, 1]

        auprc = average_precision_score(y_test, preds_proba)

        # plot confusion matrix and log it as an artifact
        ConfusionMatrixDisplay.from_predictions(y_test, preds, normalize="true")
        plt.title("Normalized Confusion Matrix")
        plt.savefig("./reports/confusion_matrix.png")

        mlflow.log_artifact("./reports/confusion_matrix.png")
        # mlflow.log_artifact("./reports/shap_values.png")

        print(f"AUPRC: {auprc:.4f}")
        # Log model parameters including fit parameters
        log_params = params.copy()

        mlflow.log_params(log_params)
        mlflow.log_metric("auprc", auprc)  # pyright: ignore

        # 1. Create a Model Signature (defines structure and data types)
        predictions = pipeline.predict(X_test)
        signature = infer_signature(X_test, predictions)

        # 2. Extract a real input example slice (e.g., first 3 rows)
        # to bundle as an artifact
        input_example = X_test.iloc[0:3]

        # Plot Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, preds_proba)
        plt.figure(figsize=(8, 5))
        plt.plot(
            recall_vals,
            precision_vals,
            marker=".",
            label=f"XGBoost (AUPRC = {auprc:.3f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig("./reports/precision_recall_curve.png")
        mlflow.log_artifact("./reports/precision_recall_curve.png")

        mlflow.sklearn.log_model(  # pyright: ignore
            pipeline,
            name="fraud_pipeline",  # pyright: ignore
            signature=signature,
            input_example=input_example,
        )
        print(f"Model saved! Run ID: {run.info.run_id}")
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/fraud_pipeline"
        registered_model = mlflow.register_model(model_uri, "fraud-detection-model")
        print(f"Model registered: {registered_model.name} (Version: {registered_model.version})")
        # Transition to Production stage
        client = mlflow.tracking.MlflowClient()  # pyright: ignore
        client.transition_model_version_stage(
            name="fraud-detection-model",
            version=registered_model.version,
            stage="Production"
        )
        print("Model transitioned to Production stage!")


if __name__ == "__main__":
    train_and_log()
