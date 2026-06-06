from fastapi import FastAPI, HTTPException # pyright: ignore
from pydantic import BaseModel # pyright: ignore
import pandas as pd # pyright: ignore
import mlflow.sklearn # pyright: ignore
import numpy as np # pyright: ignore
import sys
sys.path.insert(0, './src')
from features import XentePreprocessor  # noqa: F401 - needed for model unpickling

# Set MLflow tracking URI at module level
# mlflow.set_tracking_uri(Path.cwd() / "mlruns")

class XenteTransaction(BaseModel):
    TransactionId: str
    CustomerId: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    TransactionStartTime: str
    PricingStrategy: int
    IsCredit: int
    
    class Config:
        json_schema_extra = {
            "example": {
                'TransactionId': 'TransactionId_51888',
                'CustomerId': 'CustomerId_1',
                'ProviderId': 'ProviderId_4',
                'ProductId': 'ProductId_10',
                'ProductCategory': 'airtime',
                'ChannelId': 'ChannelId_2',
                'Amount': -10000.0,
                'TransactionStartTime': '2018-11-21T16:49:14Z',
                'PricingStrategy': 4,
                'IsCredit': 1
            }
        }

app = FastAPI(
    title="Xente Real-Time Fraud API",
    version="1.0"
)

# Sample transaction for testing
SAMPLE_TRANSACTION = {
    'TransactionId': 'TransactionId_51888',
    'CustomerId': 'CustomerId_1',
    'ProviderId': 'ProviderId_4',
    'ProductId': 'ProductId_10',
    'ProductCategory': 'airtime',
    'ChannelId': 'ChannelId_2',
    'Amount': -10000.0,
    'TransactionStartTime': '2018-11-21T16:49:14Z',
    'PricingStrategy': 4,
    'IsCredit': 1
}

MODEL_URI = "runs:/b1162ca437d6405ba89f4b463ba536e2/fraud_pipeline"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model_pipeline = None

@app.on_event("startup")
def load_model():
    global model_pipeline
    model_pipeline = mlflow.sklearn.load_model(MODEL_URI)  # pyright: ignore
    if model_pipeline:
        print("Model loaded successfully!")

@app.get("/test")
def test_prediction():
    """Test endpoint using sample transaction"""
    if not model_pipeline:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded."
        )
    
    transaction = XenteTransaction(**SAMPLE_TRANSACTION)
    df_input = pd.DataFrame([transaction.dict()])
    probability = model_pipeline.predict_proba(df_input)[0][1]
    
    return {
        "message": "Test prediction using sample transaction",
        "user_input": {"sample_transaction": SAMPLE_TRANSACTION},
        "model_output": {
            "transaction_id": transaction.TransactionId,
            "product_category": transaction.ProductCategory,
            "fraudResult": bool(probability >= 0.5),
            "fraud_probability": float(np.round(probability, 3))
        }
    }

@app.post("/predict")
def predict_fraud(transaction: XenteTransaction):
    if not model_pipeline:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded."
        )
        
    df_input = pd.DataFrame([transaction.dict()])
    
    # Notice: Our custom pipeline automatically fetches the
    # customer's historical avg amount and applies it to this
    # new payload seamlessly!
    probability = model_pipeline.predict_proba(df_input)[0][1]
    
    return {
        "transaction_id": transaction.TransactionId,
        "Product_Category": transaction.ProductCategory,
        "fraudResult": bool(probability >= 0.5),
        # probability in 3 decimal places
        "fraud_probability": float(np.round(probability, 3))
    }
