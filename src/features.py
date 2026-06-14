import pandas as pd  # pyright: ignore
import numpy as np  # pyright: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # pyright: ignore
from sklearn.model_selection import train_test_split  # pyright: ignore


class XentePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Dictionaries to act as our "Feature Store"
        self.customer_avg_amount = {}
        self.customer_tx_count = {}
        self.provider_tx_count = {}
        self.global_avg_amount = 0

    def fit(self, X, y=None):
        """
        Learn historical aggregations from the training data.
        """
        # Calculate historical stats to memorize
        self.customer_avg_amount = X.groupby("CustomerId")["Amount"].mean().to_dict()
        self.customer_tx_count = X.groupby("CustomerId").size().to_dict()
        self.provider_tx_count = X.groupby("ProviderId").size().to_dict()
        self.global_avg_amount = X["Amount"].mean()
        return self

    def transform(self, X):
        X_tf = X.copy()

        # 1. Parse Datetime
        if "TransactionStartTime" in X_tf.columns:
            X_tf["TransactionStartTime"] = pd.to_datetime(
                X_tf["TransactionStartTime"],
            )
            X_tf["Hour"] = X_tf["TransactionStartTime"].dt.hour
            X_tf["DayOfWeek"] = X_tf["TransactionStartTime"].dt.dayofweek
            X_tf["DayOfMonth"] = X_tf["TransactionStartTime"].dt.day

        # 2. Map Historical Features (Velocity/Frequency)
        # We use .map() with our dictionaries. If a customer is unseen, we fill with defaults.
        if "CustomerId" in X_tf.columns:
            X_tf["Customer_Avg_Amount"] = (
                X_tf["CustomerId"]
                .map(self.customer_avg_amount)
                .fillna(self.global_avg_amount)
            )
            X_tf["Customer_Transaction_Count"] = (
                X_tf["CustomerId"].map(self.customer_tx_count).fillna(0)
            )
            X_tf["Amount_to_Customer_Avg_Ratio"] = X_tf["Amount"] / (
                X_tf["Customer_Avg_Amount"] + 1e-5
            )

        if "ProviderId" in X_tf.columns:
            X_tf["Provider_Transaction_Count"] = (
                X_tf["ProviderId"].map(self.provider_tx_count).fillna(0)
            )

        # 3. Handle 'Value' (Amount)
        if "Value" not in X_tf.columns and "Amount" in X_tf.columns:
            X_tf["Value"] = X_tf["Amount"].abs()

        X_tf["IsCredit"] = np.where(X_tf["Amount"] < 0, 1, 0).astype(int)

        # 4. Categorical Encoding
        # For production, we should ideally use OneHotEncoder, but for MVP we use get_dummies.
        # To align columns between train/test, we ensure expected columns exist.
        categorical_cols = [
            "ProviderId",
            "ProductId",
            "ProductCategory",
            "ChannelId",
        ]

        # 5. Drop Useless Columns
        cols_to_drop = [
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "CustomerId",
            "CurrencyCode",
            "CountryCode",
            "TransactionStartTime",
            "Amount",
        ]
        X_tf = X_tf.drop(
            columns=[c for c in cols_to_drop if c in X_tf.columns], errors="ignore"
        )

        # 6. Convert categorical columns to category dtype for XGBoost native categorical support
        categorical_cols = ["ProviderId", "ProductId", "ProductCategory", "ChannelId"]
        for col in categorical_cols:
            if col in X_tf.columns:
                X_tf[col] = X_tf[col].astype("category")

        # 7. Ensure columns are in the expected order to match the trained model
        expected_columns = [
            "ProviderId",
            "ProductId",
            "ProductCategory",
            "ChannelId",
            "Value",
            "PricingStrategy",
            "Hour",
            "DayOfWeek",
            "DayOfMonth",
            "Customer_Avg_Amount",
            "Customer_Transaction_Count",
            "Amount_to_Customer_Avg_Ratio",
            "Provider_Transaction_Count",
            "IsCredit",
        ]
        # Only select columns that exist and maintain the expected order
        available_columns = [col for col in expected_columns if col in X_tf.columns]
        X_tf = X_tf[available_columns]

        return X_tf


def load_and_split_xente(filepath: str, test_size: float = 0.2):
    df = pd.read_csv(filepath)
    X = df.drop("FraudResult", axis=1)
    y = df["FraudResult"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
