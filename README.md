***

# 🚀 Enterprise Fraud Detection System: Xente (Fintech)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)

## 📌 Project Overview
This project is an end-to-end, production-ready Machine Learning pipeline designed to detect fraudulent financial transactions in real-time. Built using the **Xente Fraud Detection Challenge** dataset (Ugandan Mobile Money/E-commerce), it handles high-cardinality categorical data, extreme class imbalance, and real-world temporal dynamics.

Unlike standard exploratory data science projects, this repository implements **MLOps best practices**, including a custom Scikit-Learn feature engineering pipeline, MLflow experiment tracking, a FastAPI real-time serving layer, Docker containerization, and automated CI/CD pipelines via GitHub Actions.

## 🏗️ Architecture & Tech Stack
*   **Machine Learning**: Scikit-Learn (Pipelines, Custom Transformers), XGBoost
*   **Experiment Tracking**: MLflow (Parameters, Metrics, Artifacts, Model Signatures)
*   **Model Explainability**: SHAP (TreeExplainer)
*   **Real-Time API Serving**: FastAPI, Pydantic, Uvicorn
*   **Containerization**: Docker
*   **CI/CD**: GitHub Actions (Black, Flake8, Docker Build & Push to DockerHub)

## 🧠 Data Strategy & Feature Engineering
Real-world fraud is adversarial and rapid. Raw IDs (`TransactionId`, `BatchId`) were intentionally dropped to prevent model overfitting (memorization) and ensure robust generalization to "zero-day" accounts. 

Instead, a custom `XentePreprocessor` was built to extract **Velocity and Frequency Features**, bridging the gap between batch training and real-time inference:
*   **Temporal Engineering**: Extracted hour, day of week, and day of month from raw datetimes.
*   **Customer Velocity**: Mapped historical transaction counts (`Customer_Transaction_Count`).
*   **Behavioral Anomaly Detection**: Calculated the `Amount_to_Customer_Avg_Ratio` to instantly flag transactions that deviate severely from a specific customer's historical spending baseline.

## 📊 Model Performance
Optimized strictly for the **Area Under the Precision-Recall Curve (AUPRC)** to handle extreme class imbalance (fraud = <0.2% of data).

*   **Primary Metric (AUPRC):** `0.778`
*   **Recall:** Captures `97%` of all fraudulent transactions.
*   **False Positive Rate:** `0.026%` (Falsely flags less than one-tenth of one percent of legitimate transactions, ensuring extremely low customer friction/churn).

*(Precision-Recall curves, Normalized Confusion Matrices, and SHAP Summary plots are automatically tracked and saved as artifacts in the MLflow tracking server).*

---

## 📂 Repository Structure
```text
├── .github/workflows/
│   └── ci.yml                # Automated Linting, Docker Build, and Hub Push
├── data/
│   ├── raw/                  # training.csv (Ignored in git)
├── notebooks/
│   └── 02_xente_experiment.ipynb # EDA, Optuna Tuning, SHAP Explainability
├── reports/                  # Confusion Matrix, PR Curves (Ignored in git)
├── src/
│   ├── features.py           # Custom OOP Scikit-Learn Transformer
│   ├── train.py              # Model Training & MLflow Artifact Logging
│   └── serve.py              # FastAPI Application & Pydantic Schemas
├── Dockerfile                # Production Container definition
├── requirements.txt          # Pinned dependencies
└── README.md
```

---

## ⚙️ Local Development Setup

**1. Clone the repository and set up a virtual environment:**
```bash
git clone https://github.com/yourusername/fraud-detection-api.git
cd fraud-detection-api
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the Model & Track with MLflow:**
Ensure the raw data is located at `data/raw/train.csv`, then run:
```bash
python src/train.py
```
*(This will generate a local SQLite `mlflow.db` and an `mlruns/` directory. Copy the **Run ID** printed in the terminal).*

**4. Start the FastAPI Server:**
Update the `MODEL_URI` in `src/serve.py` with your Run ID, then launch the server:
```bash
uvicorn src.serve:app --reload --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker Deployment

The application is fully containerized. You can build and run it without requiring any local Python environments.

**Build the image:**
```bash
docker build -t xente-fraud-api:latest .
```

**Run the container:**
```bash
docker run -p 8000:8000 xente-fraud-api:latest
```

---

## 🔄 CI/CD Pipeline
This repository uses a robust GitHub Actions workflow defined in `.github/workflows/ci.yml`. 
Upon a push or pull request to the `main` branch, the pipeline automatically:
1. Provisions an Ubuntu runner and sets up Python 3.10.
2. Formats code using **Black** and lints via **Flake8**.
3. Builds the Docker container to ensure environment integrity.
4. **(On Push to Main):** Authenticates and pushes the successfully built image to **Docker Hub** tagged with both `latest` and the specific commit `SHA`.

---

## 📡 API Usage & Documentation

Once the server is running (either locally or via Docker), navigate to `http://127.0.0.1:8000/docs` to view the interactive Swagger UI.

### Example Request (`POST /predict`)
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "TransactionId": "TransactionId_51888",
  "CustomerId": "CustomerId_1",
  "ProviderId": "ProviderId_4",
  "ProductId": "ProductId_10",
  "ProductCategory": "airtime",
  "ChannelId": "ChannelId_2",
  "Amount": -10000.0,
  "Value": 10000.0,
  "TransactionStartTime": "2018-11-21T16:49:14Z",
  "PricingStrategy": 4
}'
```

### Example Response (`200 OK`)
```json
{
  "transaction_id": "TransactionId_51888",
  "Product_Category": "airtime",
  "fraudResult": false,
  "fraud_probability": 0.00004561
}
```

***
*Developed with a focus on scalable MLOps, behavioral feature engineering, and robust software architecture.*
