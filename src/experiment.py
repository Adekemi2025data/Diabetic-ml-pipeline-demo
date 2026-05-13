import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ───────────────────────────────────────────────────────────────
# Base configuration
# ───────────────────────────────────────────────────────────────
config = {
    "model_type": "logistic_regression",
    "test_size": 0.2,
    "random_state": 42,
    "handle_missing": "median",
    "scale_features": True,
    "features_to_drop": [],

    # Logistic Regression
    "lr_C": 1.0,

    # Random Forest
    "rf_n_estimators": 100,
    "rf_max_depth": None,

    # Gradient Boosting
    "gb_n_estimators": 100,
    "gb_learning_rate": 0.1,
    "gb_max_depth": 3,
}

# ───────────────────────────────────────────────────────────────
# Synthetic Diabetes Dataset Generator
# ───────────────────────────────────────────────────────────────
def generate_diabetes_dataset(n_samples=768, random_state=42):
    np.random.seed(random_state)

    df = pd.DataFrame({
        "Pregnancies": np.random.poisson(3, n_samples),
        "Glucose": np.random.normal(120, 30, n_samples).clip(50, 200),
        "BloodPressure": np.random.normal(70, 12, n_samples).clip(40, 100),
        "SkinThickness": np.random.normal(20, 10, n_samples).clip(0, 50),
        "Insulin": np.random.normal(80, 40, n_samples).clip(0, 300),
        "BMI": np.random.normal(32, 7, n_samples).clip(15, 50),
        "DiabetesPedigreeFunction": np.random.exponential(0.5, n_samples).clip(0, 3),
        "Age": np.random.normal(33, 12, n_samples).clip(18, 80),
    })

    # Probability model for Outcome
    logits = (
        0.03 * df["Pregnancies"] +
        0.04 * df["Glucose"] +
        0.02 * df["BMI"] +
        0.03 * df["Age"] -
        8
    )
    probs = 1 / (1 + np.exp(-logits))
    df["Outcome"] = (np.random.rand(n_samples) < probs).astype(int)

    return df

# ───────────────────────────────────────────────────────────────
# Data Preparation
# ───────────────────────────────────────────────────────────────
def load_and_prepare_data(config):
    print("Generating synthetic diabetes dataset...")
    df = generate_diabetes_dataset()
    print(f"Generated {len(df)} rows, {len(df.columns)} columns")

    # Drop user-specified columns
    if config["features_to_drop"]:
        df = df.drop(columns=config["features_to_drop"], errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Handle missing values
    if config["handle_missing"] == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif config["handle_missing"] == "drop":
        df = df.dropna()

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y, len(df), numeric_cols, []

# ───────────────────────────────────────────────────────────────
# Model Builder
# ───────────────────────────────────────────────────────────────
def build_model(config):
    if config["model_type"] == "logistic_regression":
        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000
        )

    elif config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"]
        )

    elif config["model_type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"]
        )

    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

# ───────────────────────────────────────────────────────────────
# MLflow Experiment Runner
# ───────────────────────────────────────────────────────────────
def run_experiment(config):
    mlflow.set_experiment("diabetes-prediction")

    with mlflow.start_run():

        # Log config parameters
        for k, v in config.items():
            mlflow.log_param(k, v)

        # Load data
        X, y, n_rows, numeric_cols, _ = load_and_prepare_data(config)

        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", X.shape[1])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=y
        )

        # Scale
        if config["scale_features"]:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Train
        model = build_model(config)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, round(v, 4))

        # Save model
        mlflow.sklearn.log_model(model, "model")

        # Save config snapshot
        with open("config_snapshot.json", "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact("config_snapshot.json")
        os.remove("config_snapshot.json")

        # Print results
        print("\nResults:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # Capture run ID
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")

    return run_id
