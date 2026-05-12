import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path so we can import preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import DiabetesPreprocessor

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CONFIG = {
    "data_url": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "target": "Outcome",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 200,
    "max_depth": 8,
    "min_accuracy": 0.60,
    "min_f1": 0.55,
}

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
def load_data(url):
    print(f"Loading diabetes dataset from {url}...")
    df = pd.read_csv(url)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

# ---------------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------------
def train_model(config=None):
    if config is None:
        config = CONFIG

    # Load raw data
    df = load_data(config["data_url"])

    # Initialize preprocessor
    prep = DiabetesPreprocessor()

    # Validate + clean
    prep.validate(df)
    df_clean = prep.run(df)

    # Quality report
    quality = prep.quality_report(df_clean)
    print(f"Data quality: {quality['total_nulls']} nulls, {quality['duplicate_rows']} duplicates")

    # Split
    X = df_clean.drop(columns=[config["target"]])
    y = df_clean[config["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config["random_state"]
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
    }

    print("\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1 Score:  {metrics['f1_score']}")

    # Threshold checks
    if metrics["accuracy"] < config["min_accuracy"]:
        print(f"\nWARNING: Accuracy {metrics['accuracy']} is below threshold {config['min_accuracy']}")
    if metrics["f1_score"] < config["min_f1"]:
        print(f"\nWARNING: F1 {metrics['f1_score']} is below threshold {config['min_f1']}")

    # Save model + scaler
    os.makedirs("models", exist_ok=True)

    model_path = "models/diabetes_model.pkl"
    scaler_path = "models/diabetes_scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/diabetes_results.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")

    return metrics

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    metrics = train_model()

    if metrics["accuracy"] < CONFIG["min_accuracy"]:
        print("\nFAILED: Accuracy below threshold")
        sys.exit(1)

    if metrics["f1_score"] < CONFIG["min_f1"]:
        print("\nFAILED: F1 score below threshold")
        sys.exit(1)

    print("\nAll thresholds passed!")
