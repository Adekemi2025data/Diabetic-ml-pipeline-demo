import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset



def compute_feature_drift(reference_df, current_df, feature):
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)
    result_dict = snapshot.dict()

    drift_score = None

    for metric in result_dict["metrics"]:
        metric_id = metric["metric_id"]

        # Match the metric for the specific feature
        if metric_id == f"ValueDrift(column={feature})":
            drift_score = float(metric["value"])
            break

    if drift_score is None:
        raise KeyError(f"No drift metric found for feature '{feature}'")

    # Drift detected if p-value < 0.05
    drift_detected = drift_score < 0.05

    return {
        "drift_score": drift_score,
        "drift_detected": drift_detected
    }



def main():
    # Load your dataset
    df = pd.read_csv("data/diabetes.csv")

    # Define your feature list
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    # Split into reference + monthly batches
    reference = df.sample(frac=0.3, random_state=42)
    month1 = df.sample(frac=0.3, random_state=1)
    month2 = df.sample(frac=0.3, random_state=2)
    month3 = df.sample(frac=0.3, random_state=3)

    timeline_results = {"timeline": {}}

    for feature in features:
        print(f"Processing feature: {feature}")

        timeline_results["timeline"][feature] = {
            "month1": compute_feature_drift(reference, month1, feature),
            "month2": compute_feature_drift(reference, month2, feature),
            "month3": compute_feature_drift(reference, month3, feature),
        }

    # Save results
    pd.DataFrame.from_dict(
        {f: timeline_results["timeline"][f] for f in features},
        orient="index"
    ).to_csv("diabetes_feature_drift_timeline.csv")

    print("Timeline drift analysis saved to diabetes_feature_drift_timeline.csv")


if __name__ == "__main__":
    main()
