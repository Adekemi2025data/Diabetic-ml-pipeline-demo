import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

def generate_report(reference_path, current_path, output_path):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_path)

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    generate_report("drift_data/reference_diabetes.csv",
                    "drift_data/month1_diabetes.csv",
                    "reports/drift_month1.html")

    generate_report("drift_data/reference_diabetes.csv",
                    "drift_data/month2_diabetes.csv",
                    "reports/drift_month2.html")

    generate_report("drift_data/reference_diabetes.csv",
                    "drift_data/month3_diabetes.csv",
                    "reports/drift_month3.html")

    print("Drift reports generated.")
