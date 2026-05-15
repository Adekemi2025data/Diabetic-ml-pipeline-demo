import pandas as pd
import numpy as np
import os

def load_reference():
    df = pd.read_csv("drift_data/reference_diabetes.csv")
    return df

def simulate_month(df, month):
    df_new = df.copy()

    if month == 1:
        df_new["Glucose"] *= 1.02
        df_new["BMI"] *= 1.01

    if month == 2:
        df_new["Glucose"] *= 1.05
        df_new["BloodPressure"] *= 0.97
        df_new["Age"] += 1

    if month == 3:
        df_new["Glucose"] *= 1.10
        df_new["BloodPressure"] *= 0.95
        df_new["SkinThickness"] *= 1.15
        df_new["BMI"] *= 1.08
        df_new["Age"] += 2

    return df_new

if __name__ == "__main__":
    os.makedirs("drift_data", exist_ok=True)

    df_ref = load_reference()

    simulate_month(df_ref, 1).to_csv("drift_data/month1_diabetes.csv", index=False)
    simulate_month(df_ref, 2).to_csv("drift_data/month2_diabetes.csv", index=False)
    simulate_month(df_ref, 3).to_csv("drift_data/month3_diabetes.csv", index=False)

    print("Drift simulation files created.")
