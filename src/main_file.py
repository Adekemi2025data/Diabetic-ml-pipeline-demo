import pandas as pd

df = pd.read_csv("data/diabetes.csv")

print(df.head())
print(df.isnull().sum())
print(df.info())
print(df.shape)
print(df.duplicated().sum())
print((df == 0).sum())
cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_invalid_zeros:
    df[col] = df[col].replace(0, df[col].median())
print((df == 0).sum())