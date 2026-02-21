import pandas as pd

df = pd.read_csv("SriLanka_Weather_Dataset.csv")

print("Rows, Columns =", df.shape)
print("Column names =", list(df.columns))
print(df.head(3))
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("\nCategorical Columns:")
print(df.select_dtypes(include=["object"]).columns)