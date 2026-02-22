import pandas as pd

df = pd.read_csv("SriLanka_Weather_Dataset.csv")

# 1) Missing values
print("Missing values (top):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# 2) Duplicates
print("Duplicate rows:", df.duplicated().sum())

# 3) Convert time + create features
df["time"] = pd.to_datetime(df["time"])
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek

# 4) Leakage removal 
df = df.drop(columns=["precipitation_sum"], errors="ignore")

# 5) One-hot encoding (city)
df = pd.get_dummies(df, columns=["city"], drop_first=True)

print(df.head(3))
print("After preprocessing shape:", df.shape)