import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# --------------------------
# Load
# --------------------------
df = pd.read_csv("SriLanka_Weather_Dataset.csv")

# --------------------------
# Preprocessing (same as training)
# --------------------------
df = df.drop(columns=["precipitation_sum", "precipitation_hours"], errors="ignore")

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek
df = df.drop(columns=["time"], errors="ignore")

df = df.drop(columns=["sunrise", "sunset", "country"], errors="ignore")

# One-hot encode city
df = pd.get_dummies(df, columns=["city"], drop_first=True)

# Make sure numeric
non_numeric = df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
if "rain_sum" in non_numeric:
    non_numeric.remove("rain_sum")
df = df.drop(columns=non_numeric, errors="ignore")

for col in df.columns:
    if col != "rain_sum":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

df = df.dropna()

X = df.drop(columns=["rain_sum"]).astype(float)
y = df["rain_sum"]

# Save feature columns for app
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

# Train model on full data
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X, y)

# Save model
joblib.dump(model, "xgb_rain_model.pkl")

print("✅ Model saved as xgb_rain_model.pkl")
print("✅ Feature columns saved as feature_columns.pkl")