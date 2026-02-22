
# Rainfall Prediction using XGBoost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import xgboost as xgb
import shap


# 1. Load Dataset
print("Original shape:", df.shape)

# 2. Preprocessing

# Remove leakage columns
df = df.drop(columns=["precipitation_sum", "precipitation_hours"], errors="ignore")

# Convert time column
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Extract time features
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek

# Drop original time column
df = df.drop(columns=["time"], errors="ignore")

# Drop string columns
df = df.drop(columns=["sunrise", "sunset", "country"], errors="ignore")

# One-hot encode city column
df = pd.get_dummies(df, columns=["city"], drop_first=True)

# Ensure all features numeric
non_numeric = df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
if "rain_sum" in non_numeric:
    non_numeric.remove("rain_sum")

if non_numeric:
    print("Dropping non-numeric columns:", non_numeric)
    df = df.drop(columns=non_numeric)

# Convert features to float
for col in df.columns:
    if col != "rain_sum":
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

# Drop rows with NaN
df = df.dropna()

print("After preprocessing:", df.shape)

# 3. Define Features & Target

X = df.drop(columns=["rain_sum"]).astype(float)
y = df["rain_sum"]

# 4. Train / Validation / Test Split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# 5. XGBoost Model

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Evaluation

y_test_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print("\n===== MODEL PERFORMANCE =====")
print("Test RMSE:", rmse)
print("Test R2:", r2)

# 7. Actual vs Predicted Plot

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.show()

# 8. Feature Importance

plt.figure(figsize=(10,6))
xgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Important Features")
plt.show()

# 9. SHAP Explainability

print("\nGenerating SHAP plots...")

# TreeExplainer is safest for XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, max_display=15)

# Bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15)

# Dependence plot for most important feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_feature = X_test.columns[np.argmax(mean_abs_shap)]

print("Top SHAP feature:", top_feature)

shap.dependence_plot(top_feature, shap_values, X_test)

print("\n✅ SHAP Completed Successfully")