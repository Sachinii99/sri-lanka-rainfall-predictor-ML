# ==========================================
# XAI Analysis: SHAP + LIME + Feature Importance + PDP
# Rainfall Prediction using XGBoost (Regression)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Fix blank SHAP windows in some environments
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

from xgboost import XGBRegressor
import xgboost as xgb

import shap
from lime.lime_tabular import LimeTabularExplainer


# 1) Load Dataset
df = pd.read_csv("SriLanka_Weather_Dataset.csv")
print("Original shape:", df.shape)

# 2) Preprocessing
# Drop leakage-prone precipitation aggregates (keep ONLY rain_sum as target)
df = df.drop(columns=["precipitation_sum", "precipitation_hours"], errors="ignore")

# Convert time and create time features
df["time"] = pd.to_datetime(df["time"])
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek
df = df.drop(columns=["time"], errors="ignore")

# Drop string columns (XGBoost cannot take raw strings)
df = df.drop(columns=["sunrise", "sunset", "country"], errors="ignore")

# One-hot encode city
df = pd.get_dummies(df, columns=["city"], drop_first=True)

print("After preprocessing:", df.shape)

# 3) Features & Target
TARGET = "rain_sum"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Safety: ensure all dtypes are numeric/bool
non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
if non_numeric:
    print("Dropping non-numeric columns:", non_numeric)
    X = X.drop(columns=non_numeric)

# 4) Split (Train/Val/Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

# 5) Train XGBoost Model
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

print("\nModel trained. Generating XAI outputs...")

# A) Feature Importance (XGBoost built-in)
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10)
plt.title("XGBoost Feature Importance (Top 10)")
plt.tight_layout()
plt.show()

# B) SHAP (Global + Local)
print("\n[SHAP] Computing SHAP values...")

# TreeExplainer is more stable for XGBoost
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test)

# 1) Global SHAP summary (beeswarm)
shap.summary_plot(shap_values, X_test, max_display=15)
plt.show()

# 2) Global SHAP bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15)
plt.show()

# 3) SHAP dependence plot for the top SHAP feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = int(np.argmax(mean_abs_shap))
top_feature = X_test.columns[top_idx]
print("[SHAP] Top feature:", top_feature)

shap.dependence_plot(top_feature, shap_values, X_test, show=True)

# 4) Local explanation (single instance waterfall)
row_idx = 0
print("[SHAP] Local explanation for test row:", row_idx)

# Build Explanation object for waterfall
base_value = shap_explainer.expected_value
exp = shap.Explanation(
    values=shap_values[row_idx],
    base_values=base_value,
    data=X_test.iloc[row_idx],
    feature_names=X_test.columns
)

shap.plots.waterfall(exp, show=True)

# C) LIME (Local explanation)
print("\n[LIME] Generating local explanation...")

feature_names = X_train.columns.tolist()

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=True
)

i = 0  # test instance index
lime_exp = lime_explainer.explain_instance(
    data_row=X_test.values[i],
    predict_fn=model.predict,
    num_features=10
)

print("\nLIME Explanation (Top 10 features):")
for f, w in lime_exp.as_list():
    print(f"{f}: {w}")

fig = lime_exp.as_pyplot_figure()
plt.title("LIME Local Explanation (Top 10)")
plt.tight_layout()
plt.show()

# D) PDP (Partial Dependence Plots)
print("\n[PDP] Generating PDP plots...")

# Choose top 3 SHAP features for PDP
top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
top3_features = [X_test.columns[i] for i in top3_idx]
print("[PDP] Top 3 features:", top3_features)

for feat in top3_features:
    PartialDependenceDisplay.from_estimator(
        model, X_test, [feat], grid_resolution=20
    )
    plt.title(f"Partial Dependence Plot (PDP) - {feat}")
    plt.tight_layout()
    plt.show()

print("\n✅ XAI completed successfully!")

