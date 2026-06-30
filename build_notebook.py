import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md("""
# Agricultural Supply Chain Loss Prediction

Predicting the percentage of crop loss (`loss_percent`) for a shipment based on
storage conditions, transport mode, and travel details.

**Note on data:** Real shipment-level spoilage data is not publicly available at
the scale needed for this project, so the dataset (`data/shipments.csv`) is
synthetically generated using domain-informed rules (see `generate_data.py`):
longer storage, higher humidity, higher temperature, and longer delays all
increase spoilage, with random noise added so relationships aren't perfectly
clean. This keeps the project honest while still letting us practice the full
data science workflow: cleaning, SQL-style analysis, EDA, feature engineering,
and modeling.
"""))

cells.append(md("## 1. Setup"))
cells.append(code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
"""))

cells.append(md("## 2. Load Data"))
cells.append(code("""
df = pd.read_csv("../data/shipments.csv")
print(df.shape)
df.head()
"""))

cells.append(md("""
## 3. Data Cleaning

Checking for and fixing:
- missing values
- duplicate rows
- obviously invalid values (e.g. negative days, humidity outside 0-100%)
"""))
cells.append(code("""
print("Missing values per column:")
print(df.isna().sum())
print("\\nDuplicate rows:", df.duplicated().any() and df.duplicated().sum() or df.duplicated(subset=df.columns.difference(['shipment_id'])).sum())
"""))
cells.append(code("""
# Drop exact duplicate shipments (ignoring shipment_id, since IDs are always unique)
before = len(df)
df = df.drop_duplicates(subset=df.columns.difference(["shipment_id"]))
print(f"Dropped {before - len(df)} duplicate rows")

# Fill missing humidity with the median (robust to outliers)
df["humidity"] = df["humidity"].fillna(df["humidity"].median())

# Sanity-check value ranges
df = df[(df["humidity"].between(0, 100)) & (df["storage_days"] >= 0) & (df["loss_percent"].between(0, 100))]

print(df.shape)
df.isna().sum().sum(), "missing values remaining"
"""))

cells.append(md("""
## 4. SQL-Style Analysis

Using pandas to answer the same questions SQL `GROUP BY` queries would.
Equivalent SQL is shown in comments for reference.
"""))
cells.append(code("""
# SQL: SELECT crop_type, AVG(loss_percent) FROM shipments GROUP BY crop_type ORDER BY 2 DESC;
df.groupby("crop_type")["loss_percent"].mean().sort_values(ascending=False).round(2)
"""))
cells.append(code("""
# SQL: SELECT region, AVG(loss_percent) FROM shipments GROUP BY region ORDER BY 2 DESC;
df.groupby("region")["loss_percent"].mean().sort_values(ascending=False).round(2)
"""))
cells.append(code("""
# SQL: SELECT transport_mode, AVG(loss_percent) FROM shipments GROUP BY transport_mode ORDER BY 2 DESC;
df.groupby("transport_mode")["loss_percent"].mean().sort_values(ascending=False).round(2)
"""))

cells.append(md("## 5. Exploratory Data Analysis"))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=df, x="storage_days", y="loss_percent", alpha=0.4, ax=axes[0])
axes[0].set_title("Storage Days vs Loss %")
sns.scatterplot(data=df, x="humidity", y="loss_percent", alpha=0.4, ax=axes[1])
axes[1].set_title("Humidity vs Loss %")
plt.tight_layout()
plt.savefig("../images/scatter_storage_humidity.png", dpi=120)
plt.show()
"""))
cells.append(code("""
plt.figure(figsize=(6, 4))
df["loss_percent"].hist(bins=30)
plt.title("Distribution of Loss %")
plt.xlabel("loss_percent")
plt.savefig("../images/loss_distribution.png", dpi=120)
plt.show()
"""))
cells.append(code("""
plt.figure(figsize=(7, 5))
numeric_cols = df.select_dtypes(include=np.number).drop(columns=["shipment_id"])
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../images/correlation_heatmap.png", dpi=120)
plt.show()
"""))
cells.append(code("""
plt.figure(figsize=(8, 5))
df.groupby("crop_type")["loss_percent"].mean().sort_values().plot(kind="barh")
plt.title("Average Loss % by Crop Type")
plt.xlabel("Average loss_percent")
plt.tight_layout()
plt.savefig("../images/loss_by_crop.png", dpi=120)
plt.show()
"""))

cells.append(md("""
**Observations:**
- Loss % tends to rise with storage_days and humidity, consistent with how spoilage works in practice.
- Perishables (tomato, banana, mango) show higher average loss than grains (wheat, rice).
- Open Cart transport shows the highest average loss; Refrigerated Truck the lowest.
"""))

cells.append(md("""
## 6. Feature Engineering

Adding a simple `risk_score` combining temperature and humidity, since heat and
moisture together accelerate spoilage more than either alone.
"""))
cells.append(code("""
df["risk_score"] = df["temperature"] * df["humidity"] / 100
df[["temperature", "humidity", "risk_score", "loss_percent"]].head()
"""))

cells.append(md("## 7. Model Training"))
cells.append(code("""
features = ["crop_type", "storage_days", "temperature", "humidity",
            "distance_km", "transport_mode", "delay_hours", "region", "risk_score"]
target = "loss_percent"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = ["crop_type", "transport_mode", "region"]
numeric_cols = [c for c in features if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")
"""))

cells.append(md("### Baseline: Linear Regression"))
cells.append(code("""
lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

print("Linear Regression")
print("R2:", round(r2_score(y_test, lr_preds), 3))
print("MAE:", round(mean_absolute_error(y_test, lr_preds), 3))
"""))

cells.append(md("### Final Model: Random Forest"))
cells.append(code("""
rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

print("Random Forest")
print("R2:", round(r2_score(y_test, rf_preds), 3))
print("MAE:", round(mean_absolute_error(y_test, rf_preds), 3))
"""))

cells.append(code("""
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_preds, alpha=0.4)
plt.plot([0, y_test.max()], [0, y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual loss_percent")
plt.ylabel("Predicted loss_percent")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.savefig("../images/actual_vs_predicted.png", dpi=120)
plt.show()
"""))

cells.append(md("""
## 8. Feature Importance
"""))
cells.append(code("""
ohe = rf_pipeline.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = list(cat_feature_names) + numeric_cols

importances = rf_pipeline.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
importance_df = importance_df.sort_values("importance", ascending=False).head(12)

plt.figure(figsize=(8, 6))
sns.barplot(data=importance_df, y="feature", x="importance")
plt.title("Top Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("../images/feature_importance.png", dpi=120)
plt.show()
"""))

cells.append(md("""
## 9. Summary of Results

| Model | R² | MAE |
|---|---|---|
| Linear Regression (baseline) | see output above | see output above |
| Random Forest (final) | see output above | see output above |

**Key insights (derived from this synthetic dataset):**
- Storage duration and humidity are among the strongest drivers of crop loss.
- Loss accelerates noticeably once storage exceeds roughly 5-7 days.
- Refrigerated transport meaningfully reduces loss compared to open-cart transport.
- Perishable crops (tomato, banana, mango) consistently show higher average loss
  than grains (wheat, rice), as expected.

**Limitations:** This dataset is synthetically generated, so it demonstrates the
analysis workflow rather than ground-truth agricultural outcomes. To productionize
this, the simulation rules in `generate_data.py` should be replaced or validated
against real shipment/spoilage records (e.g. from FAO, government open data
portals, or cold-chain logistics partners).
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/agri_loss_prediction.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook written.")
