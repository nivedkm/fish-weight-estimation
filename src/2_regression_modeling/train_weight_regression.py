"""
Fish Weight Estimation — Regression Model Training
===================================================
Takes the existing clean dataset (which already has dimensions + known weights)
and trains multiple regression models to predict Weight from dimensions.

Input:  fish_frames_200_ENHANCED_clean_unique_no_repeat.csv
Output: Prints model comparison, best model accuracy, and saves predictions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Load Dataset ──────────────────────────────────────────────────
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(ROOT_DIR, "data", "fish_frames_unet.csv")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows, {df['FishID'].nunique()} fish\n")

# ── Features & Target ────────────────────────────────────────────
feature_cols = [
    "Length (cm)", "Width (cm)", "Height (cm)",
    "Area (cm²)", "Perimeter (cm)",
    "Volume (cm³)", "Surface Area (cm²)",
    "Aspect Ratio", "Elongation", "Compactness",
    "Rectangularity", "Equivalent Diameter (cm)"
]

X = df[feature_cols].values
y = df["Weight (g)"].values
groups = df["FishID"].values  # For Leave-One-Fish-Out CV

# ── Models to Compare ────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge Regression":  Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
}

# ── Leave-One-Fish-Out Cross Validation ──────────────────────────
# This is the gold standard for fish research:
# Train on 14 fish, predict the held-out fish. Repeat for all 15.
logo = LeaveOneGroupOut()

print("=" * 70)
print("MODEL COMPARISON — Leave-One-Fish-Out Cross Validation")
print("=" * 70)

best_mae = float("inf")
best_name = ""
best_preds = None

for name, model in models.items():
    preds = cross_val_predict(model, X, y, groups=groups, cv=logo)
    
    # Per-fish: average predicted weight vs true weight
    pred_df = pd.DataFrame({"FishID": groups, "True": y, "Pred": preds})
    fish_avg = pred_df.groupby("FishID").agg(
        True_Weight=("True", "first"),
        Pred_Weight=("Pred", "median")
    ).reset_index()
    
    mae = mean_absolute_error(fish_avg["True_Weight"], fish_avg["Pred_Weight"])
    r2 = r2_score(fish_avg["True_Weight"], fish_avg["Pred_Weight"])
    mape = np.mean(np.abs(fish_avg["True_Weight"] - fish_avg["Pred_Weight"]) / fish_avg["True_Weight"]) * 100
    
    print(f"\n  {name}:")
    print(f"    MAE  = {mae:.2f} g")
    print(f"    MAPE = {mape:.1f}%")
    print(f"    R²   = {r2:.4f}")
    
    if mae < best_mae:
        best_mae = mae
        best_name = name
        best_preds = fish_avg.copy()

# ── Best Model Results ────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"BEST MODEL: {best_name}  (MAE = {best_mae:.2f} g)")
print("=" * 70)

print("\nPer-Fish Predictions (Leave-One-Fish-Out):")
print("-" * 50)
print(f"{'FishID':<10} {'True (g)':>10} {'Predicted (g)':>14} {'Error (g)':>10}")
print("-" * 50)

for _, row in best_preds.iterrows():
    err = row["Pred_Weight"] - row["True_Weight"]
    print(f"{row['FishID']:<10} {row['True_Weight']:>10.2f} {row['Pred_Weight']:>14.2f} {err:>+10.2f}")

OUT_PATH = os.path.join(ROOT_DIR, "data", "weight_predictions_LOFO.csv")
best_preds.to_csv(OUT_PATH, index=False)
print(f"\nSaved predictions to: {OUT_PATH}")
