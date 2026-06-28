import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Configuration
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNET_CSV = os.path.join(ROOT_DIR, "data", "fish_frames_unet.csv")

feature_cols = [
    "Length (cm)", "Width (cm)", "Height (cm)",
    "Area (cm²)", "Perimeter (cm)",
    "Volume (cm³)", "Surface Area (cm²)",
    "Aspect Ratio", "Elongation", "Compactness",
    "Rectangularity", "Equivalent Diameter (cm)"
]

def evaluate_dataset(csv_path, model, model_name, dataset_name):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None, None
        
    df = pd.read_csv(csv_path)
    # Drop rows with NaN in features or target
    df = df.dropna(subset=feature_cols + ["Weight (g)", "FishID"])
    
    if len(df) == 0:
        print(f"Error: No valid rows in {csv_path}.")
        return None, None
        
    X = df[feature_cols].values
    y = df["Weight (g)"].values
    groups = df["FishID"].values
    
    logo = LeaveOneGroupOut()
    # Safely parallelizing the folds while restricting individual models to prevent deadlock
    preds = cross_val_predict(model, X, y, groups=groups, cv=logo, n_jobs=-1)
    
    # Per-fish aggregation for Weight
    pred_df = pd.DataFrame({"FishID": groups, "True": y, "Pred": preds})
    
    fish_avg = pred_df.groupby("FishID").agg(
        True_Weight=("True", "first"),
        Pred_Weight=("Pred", "median")
    ).reset_index()
    
    weight_mae = mean_absolute_error(fish_avg["True_Weight"], fish_avg["Pred_Weight"])
    weight_rmse = np.sqrt(mean_squared_error(fish_avg["True_Weight"], fish_avg["Pred_Weight"]))
    weight_r2 = r2_score(fish_avg["True_Weight"], fish_avg["Pred_Weight"])
    weight_mape = np.mean(np.abs(fish_avg["True_Weight"] - fish_avg["Pred_Weight"]) / fish_avg["True_Weight"]) * 100
    
    return weight_mae, weight_rmse, weight_r2, weight_mape

def get_dimension_errors(csv_path):
    if not os.path.exists(csv_path): return None, None
    df = pd.read_csv(csv_path).dropna(subset=feature_cols + ["Weight (g)", "FishID"])
    if len(df) == 0: return None, None
    dim_avg = df.groupby("FishID").agg(
        True_Length=("True_Length_cm", "first"),
        Pred_Length=("Length (cm)", "median"),
        True_Height=("True_Height_cm", "first"),
        Pred_Height=("Height (cm)", "median")
    ).dropna()
    len_mae = mean_absolute_error(dim_avg["True_Length"], dim_avg["Pred_Length"])
    hgt_mae = mean_absolute_error(dim_avg["True_Height"], dim_avg["Pred_Height"])
    return len_mae, hgt_mae

def main():
    print("=" * 70)
    print("WEIGHT ESTIMATION PIPELINE COMPARISON (SAM vs UNet)")
    print("=" * 70)
    
    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge Regression":  Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1),
        "XGBoost":           xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1),
        "LightGBM":          lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=1),
        "Ensemble (Voting)": VotingRegressor(estimators=[
            ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)),
            ('et', ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1))
        ], n_jobs=1),
        "Super Ensemble":    VotingRegressor(estimators=[
            ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)),
            ('lgb', lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=1)),
            ('et', ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1))
        ], n_jobs=1)
    }
    
    print("\n[1] FEATURE EXTRACTION DIMENSION ERRORS (Independent of Regression Model)")
    print("-" * 70)
    unet_len, unet_hgt = get_dimension_errors(UNET_CSV)
    
    if unet_len is not None:
        print(f"UNet Advanced  -> Length MAE: {unet_len:>6.2f} cm | Height MAE: {unet_hgt:>6.2f} cm")
    
    print("\n[2] WEIGHT ESTIMATION REGRESSION ERRORS (Leave-One-Fish-Out CV)")
    print("-" * 70)
    
    for model_name, model in models.items():
        print(f"\n--- Model: {model_name} ---")
        
        unet_weight_mae, unet_weight_rmse, unet_weight_r2, unet_weight_mape = evaluate_dataset(UNET_CSV, model, model_name, "UNet Advanced Masks")
        
        if unet_weight_mae is not None:
            print(f"UNet Pipeline  -> Weight MAE: {unet_weight_mae:>6.2f} g  | RMSE: {unet_weight_rmse:>6.2f} g | R2: {unet_weight_r2:>6.4f} | MAPE: {unet_weight_mape:>5.1f}%")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
