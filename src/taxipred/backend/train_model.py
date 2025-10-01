import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib 

from taxipred.utils.constants import CLEAN_TAXI_CSV_PATH


target = "trip_price"


features = [
    "trip_distance_km",
    "trip_duration_minutes",
    "per_km_rate",
    "per_minute_rate",
    "base_fare"
]

df = pd.read_csv(CLEAN_TAXI_CSV_PATH)

X = df[features].copy()
y = df[target].copy()

print(f"Data: {df.shape[0]} rader, {df.shape[1]} kolumner")
print(f"Target: {target}")
print(f"Features ({len(features)} st): {features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

results = {}
trained = {}

for name, model in models.items():
    print(f"\nTränar {name}...")
    model.fit(X_train,y_train)
    trained[name] = model

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
    print(f"{name}: MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}")

best_name = min(results, key=lambda k: results[k]["mae"])
best_model = trained[best_name]
print(f"\nBästa modell: {best_name} (MAE={results[best_name]['mae']:.3f})")

best_model.fit(X,y)

feat_importance = []
if hasattr(best_model, "feature_importances_"):
    imps = best_model.feature_importances_
    feat_importance = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)
    print("\nFeature importance (topp 10):")
    for f, v in feat_importance[:10]:
        print(f"  {f}: {v:.4f}")

joblib.dump(
    {
    "model": best_model,
    "best_model_name": best_name,
    "features": features,
    "target": target,
    },"taxi_model.joblib",
)
joblib.dump(results, "taxi_metrics.joblib")
joblib.dump(feat_importance, "taxi_feature_importance.joblib")

print("Sparat metrics till taxi_metrics.joblib")
print("Sparat feature importance till taxi_feature_importance.joblib")


