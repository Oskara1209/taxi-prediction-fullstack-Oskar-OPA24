import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib 

from taxipred.utils.constants import CLEAN_TAXI_CSV_PATH


target = "trip_price"


numeric_features = [
    "trip_distance_km",
    "trip_duration_minutes",
    "per_km_rate",
    "per_minute_rate",
    "base_fare"
]

categorical_features = ["weather", "traffic_conditions"]

df = pd.read_csv(CLEAN_TAXI_CSV_PATH)

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

print(f"Data: {df.shape[0]} rader, {df.shape[1]} kolumner")
print(f"Target: {target}")
print(f"Num Features {numeric_features}")
print(f"Cat Fearures {categorical_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", "passthrough", numeric_features),
    ],
    remainder="drop",
    verbose_feature_names_out= False
)

print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

pipelines = {
    "LinearRegression": Pipeline(steps=[
        ("prep", preprocess),
        ("model", LinearRegression())
    ]),
    "RandomForest": Pipeline(steps=[
        ("prepp", preprocess),
        ("model", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1 ))
    ]),
    "GradientBoosting": Pipeline(steps=[
        ("prep", preprocess),
        ("model", GradientBoostingRegressor(random_state=42) )
    ])
}

results = {}
trained = {}

for name, pipe in pipelines.items():
    print(f"\nTränar {name}...")
    pipe.fit(X_train,y_train)
    trained[name] = pipe

    preds = pipe.predict(X_test)
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
inner_model = getattr(best_model, "named_steps", {}).get("model", None)
if hasattr(inner_model, "feature_importances_"):
    feature_names = best_model.named_steps["prep"].get_feature_names_out()
    imps = inner_model.feature_importances_
    feat_importance = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
    print("\nFeature importance (topp 10):")
    for f, v in feat_importance[:10]:
        print(f"  {f}: {v:.4f}")

def group_metrics(df_eval, model, by_cols):
    g = []
    for keys, grp in df_eval.groupby(by_cols):
        Xg = grp[numeric_features + categorical_features]
        yg = grp[target]
        pg = model.predict(Xg)
        g.append({
            "group": keys if isinstance(keys, tuple) else (keys,),
            "n": len(grp),
            "MAE": mean_absolute_error(yg, pg),
            "RMSE": np.sqrt(mean_squared_error(yg, pg)),
            "R2": r2_score(yg, pg)
        })
    return pd.DataFrame(g).sort_values("MAE")

print("\nPer-weather metrics:")
print(group_metrics(df, best_model, ["weather"]).to_string(index=False))

print("\nPer-traffic metrics:")
print(group_metrics(df, best_model, ["traffic_conditions"]).to_string(index=False))

print("\nPer (weather, traffic) metrics:")
print(group_metrics(df, best_model, ["weather", "traffic_conditions"]).to_string(index=False))

joblib.dump(best_model, "taxi_model.joblib")
joblib.dump(results, "taxi_metrics.joblib")
joblib.dump(feat_importance, "taxi_feature_importance.joblib")

print("Sparat metrics till taxi_metrics.joblib")
print("Sparat feature importance till taxi_feature_importance.joblib")


