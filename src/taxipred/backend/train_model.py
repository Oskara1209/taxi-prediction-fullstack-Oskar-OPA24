import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib 

from taxipred.utils.constants import CLEAN_TAXI_CSV_PATH

df = pd.read_csv(CLEAN_TAXI_CSV_PATH)

target = "trip_price"

X = df.drop(columns=[target])
y = df[target]

X = pd.get_dummies(X, drop_first= False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": ("Scaled", LinearRegression()),
    "Random forest":("Raw", RandomForestRegressor(n_estimators=300, random_state=42)),
    "Gradient booset": ("Raw", GradientBoostingRegressor(n_estimators=300, random_state=42))
}

results = {}
for name, (mode, model) in models.items():
    if mode == "Scaled":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model, "mode": mode}
