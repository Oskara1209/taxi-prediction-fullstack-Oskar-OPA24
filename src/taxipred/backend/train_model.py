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
