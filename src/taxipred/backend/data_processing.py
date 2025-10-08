from taxipred.utils.constants import CLEAN_TAXI_CSV_PATH
import pandas as pd
from pydantic import BaseModel, Field
import json
from typing import Optional


    
class Rates(BaseModel):
    per_km_rate: float
    per_minute_rate: float
    base_fare: float

class TripInput(BaseModel):
    trip_distance_km: float 
    trip_duration_minutes: float
    per_km_rate: float
    per_minute_rate: float
    base_fare: float

    @classmethod
    def with_defaults(cls, distance: float, duration: float, rates: "Rates"):
        return cls(
            trip_distance_km=distance,
            trip_duration_minutes=duration,
            per_km_rate=rates.per_km_rate,
            per_minute_rate=rates.per_minute_rate,
            base_fare=rates.base_fare,
        )

class PriceInput(BaseModel):
    distance_km: float
    duration_min: float
    weather: Optional[str] = None 
    traffic_conditions: Optional[str] = None 
    per_km_rate: Optional[float] = None 
    per_minute_rate: Optional[float] = None 
    base_fare: Optional[float] = None
    currency: Optional[str] = None 

class PredictionOutput(BaseModel):
    predicted_price: float
    per_km_rate: float
    base_fare: float
    used_feature_set: str
    currency: str
    

class CleanTaxiData:
    def __init__(self):
        self.df = pd.read_csv(CLEAN_TAXI_CSV_PATH)

    def to_json(self):
        return json.loads(self.df.to_json(orient="records"))
    
    def average_rates(self) -> Rates:
        return Rates(
            per_km_rate= round(float(self.df["per_km_rate"].mean()), 2),
            per_minute_rate= round(float(self.df["per_minute_rate"].mean()), 2),
            base_fare= round(float(self.df["base_fare"].mean()), 2)
        )
    
    def group_rates(self, weather: Optional[str] = None, traffic: Optional[str] = None) -> Optional[Rates]:
        g = self.df
        if weather is not None:
            g = g[g["weather"] == weather]
        if traffic is not None:
            g = g[g["traffic_conditions"] == traffic]
        if g.empty:
            return None
        return Rates(
            per_km_rate= round(float(g["per_km_rate"].mean()), 2),
            per_minute_rate= round(float(g["per_minute_rate"].mean()), 2),
            base_fare= round(float(g["base_fare"].mean()), 2),
        )

