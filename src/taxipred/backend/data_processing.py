from taxipred.utils.constants import CLEAN_TAXI_CSV_PATH
import pandas as pd
import json


class CleanTaxiData:
    def __init__(self):
        self.df = pd.read_csv(CLEAN_TAXI_CSV_PATH)

    def to_json(self):
        return json.loads(self.df.to_json(orient="records"))
    
    def average_rates(self) -> dict:
        """Returnerar medelvärden för km/min/base ur datan"""
        return {
            "per_km_rate": round(self.df["fare_per_km"].mean(), 2),
            "per_minute_rate": round(self.df["fare_per_minute"].mean(), 2),
            "base_fare": round(self.df["fare_base"].mean(), 2),
        }
