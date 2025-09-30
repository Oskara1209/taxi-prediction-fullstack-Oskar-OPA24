from fastapi import FastAPI
from taxipred.backend.data_processing import CleanTaxiData

app = FastAPI()

taxi_data = CleanTaxiData()

@app.get("/taxi/")
async def read_taxi_data():
    return taxi_data.to_json()
