from fastapi import FastAPI, HTTPException, Query
from taxipred.backend.data_processing import CleanTaxiData
from dotenv import load_dotenv 
import os 
import requests

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("GRAPHOPPER_API_KEY") 
if not API_KEY:
    raise RuntimeError("GRAPHOPPER_API_KEY saknas")

BASE = "https://graphhopper.com/api/1"
TIMEOUT = 15

taxi_data = CleanTaxiData()

@app.get("/taxi/")
async def read_taxi_data():
    return taxi_data.to_json()
