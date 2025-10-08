from fastapi import FastAPI, HTTPException, Query
from taxipred.backend.data_processing import CleanTaxiData
from dotenv import load_dotenv 
import os 
import requests

load_dotenv()

app = FastAPI()

taxi_data = CleanTaxiData()

API_KEY = os.getenv("GRAPHOPPER_API_KEY") 
if not API_KEY:
    raise RuntimeError("GRAPHOPPER_API_KEY saknas")

BASE = "https://graphhopper.com/api/1"
TIMEOUT = 15

FX = {"USD": 1.00, "SEK": 11.00, "NOK": 11.50, "EUR": 0.92}
SUPPORTED_CURRENCIES = set(FX.keys())

def load_model():
    with as_file(MODELS_PATH) as model_path:
        return joblib.load(model_path)

@app.get("/taxi/")
async def read_taxi_data():
    return taxi_data.to_json()

@app.get("/schema")
async def data_schema():
    df = taxi_data.df
    return {"columns": list(df.columns)}

@app.get("/geocode")
async def geocode(q: str, limit: int = 5, locale: str = "sv"):
    url = f"{BASE}/geocode"
    params = {"q": q, "limit": limit, "locale": locale, "key": API_KEY}
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    hits = r.json().get("hits", [])

    def build_label(h: dict) -> str:

        name        = h.get("name") or ""
        street      = h.get("street") or ""
        housenr     = h.get("housenumber") or ""
        postcode    = h.get("postcode") or ""
        city        = h.get("city") or h.get("locality") or ""
        
        admin_level = h.get("state") or h.get("county") or ""
        country     = h.get("country") or ""

        primary = (f"{street} {housenr}".strip()) if (street or housenr) else name

       
        line2_parts = []
        if postcode and city:
            line2_parts.append(f"{postcode} {city}".strip())
        elif city:
            line2_parts.append(city)
        if admin_level:
            line2_parts.append(admin_level)
        if country:
            line2_parts.append(country)

        line2 = ", ".join(p for p in line2_parts if p)
        if primary and line2:
            return f"{primary}, {line2}"
        return primary or line2 or name or country

    results = []
    for h in hits:
        point = h.get("point") or {}
        lat = point.get("lat")
        lon = point.get("lng")
        if lat is None or lon is None:
            continue
        results.append({
            "lat": lat,
            "lon": lon,
            "label": build_label(h),
        })
    return results

@app.get("/route/")
async def get_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, profile: str = "car"):
    url = f"{BASE}/route"
    params = {
        "point": [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
        "profile": profile,
        "locale": "sv",
        "points_encoded": "false",
        "instructions": "false",
        "key": API_KEY,
    }
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data.get("paths"):
        raise HTTPException(404, detail="Ingen rutt hittades")

    path = data["paths"][0]
    return {
        "distance_km": round(path["distance"] / 1000, 2),
        "duration_min": round(path["time"] / 60000, 1),
        "points": path.get("points"),   # <<— LineString med coordinates = [[lon,lat,(ev. ele)], ...]
        "bbox": path.get("bbox"),       # [min_lon, min_lat, max_lon, max_lat]
    }