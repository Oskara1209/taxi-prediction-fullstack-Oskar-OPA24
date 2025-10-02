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

@app.get("/geocode")
def geocode(q: str, limit: int = 5, locale: str = "sv"):
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
