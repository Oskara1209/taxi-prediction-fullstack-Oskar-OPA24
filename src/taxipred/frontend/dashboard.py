import streamlit as st
import pandas as pd
import requests
from streamlit_searchbox import st_searchbox
import folium
from streamlit_folium import st_folium

from taxipred.utils.helpers import read_api_endpoint

BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Taxi Prediction Dashboard", layout="wide")

# --- state för vald rutt (så vi kan rendera efter rerun) ---
if "route_payload" not in st.session_state:
    st.session_state.route_payload = None  # {"data":..., "start":..., "end":...}

# ---------- Backend helpers ----------
def geocode_suggest(q: str, limit: int = 5, locale: str = "sv"):
    if not q or len(q.strip()) < 2:
        return []
    try:
        r = requests.get(f"{BACKEND}/geocode", params={"q": q, "limit": limit, "locale": locale}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return []

def fetch_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, profile: str = "car"):
    params = {"start_lat": start_lat, "start_lon": start_lon, "end_lat": end_lat, "end_lon": end_lon, "profile": profile}
    r = requests.get(f"{BACKEND}/route/", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data.get("details") or data["error"])
    return data

# ---------- Searchbox adapters ----------
def suggest_labels_start(search: str):
    hits = geocode_suggest(search)
    return [(h.get("label", ""), h) for h in hits]

def suggest_labels_end(search: str):
    hits = geocode_suggest(search)
    return [(h.get("label", ""), h) for h in hits]

def main():
    st.markdown("# Taxi Prediction Dashboard")

    st.dataframe(df)


if __name__ == "__main__":
    main()
