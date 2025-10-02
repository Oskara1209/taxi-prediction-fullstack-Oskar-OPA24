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


data = read_api_endpoint("taxi")

df = pd.DataFrame(data.json())


def main():
    st.markdown("# Taxi Prediction Dashboard")

    st.dataframe(df)


if __name__ == "__main__":
    main()
