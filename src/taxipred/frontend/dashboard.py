import streamlit as st
import pandas as pd
import requests
from streamlit_searchbox import st_searchbox
import folium
from streamlit_folium import st_folium

from taxipred.utils.helpers import read_api_endpoint

BACKEND = "http://localhost:8000"
st.set_page_config(page_title="Taxi Prediction Dashboard", layout="wide")

CURRENCY_SYMBOL = {"USD": "$", "SEK": "kr", "NOK": "kr", "EUR": "€"}

GLOBAL = "Global(medel)"
PER_CONDITIONS = "Per väder/trafik"
CUSTOM = "Egen"

SS_ROUTE = "route_payload"
SS_QUOTE = "price_quote"

# ---------- Init state ----------
if SS_ROUTE not in st.session_state:
    st.session_state[SS_ROUTE]= None  
if SS_QUOTE not in st.session_state:
    st.session_state[SS_QUOTE] = None 



# ---------- Backend helpers ----------

def load_taxi_df():
    resp = read_api_endpoint("taxi")
    df = pd.DataFrame(resp.json())
    return df

def try_get(url: str, **params):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def try_post(url: str, json_body: dict | list):
    r = requests.post(url, json=json_body, timeout=30)
    r.raise_for_status()
    return r.json()

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

def fetch_ml_meta():
    r = requests.get(f"{BACKEND}/ml/meta", timeout=10)
    r.raise_for_status()
    return r.json()

def post_predict(payload: dict):
    r = requests.post(f"{BACKEND}/api/predict", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def post_predict_batch(payloads: List[dict]):
    r = requests.post(f"{BACKEND}/api/predict_batch", json=payloads, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------- UI-komponenter ----------

def adress_autocomplete_block() -> Optional[dict]:
        """Returnerar {"start": {...}, "end": {...}, "profile": str} eller None om inte redo."""
        c1, c2 = st.columns(2)

        with c1:
            st.write("Startadress")
            start = st_searchbox(
                search_function=lambda s: [(h.get("label",""), h) for h in geocode_suggest(s)],
                placeholder="Skriv t.ex. 'Sixten Camps Gata...'",
                clear_on_submit=False,
                key="sb_start",
            )
        with c2:
            st.write("Måladress")
            end = st_searchbox(
                search_function=lambda s: [(h.get("label",""), h) for h in geocode_suggest(s)],
                placeholder="Skriv t.ex. 'Persvägen 12A...'",
                clear_on_submit=False,
                key="sb_end",
            )
        profile = st.selectbox("Profil", ["car", "bike", "foot"], index=0)

        if not (start and end):
            st.info("Sök och välj både start och mål i fälten ovan för att fortsätta.")
            return None
        return {"start": start, "end": end, "profile": profile}

def coords_input_block() -> dict:
    c1, c2 = st.columns(2)
    with c1:
        start_lat = st.number_input("Start lat", value=59.330000, format="%.6f")
        start_lon = st.number_input("Start lon", value=18.058000, format="%.6f")
    with c2:
        end_lat = st.number_input("Mål lat", value=59.858000, format="%.6f")
        end_lon = st.number_input("Mål lon", value=17.645000, format="%.6f")
    profile = st.selectbox("Profil", ["car", "bike", "foot"], index=0, key="profile_coords")
    return {"start_lat": start_lat, "start_lon": start_lon, "end_lat": end_lat, "end_lon": end_lon, "profile": profile}

def tariff_controls(weather_opt: list, traffic_opt: list) -> tuple[str, Optional[str], Optional[str], dict]:
    """Returnerar (tariff_mode, weather, traffic, custom_rates) där vissa kan vara None/tomma."""
    tariff_mode = st.radio("Tariffkälla", [GLOBAL, PER_CONDITIONS, CUSTOM], horizontal=True)
    sel_weather, sel_traffic = None, None
    custom_rates: dict = {}

    if tariff_mode == PER_CONDITIONS:
        col_w, col_t = st.columns(2)
        with col_w:
            sel_weather = st.selectbox("Weather", weather_opt or ["Clear"], key="sel_weather")
        with col_t:
            default_index = traffic_opt.index("Medium") if "Medium" in traffic_opt else 0
            sel_traffic = st.selectbox("Traffic", traffic_opt or ["Medium"], index=default_index, key="sel_traffic")

    if tariff_mode == CUSTOM:
        c1, c2, c3 = st.columns(3)
        custom_rates["base_fare"]       = c1.number_input("Startkostnad", min_value=0.0, value=50.0, step=1.0)
        custom_rates["per_km_rate"]     = c2.number_input("Pris per km", min_value=0.0, value=12.0, step=0.5)
        custom_rates["per_minute_rate"] = c3.number_input("Pris per minut", min_value=0.0, value=3.0, step=0.5)

    return tariff_mode, sel_weather, sel_traffic, custom_rates


def currency_selector() -> str:
    col_cur, _ = st.columns([1,3])
    with col_cur:
        return st.selectbox("Valuta", ["SEK", "NOK", "EUR", "USD"], index=0, key="sel_currency")

    # Karta
    coords = (data.get("points") or {}).get("coordinates", [])
    route_latlon = [(lat, lon) for lon, lat, *rest in coords]
    if route_latlon:
        m = folium.Map(location=route_latlon[0], zoom_start=12)
        folium.PolyLine(route_latlon, weight=5, opacity=0.9).add_to(m)
        folium.Marker(route_latlon[0], tooltip=f"Start: {s['label']}" if s else "Start").add_to(m)
        folium.Marker(route_latlon[-1], tooltip=f"Mål: {e['label']}" if e else "Mål").add_to(m)
        if data.get("bbox"):
            min_lon, min_lat, max_lon, max_lat = data["bbox"]
            m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
        st_folium(m, width=900, height=520)
    else:
        st.warning("Kunde inte rita rutt – saknar points i svaret.")

def main():
    st.markdown("# Taxi Prediction Dashboard")

    tab1, tab2 = st.tabs(["Data (historik)", "Rutt (Beräkna din resa)"])

    with tab1:
        data_resp = read_api_endpoint("taxi")
        df = pd.DataFrame(data_resp.json())
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Beräkna distans & tid")
        mode = st.radio("Inmatning", ["Adresser (autocomplete)", "Koordinater"], horizontal=True)

        if mode == "Adresser (autocomplete)":
            c1, c2 = st.columns(2)
            with c1:
                st.write("Startadress")
                start_selected = st_searchbox(
                    search_function=suggest_labels_start,
                    placeholder="Skriv t.ex. 'Sixten Camps Gata...'",
                    clear_on_submit=False,
                    key="sb_start",
                )
            with c2:
                st.write("Måladress")
                end_selected = st_searchbox(
                    search_function=suggest_labels_end,
                    placeholder="Skriv t.ex. 'Persvägen 12A...'",
                    clear_on_submit=False,
                    key="sb_end",
                )

            profile = st.selectbox("Profil", ["car", "bike", "foot"], index=0)
            btn_disabled = not (start_selected and end_selected)

            if st.button("Hämta distans & tid", type="primary", disabled=btn_disabled, key="btn_addr"):
                try:
                    with st.spinner("Hämtar rutt..."):
                        data = fetch_route(start_selected["lat"], start_selected["lon"],
                                           end_selected["lat"], end_selected["lon"], profile=profile)
                    # SPARA i session_state → rendera nedan varje rerun
                    st.session_state.route_payload = {"data": data, "start": start_selected, "end": end_selected}
                except Exception as ex:
                    st.error(f"Något gick fel: {ex}")

            if btn_disabled:
                st.info("Sök och välj både start och mål i fälten ovan för att fortsätta.")

        else:
            c1, c2 = st.columns(2)
            with c1:
                start_lat = st.number_input("Start lat", value=59.330000, format="%.6f")
                start_lon = st.number_input("Start lon", value=18.058000, format="%.6f")
            with c2:
                end_lat = st.number_input("Mål lat", value=59.858000, format="%.6f")
                end_lon = st.number_input("Mål lon", value=17.645000, format="%.6f")

            profile = st.selectbox("Profil", ["car", "bike", "foot"], index=0, key="profile_coords")

            if st.button("Hämta distans & tid", type="primary", key="btn_coords"):
                try:
                    with st.spinner("Hämtar rutt..."):
                        data = fetch_route(start_lat, start_lon, end_lat, end_lon, profile=profile)
                    st.session_state.route_payload = {"data": data, "start": None, "end": None}
                except Exception as ex:
                    st.error(f"Något gick fel: {ex}")

        # ---- Rendera rutt (oavsett läge) om vi har en sparad payload ----
        if st.session_state.route_payload:
            render_route_block(st.session_state.route_payload)
            st.button("Rensa ruta", on_click=lambda: st.session_state.update(route_payload=None))

        

if __name__ == "__main__":
    main()
