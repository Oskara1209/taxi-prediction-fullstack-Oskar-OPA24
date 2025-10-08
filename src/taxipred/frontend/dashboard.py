import streamlit as st
import pandas as pd
import requests
from streamlit_searchbox import st_searchbox
import folium
from streamlit_folium import st_folium
from typing import List
from typing import List, Optional
import altair as alt
import numpy as np

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
    
def render_route_map(route_payload: dict):
    data = route_payload["data"]; s = route_payload.get("start"); e = route_payload.get("end")
    cols = st.columns(2)
    cols[0].metric("Distans", f"{data['distance_km']} km")
    cols[1].metric("Tid", f"{data['duration_min']} min")

    coords = (data.get("points") or {}).get("coordinates", [])
    route_latlon = [(lat, lon) for lon, lat, *_ in coords]
    if not route_latlon:
        st.warning("Kunde inte rita rutt – saknar points i svaret.")
        return

    m = folium.Map(location=route_latlon[0], zoom_start=12)
    folium.PolyLine(route_latlon, weight=5, opacity=0.9).add_to(m)
    folium.Marker(route_latlon[0], tooltip=f"Start: {s['label']}" if s else "Start").add_to(m)
    folium.Marker(route_latlon[-1], tooltip=f"Mål: {e['label']}" if e else "Mål").add_to(m)
    if data.get("bbox"):
        min_lon, min_lat, max_lon, max_lat = data["bbox"]
        m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
    st_folium(m, width=900, height=520)

def build_predict_payload(dist: float, dur: float, currency: str,tariff_mode: str, weather: Optional[str], traffic: Optional[str],custom_rates: dict) -> dict:
    """Skapa payload mot /api/predict – features skickas alltid in när vi har dem."""
    payload = {
        "distance_km": dist,
        "duration_min": dur,
        "currency": currency,
    }
    if tariff_mode == PER_CONDITIONS:
        payload.update({"weather": weather, "traffic_conditions": traffic})
    elif tariff_mode == CUSTOM:
        payload.update(custom_rates)
        payload.update({"weather": weather, "traffic_conditions": traffic})
    else:
        pass
    return payload

def render_price_metrics(quote: dict, symbol: str):
    total = float(quote.get("predicted_price", 0.0))
    base  = float(quote.get("base_fare", 0.0))
    per_km = float(quote.get("per_km_rate", 0.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Totalpris", f"{total:.2f} {symbol}")
    c2.metric("Startkostnad", f"{base:.2f} {symbol}")
    c3.metric("Pris per km", f"{per_km:.2f} {symbol}/km")
    st.caption(f"Featureset: {quote.get('used_feature_set','?')}")


def whatif_panel(dist: float, dur: float, currency: str,
                 weather_opt: list, traffic_opt: list):
    with st.expander("Pris för alla väder och trafikkombinationer"):
        with st.spinner("Beräknar scenarier..."):
            scenarios = [{  
                "distance_km": dist,
                "duration_min": dur,
                "weather": w,
                "traffic_conditions": t,
                "currency": currency,
            } for w in weather_opt for t in traffic_opt]

            try:
                resp = post_predict_batch(scenarios)
            except requests.RequestException:
                resp = [post_predict(p) for p in scenarios]

            rows = [{"weather": s["weather"], "traffic": s["traffic_conditions"], "price": r["predicted_price"]}
                    for s, r in zip(scenarios, resp)]
            df = pd.DataFrame(rows)


        # Gemensamma encodings (snygg sortering + tooltip)
        x_weather = alt.X('weather:N', sort=weather_opt, title='Weather')
        t_traffic = alt.Color('traffic:N', sort=traffic_opt, title='Traffic')
        tip = [alt.Tooltip('weather:N', title='Weather'),
               alt.Tooltip('traffic:N', title='Traffic'),
               alt.Tooltip('price:Q', title='Pris')]

        chart = (alt.Chart(df).mark_bar().encode(
                           x=x_weather,
                           y=alt.Y('price:Q', title=f'Pris ({currency})'),
                           color=t_traffic,
                           xOffset='traffic:N',  
                           tooltip=tip
                       ))
        st.altair_chart(chart, use_container_width=True)



        with st.expander("Tabell"):
            pvt = (df.pivot(index="weather", columns="traffic", values="price")
                     .reindex(index=weather_opt, columns=traffic_opt))
            st.dataframe(pvt, use_container_width=True)


def main():
    st.markdown("# Taxi Prediction Dashboard")
    tab_data, tab_route, tab_statistics = st.tabs(["Data (historik)", "Rutt (Beräkna din resa)", "Statistik"])

    # ---- Flik 1: Data ----
    with tab_data:
        df = load_taxi_df()
        st.dataframe(df, use_container_width=True)

    with tab_statistics:
        st.subheader("Diagram")
        df = load_taxi_df()
        needed = {"trip_price","trip_distance_km","trip_duration_minutes",
              "per_km_rate","per_minute_rate","base_fare","weather","traffic_conditions"}
        miss = needed - set(df.columns)
        if miss:
            st.warning(f"Saknar kolumner: {sorted(miss)}")
            st.stop()
        df = df.dropna(subset=["trip_price","trip_distance_km","trip_duration_minutes"])

        # Härleder extra nyckeltal
        df["price_per_km"] = df["trip_price"] / df["trip_distance_km"].replace(0, np.nan)
        df["formula_price"] = (df["base_fare"]
                           + df["per_km_rate"]*df["trip_distance_km"]
                           + df["per_minute_rate"]*df["trip_duration_minutes"])
        df["residual_formula"] = df["trip_price"] - df["formula_price"]

        # --- Små filter för interaktivitet ---
        c1, c2, c3 = st.columns(3)
        with c1:
            w_sel = st.multiselect("Weather", sorted(df["weather"].dropna().unique().tolist()),
                               default=sorted(df["weather"].dropna().unique().tolist()))
        with c2:
            t_sel = st.multiselect("Traffic", sorted(df["traffic_conditions"].dropna().unique().tolist()),
                               default=sorted(df["traffic_conditions"].dropna().unique().tolist()))
        with c3:
            maxbins = st.slider("Antal bins (hist)", 10, 60, 30)

        mask = df["weather"].isin(w_sel) & df["traffic_conditions"].isin(t_sel)
        dff = df.loc[mask].copy()
        if dff.empty:
            st.info("Inga rader matchar filtret.")
            st.stop()

        st.markdown("### 1) Prisfördelning")
        hist_price = (
        alt.Chart(dff).mark_bar().encode(
            x=alt.X("trip_price:Q", bin=alt.Bin(maxbins=maxbins), title="Pris (USD)"),
            y=alt.Y("count()", title="Antal"),
            tooltip=[alt.Tooltip("count()", title="Antal")]
            ).properties(height=260)
        )
        st.altair_chart(hist_price, use_container_width=True)

        st.markdown("### 2) Pris vs distans (med trend)")
        pts = (alt.Chart(dff).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X("trip_distance_km:Q", title="Distans (km)"),
            y=alt.Y("trip_price:Q", title="Pris (USD)"),
            color=alt.Color("traffic_conditions:N", title="Traffic"),
            tooltip=["trip_distance_km","trip_duration_minutes","trip_price","weather","traffic_conditions"]
            )
        )
        trend = (alt.Chart(dff).transform_regression("trip_distance_km", "trip_price").mark_line().encode(x="trip_distance_km:Q", y="trip_price:Q"))
        st.altair_chart(pts + trend, use_container_width=True)

        st.markdown("### 3) Medelpris per väder (serier = traffic)")
        bar_grp = (
        alt.Chart(dff).mark_bar().encode(
            x=alt.X("weather:N", title="Weather", sort=sorted(dff["weather"].unique().tolist())),
            y=alt.Y("mean(trip_price):Q", title="Medelpris (USD)"),
            color=alt.Color("traffic_conditions:N", title="Traffic",
                            sort=sorted(dff["traffic_conditions"].unique().tolist())),
            xOffset="traffic_conditions:N",
            tooltip=["weather","traffic_conditions", alt.Tooltip("mean(trip_price):Q", title="Medelpris")]
            )
        )
        st.altair_chart(bar_grp, use_container_width=True)

        st.markdown("### 4) Heatmap: medelpris per (väder × traffic)")
        heat = (alt.Chart(dff).mark_rect().encode(
            x=alt.X("traffic_conditions:N", title="Traffic",
                    sort=sorted(dff["traffic_conditions"].unique().tolist())),
            y=alt.Y("weather:N", title="Weather",
                    sort=sorted(dff["weather"].unique().tolist())),
            color=alt.Color("mean(trip_price):Q", title="Medelpris (USD)"),
            tooltip=["weather","traffic_conditions", alt.Tooltip("mean(trip_price):Q", title="Medelpris")]
            ).properties(height=220)
        )
        text = (alt.Chart(dff).mark_text(baseline="middle").encode(
            x="traffic_conditions:N",
            y="weather:N",
            text=alt.Text("mean(trip_price):Q", format=".1f")
            )
        )
        st.altair_chart(heat + text, use_container_width=True)

        st.markdown("### 5) Boxplot: Pris per trafiknivå")
        box = (alt.Chart(dff).mark_boxplot().encode(
            x=alt.X("traffic_conditions:N", title="Traffic",
                    sort=sorted(dff["traffic_conditions"].unique().tolist())),
            y=alt.Y("trip_price:Q", title="Pris (USD)"),
            color=alt.Color("traffic_conditions:N", legend=None)
            ).properties(height=280)
        )
        st.altair_chart(box, use_container_width=True)

        st.markdown("### 6) Pris per km (histogram)")
        hist_ppk = (
            alt.Chart(dff.replace([np.inf, -np.inf], np.nan).dropna(subset=["price_per_km"]))
            .mark_bar()
            .encode(
                x=alt.X("price_per_km:Q", bin=alt.Bin(maxbins=maxbins), title="Pris per km (USD/km)"),
                y="count()",
                tooltip=[alt.Tooltip("count()", title="Antal")]
            ).properties(height=260)
            )
        st.altair_chart(hist_ppk, use_container_width=True)

        st.markdown("### 7) Residual mot enkel formel (pris – formel)")
        c1, c2 = st.columns(2)
        with c1:
            resid_hist = (
                alt.Chart(dff)
                .mark_bar()
                .encode(
                    x=alt.X("residual_formula:Q", bin=alt.Bin(maxbins=maxbins), title="Residual (USD)"),
                    y="count()",
                    tooltip=[alt.Tooltip("count()", title="Antal")]
                ).properties(height=260)
            )
            st.altair_chart(resid_hist, use_container_width=True)
        with c2:
            resid_scatter = (
                alt.Chart(dff)
                .mark_circle(size=60, opacity=0.5)
                .encode(
                    x=alt.X("trip_distance_km:Q", title="Distans (km)"),
                    y=alt.Y("residual_formula:Q", title="Residual (USD)"),
                    color="weather:N",
                    tooltip=["trip_distance_km","trip_price","formula_price","residual_formula","weather","traffic_conditions"]
                )
            )
            st.altair_chart(resid_scatter, use_container_width=True)

    # ---- Flik 2: Rutt & pris ----
    with tab_route:
        st.subheader("Beräkna distans & tid")
        mode = st.radio("Inmatning", ["Adresser", "Koordinater"], horizontal=True)

        # Inmatning + hämta rutt
        if mode == "Adresser":
            block = adress_autocomplete_block()
            if st.button("Hämta distans & tid", type="primary", disabled=(block is None), key="btn_addr"):
                try:
                    data = fetch_route(block["start"]["lat"], block["start"]["lon"],
                                       block["end"]["lat"], block["end"]["lon"], profile=block["profile"])
                    st.session_state[SS_ROUTE] = {"data": data, "start": block["start"], "end": block["end"]}
                    st.session_state[SS_QUOTE] = None
                except Exception as ex:
                    st.error(f"Något gick fel: {ex}")
        else:
            coords = coords_input_block()
            if st.button("Hämta distans & tid", type="primary", key="btn_coords"):
                try:
                    data = fetch_route(coords["start_lat"], coords["start_lon"],
                                       coords["end_lat"], coords["end_lon"], profile=coords["profile"])
                    st.session_state[SS_ROUTE] = {"data": data, "start": None, "end": None}
                    st.session_state[SS_QUOTE] = None
                except Exception as ex:
                    st.error(f"Något gick fel: {ex}")

        # Rendera rutt + pris
        route_payload = st.session_state[SS_ROUTE]
        if route_payload:
            render_route_map(route_payload)

            st.divider()
            st.subheader("Pris för resan")

            dist = route_payload["data"]["distance_km"]
            dur  = route_payload["data"]["duration_min"]

            # meta 
            try:
                meta = fetch_ml_meta()
                weather_opt = meta.get("weather_values", [])
                traffic_opt = meta.get("traffic_values", [])
            except requests.RequestException:
                weather_opt = ["Clear", "Rain", "Snow"]
                traffic_opt = ["Low", "Medium", "High"]

            # 1) Tariff först
            tariff_mode, sel_weather, sel_traffic, custom_rates = tariff_controls(weather_opt, traffic_opt)

            # 2) Valuta
            currency_choice = currency_selector()

            # 3) Beräkna pris
            cols = st.columns([1, 1, 4])
            with cols[0]:
                if st.button("Beräkna pris", key="btn_price"):
                    try:
                        payload = build_predict_payload(dist, dur, currency_choice,
                                                        tariff_mode, sel_weather, sel_traffic, custom_rates)
                        quote = post_predict(payload)
                        st.session_state[SS_QUOTE] = quote
                    except requests.RequestException as ex:
                        st.error(f"Kunde inte hämta pris: {ex}")

            with cols[1]:
                st.button("Rensa pris", on_click=lambda: st.session_state.update({SS_QUOTE: None}))

            # 4) Visa pris
            quote = st.session_state[SS_QUOTE]
            if quote:
                sym = CURRENCY_SYMBOL.get(quote.get("currency", "USD"), "USD")
                render_price_metrics(quote, sym)
                # What-if
                whatif_panel(dist, dur, currency_choice, weather_opt, traffic_opt)

            st.divider()
            st.button("Rensa ruta", on_click=lambda: st.session_state.update({SS_ROUTE: None}))


if __name__ == "__main__":
    main()