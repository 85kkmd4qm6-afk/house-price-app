import math
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import pydeck as pdk
from datetime import date, timedelta

import os
import subprocess


if not os.path.exists("app.duckdb"):
    subprocess.run(["python", "build_db.py"], check=True)

DB_PATH = "app.duckdb"

import streamlit as st
st.write("DEPLOY CHECK: v4 - if you can read this, Streamlit is using my latest app.py")


# -----------------------
# Helpers
# -----------------------
@st.cache_resource
def get_con():
    return duckdb.connect(DB_PATH, read_only=True)

def normalize_postcode(pc: str) -> str:
    if not pc:
        return ""
    return pc.strip().upper().replace(" ", "")

def haversine_km(lat1, lon1, lat2, lon2):
    # Great-circle distance
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@st.cache_data
def get_distinct_postcodes(limit=2000):
    con = get_con()
    df = con.execute(f"""
        SELECT postcode, COUNT(*) AS n
        FROM sales
        WHERE postcode IS NOT NULL
        GROUP BY 1
        ORDER BY n DESC
        LIMIT {limit};
    """).df()
    return df["postcode"].dropna().tolist()

from datetime import date

@st.cache_data
def get_date_range():
    con = get_con()
    min_d, max_d = con.execute(
        "SELECT MIN(transfer_date), MAX(transfer_date) FROM sales;"
    ).fetchone()

    # Convert datetime → date if needed
    if hasattr(min_d, "date"):
        min_d = min_d.date()
    if hasattr(max_d, "date"):
        max_d = max_d.date()

    return min_d, max_d


@st.cache_data
def trend(postcode, property_types, start_d, end_d, freq):
    con = get_con()
    # freq: 'month' or 'quarter'
    if freq == "Month":
        bucket = "date_trunc('month', transfer_date)"
    else:
        bucket = "date_trunc('quarter', transfer_date)"

    pc = postcode.strip().upper()
    types_sql = ""
    params = [pc, start_d, end_d]

    if property_types and "All" not in property_types:
        types_sql = "AND property_type IN (" + ",".join(["?"] * len(property_types)) + ")"
        params.extend(property_types)

    q = f"""
        SELECT
            {bucket} AS period,
            COUNT(*) AS sales_count,
            AVG(price) AS avg_price,
            MEDIAN(price) AS median_price
        FROM sales
        WHERE postcode = ?
          AND transfer_date BETWEEN ? AND ?
          {types_sql}
        GROUP BY 1
        ORDER BY 1;
    """
    return con.execute(q, params).df()

@st.cache_data
def compare_postcodes(postcodes, property_types, start_d, end_d):
    con = get_con()
    params = [start_d, end_d] + postcodes

    types_sql = ""
    if property_types and "All" not in property_types:
        types_sql = "AND property_type IN (" + ",".join(["?"] * len(property_types)) + ")"
        params.extend(property_types)

    q = f"""
        SELECT
            postcode,
            COUNT(*) AS sales_count,
            AVG(price) AS avg_price,
            MEDIAN(price) AS median_price,
            MIN(transfer_date) AS first_sale,
            MAX(transfer_date) AS last_sale
        FROM sales
        WHERE transfer_date BETWEEN ? AND ?
          AND postcode IN ({",".join(["?"] * len(postcodes))})
          {types_sql}
        GROUP BY 1
        ORDER BY median_price DESC;
    """
    return con.execute(q, params).df()

@st.cache_data
def search_sales_history(postcode, street_contains, paon_contains, start_d, end_d):
    con = get_con()
    params = [postcode.strip().upper(), start_d, end_d]

    street_sql = ""
    paon_sql = ""
    if street_contains:
        street_sql = "AND LOWER(street) LIKE ?"
        params.append(f"%{street_contains.lower()}%")
    if paon_contains:
        paon_sql = "AND LOWER(paon) LIKE ?"
        params.append(f"%{paon_contains.lower()}%")

    q = f"""
        SELECT
            transfer_date,
            price,
            property_type,
            old_new,
            duration,
            paon, saon, street, locality, town_city, district, county,
            tx_id
        FROM sales
        WHERE postcode = ?
          AND transfer_date BETWEEN ? AND ?
          {street_sql}
          {paon_sql}
        ORDER BY transfer_date DESC
        LIMIT 200;
    """
    return con.execute(q, params).df()

@st.cache_data
def postcode_to_coord(postcode):
    con = get_con()
    # returns (lat, lon) or None
    pc_norm = normalize_postcode(postcode)
    # If postcodes table doesn't exist, this will error; handle gracefully
    try:
        row = con.execute("""
            SELECT lat, lon
            FROM postcodes
            WHERE postcode_nospace = ?
            LIMIT 1;
        """, [pc_norm]).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return float(row[0]), float(row[1])

@st.cache_data
def nearby_sales_by_postcode_centroid(center_postcode, radius_km, since_days, property_types):
    con = get_con()
    center = postcode_to_coord(center_postcode)
    if center is None:
        return pd.DataFrame(), None

    lat0, lon0 = center
    since_date = date.today() - timedelta(days=since_days)

    # Get recent sales and join to postcode coordinates (centroids)
    types_sql = ""
    params = [since_date]

    if property_types and "All" not in property_types:
        types_sql = "AND s.property_type IN (" + ",".join(["?"] * len(property_types)) + ")"
        params.extend(property_types)

    # If postcodes table isn't present, bail
    try:
        df = con.execute(f"""
            SELECT
                s.transfer_date,
                s.price,
                s.property_type,
                s.postcode,
                p.lat, p.lon,
                s.paon, s.street, s.town_city, s.district,
                s.tx_id
            FROM sales s
            JOIN postcodes p
              ON UPPER(REPLACE(TRIM(s.postcode), ' ', '')) = p.postcode_nospace
            WHERE s.transfer_date >= ?
              {types_sql}
            LIMIT 50000;
        """, params).df()
    except Exception:
        return pd.DataFrame(), center

    if df.empty:
        return df, center

    # Compute distance in Python (fine for MVP; can optimize later)
    dists = []
    for r in df.itertuples(index=False):
        d = haversine_km(lat0, lon0, r.lat, r.lon)
        dists.append(d)

    df["distance_km"] = dists
    df = df[df["distance_km"] <= radius_km].copy()
    df.sort_values(["transfer_date", "distance_km"], ascending=[False, True], inplace=True)
    df = df.head(300)
    return df, center

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="UK House Price Explorer", layout="wide")

st.title("UK House Price Explorer (Price Paid Data)")
st.caption("Built with Streamlit + DuckDB. Data: HM Land Registry Price Paid Data; optional ONSPD for postcode centroids.")

min_d, max_d = get_date_range()

with st.sidebar:
    st.header("Global filters")
    start_d = st.date_input("From", value=max(min_d, date(max_d.year - 5, 1, 1)))
    end_d = st.date_input("To", value=max_d)
    if start_d > end_d:
        st.error("From date must be <= To date.")

    property_types = st.multiselect(
        "Property type",
        options=["All", "D", "S", "T", "F", "O"],
        default=["All"],
        help="D=Detached, S=Semi, T=Terraced, F=Flat, O=Other (PPD codes)."
    )

tabs = st.tabs(["Trends", "Compare", "Sales history", "Nearby sales"])

# ---- Trends ----
with tabs[0]:
    st.subheader("Trends for a postcode")
    col1, col2, col3 = st.columns([2, 1, 1])

    popular_postcodes = get_distinct_postcodes(limit=3000)

    with col1:
        pc = st.selectbox("Postcode", options=popular_postcodes, index=0)
    with col2:
        freq = st.selectbox("Time bucket", ["Month", "Quarter"], index=1)
    with col3:
        metric = st.selectbox("Metric", ["Median price", "Average price", "Sales volume"], index=0)

    df = trend(pc, property_types, start_d, end_d, freq)
    if df.empty:
        st.info("No data found for that selection.")
    else:
        if metric == "Sales volume":
            fig = px.bar(df, x="period", y="sales_count")
        elif metric == "Average price":
            fig = px.line(df, x="period", y="avg_price", markers=True)
        else:
            fig = px.line(df, x="period", y="median_price", markers=True)

        st.plotly_chart(fig, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Sales", f"{int(df['sales_count'].sum()):,}")
        k2.metric("Latest median", f"£{int(df['median_price'].dropna().iloc[-1]):,}" if df["median_price"].notna().any() else "—")
        k3.metric("Latest average", f"£{int(df['avg_price'].dropna().iloc[-1]):,}" if df["avg_price"].notna().any() else "—")

        st.dataframe(df, use_container_width=True)

# ---- Compare ----
with tabs[1]:
    st.subheader("Compare postcodes")
    st.caption("Select two postcodes to compare their stats over the chosen date range and property type filter.")

    sel = st.multiselect("Choose postcodes", options=get_distinct_postcodes(limit=5000), default=get_distinct_postcodes(limit=5)[:2])
    if len(sel) < 2:
        st.info("Pick at least two postcodes.")
    else:
        cmp = compare_postcodes(sel[:6], property_types, start_d, end_d)  # cap to 6 to keep charts readable
        st.dataframe(cmp, use_container_width=True)

        # Chart: median by postcode
        fig = px.bar(cmp, x="postcode", y="median_price")
        st.plotly_chart(fig, use_container_width=True)

# ---- Sales history ----
with tabs[2]:
    st.subheader("Sales history lookup")
    st.caption("Search within a postcode. Use street/PAON contains filters to narrow down.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        hist_pc = st.selectbox("Postcode", options=get_distinct_postcodes(limit=5000), index=0, key="hist_pc")
    with c2:
        street_contains = st.text_input("Street contains (optional)", value="")
    with c3:
        paon_contains = st.text_input("Building no/name contains (optional)", value="")

    hist = search_sales_history(hist_pc, street_contains.strip(), paon_contains.strip(), start_d, end_d)
    if hist.empty:
        st.info("No matches. Try removing filters or expanding the date range.")
    else:
        st.dataframe(hist, use_container_width=True)

        # Quick chart: prices over time
        fig = px.scatter(hist, x="transfer_date", y="price", hover_data=["paon", "street", "property_type"])
        st.plotly_chart(fig, use_container_width=True)

# ---- Nearby sales ----
with tabs[3]:
    st.subheader("Nearby sales (postcode centroid proximity)")
    st.caption("Requires ONSPD loaded. Uses postcode centroid distances (approximate).")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        near_pc = st.selectbox("Center postcode", options=get_distinct_postcodes(limit=5000), index=0, key="near_pc")
    with c2:
        radius_km = st.slider("Radius (km)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    with c3:
        since_days = st.slider("Recent (days)", min_value=30, max_value=3650, value=365, step=30)

    df_near, center = nearby_sales_by_postcode_centroid(near_pc, radius_km, since_days, property_types)

    if center is None:
        st.error("No postcode coordinates found (is ONSPD loaded into data/onspd.csv and build_db.py re-run?).")
    elif df_near.empty:
        st.info("No nearby sales matched your filters.")
    else:
        st.dataframe(df_near, use_container_width=True)

        # Map
        lat0, lon0 = center
        map_df = df_near[["lat", "lon", "price", "transfer_date", "postcode", "property_type", "distance_km"]].copy()

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_radius=40,
            pickable=True
        )

        center_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": lat0, "lon": lon0, "label": "Center"}]),
            get_position=["lon", "lat"],
            get_radius=90,
            pickable=True
        )

        view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=11)
        deck = pdk.Deck(layers=[layer, center_layer], initial_view_state=view_state, tooltip={"text": "{postcode}\n£{price}\n{transfer_date}\n{property_type}\n{distance_km} km"})
        st.pydeck_chart(deck)
