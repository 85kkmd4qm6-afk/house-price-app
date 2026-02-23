# app.py — UK House Price Explorer (Streamlit + DuckDB)
# Works locally and on Streamlit Community Cloud.

import os
import sys
import subprocess
import time
from datetime import date
import re

import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    import pydeck as pdk
on except Exception:
    pdk = None  # map tab will degrade gracefully

DB_PATH = "app.duckdb"
LOCK_PATH = ".building_db.lock"


def ensure_db_exists():
    """Build app.duckdb if missing (cloud-safe, avoids repeated messages)."""
    if os.path.exists(DB_PATH):
        return

    # If another run is building the DB, wait rather than starting again.
    if os.path.exists(LOCK_PATH):
        with st.spinner("Database is currently being built… please wait"):
            for _ in range(120):  # wait up to ~120 seconds
                time.sleep(1)
                if os.path.exists(DB_PATH):
                    return
        st.error("Database build is taking too long. Reboot the app in Manage app.")
        st.stop()

    # Create lock so only one run builds.
    with open(LOCK_PATH, "w") as f:
        f.write("building")

    status_box = st.empty()
    try:
        with status_box.status("First run: building database from CSV…", expanded=False):
            result = subprocess.run(
                [sys.executable, "build_db.py"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                st.error("build_db.py failed. Output:")
                st.code((result.stdout or "") + "\n" + (result.stderr or ""))
                st.stop()

            if not os.path.exists(DB_PATH):
                st.error("Database build finished but app.duckdb was not created.")
                st.stop()
    finally:
        # Remove lock whether build succeeded or failed
        try:
            os.remove(LOCK_PATH)
        except FileNotFoundError:
            pass

        # Clear the status box so it doesn't remain on the page
        status_box.empty()


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection:
    ensure_db_exists()
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data
def table_exists(table_name: str) -> bool:
    con = get_con()
    q = """
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_schema='main' AND table_name = ?;
    """
    return con.execute(q, [table_name]).fetchone()[0] > 0


@st.cache_data
def get_date_range():
    con = get_con()
    min_d, max_d = con.execute(
        "SELECT MIN(transfer_date), MAX(transfer_date) FROM sales;"
    ).fetchone()

    # DuckDB might return date or datetime; normalize to datetime.date
    if hasattr(min_d, "date"):
        min_d = min_d.date()
    if hasattr(max_d, "date"):
        max_d = max_d.date()

    return min_d, max_d


@st.cache_data
def get_distinct_postcode_areas(limit: int = 2000) -> pd.DataFrame:
    """
    Returns a list of postcode districts (e.g., SW1A, M1) to help users pick.
    We derive "area" by taking the outward code (up to first space), then removing trailing digits+letters.
    This is just to populate selection lists quickly; filters still accept full postcodes.
    """
    con = get_con()
    df = con.execute(
        """
        SELECT
          REGEXP_EXTRACT(UPPER(TRIM(postcode)), '^[A-Z]{1,2}[0-9][0-9A-Z]?', 0) AS area,
          COUNT(*) AS n_sales
        FROM sales
        WHERE postcode IS NOT NULL
        GROUP BY 1
        ORDER BY n_sales DESC
        LIMIT ?;
        """,
        [limit],
    ).df()
    df = df[df["area"].notna()]
    return df


@st.cache_data
def get_property_types() -> list[str]:
    con = get_con()
    rows = con.execute(
        """
        SELECT DISTINCT property_type
        FROM sales
        WHERE property_type IS NOT NULL AND property_type <> ''
        ORDER BY property_type;
        """
    ).fetchall()
    # PPD types: D,S,T,F,O; we keep raw and also show labels in UI
    return [r[0] for r in rows]


def prop_label(code: str) -> str:
    return {
        "D": "Detached (D)",
        "S": "Semi-detached (S)",
        "T": "Terraced (T)",
        "F": "Flat/Maisonette (F)",
        "O": "Other (O)",
    }.get(code, f"{code}")


def to_sql_date(d: date) -> str:
    return d.isoformat()


# -----------------------
# Query functions
# -----------------------
@st.cache_data
def query_trend(postcode_prefix: str, prop_type: str, start_d: date, end_d: date, freq: str) -> pd.DataFrame:
    """
    postcode_prefix: can be 'SW1A' or 'M1' or full postcode, used with LIKE
    prop_type: one of D/S/T/F/O or '' for all
    freq: 'month' or 'quarter' or 'year'
    """
    con = get_con()

    pc = postcode_prefix.strip().upper()
    # For prefixes, we match starting characters.
    like_pattern = pc + "%"

    # choose time bucket
    bucket_expr = {
        "month": "DATE_TRUNC('month', transfer_date)",
        "quarter": "DATE_TRUNC('quarter', transfer_date)",
        "year": "DATE_TRUNC('year', transfer_date)",
    }[freq]

    params = [to_sql_date(start_d), to_sql_date(end_d), like_pattern]
    where = """
        transfer_date BETWEEN ?::DATE AND ?::DATE
        AND UPPER(REPLACE(postcode, ' ', '')) LIKE UPPER(REPLACE(?, ' ', ''))
    """

    if prop_type:
        where += " AND property_type = ?"
        params.append(prop_type)

    sql = f"""
        SELECT
            {bucket_expr} AS period,
            COUNT(*) AS n_sales,
            AVG(price) AS mean_price,
            MEDIAN(price) AS median_price
        FROM sales
        WHERE {where}
        GROUP BY 1
        ORDER BY 1;
    """
    df = con.execute(sql, params).df()
    return df


@st.cache_data
def query_compare(postcode_prefixes: list[str], prop_type: str, start_d: date, end_d: date) -> pd.DataFrame:
    con = get_con()

    # We'll build a UNION ALL of each prefix for simplicity and reliability.
    # This avoids complex list binding issues.
    pieces = []
    params = []
    for pc in postcode_prefixes:
        pc = pc.strip().upper()
        like_pattern = pc + "%"
        where = """
            transfer_date BETWEEN ?::DATE AND ?::DATE
            AND UPPER(REPLACE(postcode, ' ', '')) LIKE UPPER(REPLACE(?, ' ', ''))
        """
        p = [to_sql_date(start_d), to_sql_date(end_d), like_pattern]
        if prop_type:
            where += " AND property_type = ?"
            p.append(prop_type)

        pieces.append(f"""
            SELECT
              '{pc}' AS area,
              COUNT(*) AS n_sales,
              AVG(price) AS mean_price,
              MEDIAN(price) AS median_price
            FROM sales
            WHERE {where}
        """)
        params.extend(p)

    sql = " UNION ALL ".join(pieces) + " ORDER BY median_price DESC;"
    return con.execute(sql, params).df()


@st.cache_data
def query_sales_search(postcode_prefix: str, prop_type: str, start_d: date, end_d: date, limit: int = 5000) -> pd.DataFrame:
    con = get_con()
    pc = postcode_prefix.strip().upper()
    like_pattern = pc + "%"

    params = [to_sql_date(start_d), to_sql_date(end_d), like_pattern]
    where = """
        transfer_date BETWEEN ?::DATE AND ?::DATE
        AND UPPER(REPLACE(postcode, ' ', '')) LIKE UPPER(REPLACE(?, ' ', ''))
    """
    if prop_type:
        where += " AND property_type = ?"
        params.append(prop_type)

    sql = f"""
        SELECT
          transfer_date,
          price,
          postcode,
          property_type,
          paon, saon, street, town_city, district, county,
          old_new, duration, category
        FROM sales
        WHERE {where}
        ORDER BY transfer_date DESC
        LIMIT {int(limit)};
    """
    return con.execute(sql, params).df()


@st.cache_data
def query_nearby_sales(center_postcode: str, radius_km: float, days_back: int = 365, limit: int = 2000) -> pd.DataFrame:
    """
    Requires postcodes table from ONSPD:
      postcodes(postcode_nospace, postcode, lat, lon)
    """
    con = get_con()

    pc_norm = normalize_postcode(center_postcode)
    if not pc_norm:
        return pd.DataFrame()

    # get lat/lon of center postcode
    center = con.execute(
        """
        SELECT lat, lon
        FROM postcodes
        WHERE postcode_nospace = ?;
        """,
        [pc_norm],
    ).fetchone()

    if not center:
        return pd.DataFrame()

    lat0, lon0 = center

    # Haversine distance in SQL (km). Uses radians() / sin() / cos() / asin() / sqrt()
    # Earth radius ~6371 km.
    sql = f"""
        WITH recent AS (
          SELECT
            s.transfer_date,
            s.price,
            s.postcode,
            s.property_type,
            p.lat,
            p.lon
          FROM sales s
          JOIN postcodes p
            ON UPPER(REPLACE(s.postcode, ' ', '')) = p.postcode_nospace
          WHERE s.transfer_date >= (CURRENT_DATE - INTERVAL '{int(days_back)} days')
            AND s.price IS NOT NULL
            AND s.postcode IS NOT NULL
        )
        SELECT
          transfer_date, price, postcode, property_type, lat, lon,
          2 * 6371 * ASIN(
            SQRT(
              POWER(SIN((RADIANS(lat) - RADIANS(?)) / 2), 2)
              + COS(RADIANS(?)) * COS(RADIANS(lat))
              * POWER(SIN((RADIANS(lon) - RADIANS(?)) / 2), 2)
            )
          ) AS distance_km
        FROM recent
        WHERE 2 * 6371 * ASIN(
            SQRT(
              POWER(SIN((RADIANS(lat) - RADIANS(?)) / 2), 2)
              + COS(RADIANS(?)) * COS(RADIANS(lat))
              * POWER(SIN((RADIANS(lon) - RADIANS(?)) / 2), 2)
            )
          ) <= ?
        ORDER BY transfer_date DESC
        LIMIT {int(limit)};
    """
    params = [lat0, lat0, lon0, lat0, lat0, lon0, float(radius_km)]
    return con.execute(sql, params).df()


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="UK House Price Explorer", layout="wide")

st.title("UK House Price Explorer (HM Land Registry Price Paid Data)")
st.caption("Streamlit + DuckDB. Designed for exploring trends, comparisons, and recent sales.")

# Global filters
min_d, max_d = get_date_range()

with st.sidebar:
    st.header("Global filters")

    # Suggest common postcode areas to reduce typing
    areas_df = get_distinct_postcode_areas()
    area_options = areas_df["area"].tolist()
    default_area = area_options[0] if area_options else "SW1A"

    postcode_prefix = st.text_input(
        "Postcode / prefix (e.g. SW1A, M1, BS1 or full postcode)",
        value=default_area,
        help="We match all postcodes starting with this text.",
    ).strip().upper()

    types = get_property_types()
    type_labels = ["All"] + [prop_label(t) for t in types]
    type_map = {"All": ""} | {prop_label(t): t for t in types}
    type_choice = st.selectbox("Property type", options=type_labels, index=0)
    prop_type = type_map[type_choice]

    # date range picker (ensure both are datetime.date)
    default_start = max(min_d, date(max_d.year - 5, 1, 1))
    start_d = st.date_input("From", value=default_start, min_value=min_d, max_value=max_d)
    end_d = st.date_input("To", value=max_d, min_value=min_d, max_value=max_d)

    if start_d > end_d:
        st.error("Start date must be before end date.")
        st.stop()

tabs = st.tabs(["Trends", "Compare", "Sales search", "Nearby (requires ONSPD)"])

# -----------------------
# Trends tab
# -----------------------
with tabs[0]:
    st.subheader("Trends: average and median prices over time")

    freq = st.radio("Time bucket", options=["month", "quarter", "year"], index=1, horizontal=True)

    df = query_trend(postcode_prefix, prop_type, start_d, end_d, freq)

    if df.empty:
        st.warning("No sales matched your filters. Try a broader postcode prefix or wider date range.")
    else:
        metric = st.selectbox("Metric", ["median_price", "mean_price"], index=0)
        fig = px.line(df, x="period", y=metric, markers=True, title=f"{metric.replace('_', ' ').title()} over time")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Sales in range", int(df["n_sales"].sum()))
        with c2:
            st.metric("Latest median", f"£{int(df['median_price'].iloc[-1]):,}")

        st.dataframe(df, use_container_width=True)

# -----------------------
# Compare tab
# -----------------------
with tabs[1]:
    st.subheader("Compare areas")

    st.write("Enter multiple postcode prefixes (e.g. `SW1A, W1, EC1, M1`).")
    compare_text = st.text_input("Areas to compare (comma separated)", value=postcode_prefix)
    prefixes = [p.strip().upper() for p in compare_text.split(",") if p.strip()]
    prefixes = prefixes[:10]  # keep it reasonable

    if len(prefixes) < 1:
        st.info("Add at least one postcode prefix to compare.")
    else:
        cmp_df = query_compare(prefixes, prop_type, start_d, end_d)
        if cmp_df.empty:
            st.warning("No results for those prefixes. Try different ones or widen your date range.")
        else:
            fig = px.bar(cmp_df, x="area", y="median_price", title="Median price by area")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cmp_df, use_container_width=True)

# -----------------------
# Sales search tab
# -----------------------
with tabs[2]:
    st.subheader("Sales search (raw transactions)")

    limit = st.slider("Max rows to show", min_value=100, max_value=5000, value=1000, step=100)
    sales_df = query_sales_search(postcode_prefix, prop_type, start_d, end_d, limit=limit)

    if sales_df.empty:
        st.warning("No transactions found for your filters.")
    else:
        st.write("Tip: sort the table by transfer_date or price.")
        st.dataframe(sales_df, use_container_width=True)

        # optional download
        csv = sales_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="sales_export.csv", mime="text/csv")

# -----------------------
# Nearby tab
# -----------------------
with tabs[3]:
    st.subheader("Nearby recent sales (requires ONS Postcode Directory / ONSPD)")

    if not table_exists("postcodes"):
        st.warning(
            "The `postcodes` table was not found in your database. "
            "This feature requires ONSPD to be loaded into DuckDB."
        )
    else:
        center_pc = st.text_input("Center postcode (full postcode)", value="")
        radius_km = st.slider("Radius (km)", min_value=0.2, max_value=10.0, value=2.0, step=0.2)
        days_back = st.slider("Look back (days)", min_value=30, max_value=3650, value=365, step=30)
        run = st.button("Find nearby sales")

        if run:
            near_df = query_nearby_sales(center_pc, radius_km, days_back=days_back)

            if near_df.empty:
                st.warning("No nearby sales found (or the postcode wasn’t found in ONSPD).")
            else:
                st.dataframe(near_df, use_container_width=True)

                if pdk is not None:
                    # Map points
                    st.write("Map view (dots are sales).")
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=near_df,
                        get_position="[lon, lat]",
                        get_radius=20,
                        pickable=True,
                    )
                    view_state = pdk.ViewState(
                        latitude=float(near_df["lat"].mean()),
                        longitude=float(near_df["lon"].mean()),
                        zoom=11,
                    )
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{postcode}\n£{price}\n{transfer_date}"}))
                else:
                    st.info("pydeck is not available; map view disabled.")
