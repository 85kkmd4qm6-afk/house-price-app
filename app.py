import re
import pandas as pd
import plotly.express as px
import psycopg
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="NW House Price Explorer", layout="wide")

st.title("North West House Price Explorer")
st.caption(
    "HM Land Registry Price Paid Data • Filtered to North West England (ONSPD region) • 2010–present • Backend: Neon Postgres"
)

# -----------------------------
# Helpers
# -----------------------------
def normalize_postcode(pc: str) -> str:
    """Normalize postcode to compact uppercase no-space form."""
    if not pc:
        return ""
    return re.sub(r"\s+", "", pc.strip().upper())


# -----------------------------
# Database helpers
# -----------------------------
@st.cache_resource
def get_pg():
    # Put DATABASE_URL in Streamlit Cloud Secrets:
    # DATABASE_URL = "postgres://...sslmode=require"
    return psycopg.connect(st.secrets["DATABASE_URL"])


def run_query(sql: str, params=None) -> pd.DataFrame:
    with get_pg().cursor() as cur:
        cur.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


@st.cache_data
def get_date_range():
    df = run_query("SELECT MIN(transfer_date) AS min_d, MAX(transfer_date) AS max_d FROM sales;")
    return df.loc[0, "min_d"], df.loc[0, "max_d"]


@st.cache_data
def list_outcodes() -> list[str]:
    # outcode is postcode_nospace minus last 3 chars (inward code)
    df = run_query("""
        SELECT LEFT(postcode_nospace, LENGTH(postcode_nospace)-3) AS outcode
        FROM sales
        WHERE postcode_nospace IS NOT NULL AND LENGTH(postcode_nospace) > 3
        GROUP BY 1
        ORDER BY 1;
    """)
    return df["outcode"].dropna().astype(str).tolist()


@st.cache_data
def list_property_types() -> list[str]:
    df = run_query("""
        SELECT DISTINCT property_type
        FROM sales
        WHERE property_type IS NOT NULL AND property_type <> ''
        ORDER BY 1;
    """)
    return df["property_type"].astype(str).tolist()


def trend_where(where_sql: str, params: list):
    # Postgres median:
    # percentile_cont(0.5) WITHIN GROUP (ORDER BY price)
    return run_query(f"""
        SELECT
          DATE_TRUNC('month', transfer_date)::date AS month,
          COUNT(*) AS n_sales,
          AVG(price)::bigint AS mean_price,
          percentile_cont(0.5) WITHIN GROUP (ORDER BY price)::bigint AS median_price
        FROM sales
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY 1;
    """, params)


def recent_sales_where(where_sql: str, params: list, limit_n: int = 200):
    return run_query(f"""
        SELECT transfer_date, price, postcode, property_type, old_new, duration, category
        FROM sales
        WHERE {where_sql}
        ORDER BY transfer_date DESC
        LIMIT %s;
    """, params + [limit_n])


# -----------------------------
# Sidebar filters
# -----------------------------
min_d, max_d = get_date_range()
if min_d is None or max_d is None:
    st.error("No data found in table `sales`. Check your DATABASE_URL and that the table name is `sales`.")
    st.stop()

all_outcodes = list_outcodes()
prop_types = ["(Any)"] + list_property_types()

with st.sidebar:
    st.header("Filters")

    start_d, end_d = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )

    st.subheader("Area")
    mode = st.radio("Mode", ["Outcode (recommended)", "Full postcode"], index=0)

    selected_outcode = None
    selected_postcode_ns = None

    if mode.startswith("Outcode"):
        if not all_outcodes:
            st.error("No outcodes found. Check that sales.postcode_nospace contains valid postcodes.")
            st.stop()
        selected_outcode = st.selectbox("Outcode", all_outcodes, index=0)
    else:
        pc_in = st.text_input("Full postcode (e.g. PR7 2AA)")
        selected_postcode_ns = normalize_postcode(pc_in)
        if pc_in and len(selected_postcode_ns) < 5:
            st.warning("That postcode looks too short — check formatting.")

    property_type = st.selectbox("Property type", prop_types, index=0)

    st.subheader("Compare outcodes")
    compare_outcodes = st.multiselect(
        "Select outcodes to compare",
        options=all_outcodes,
        default=[],
        help="Plots median price for multiple outcodes on one chart.",
    )

    limit_recent = st.slider("Recent sales rows", 25, 500, 200, 25)


# -----------------------------
# Build WHERE clauses
# -----------------------------
filters = []
params = []

filters.append("transfer_date BETWEEN %s AND %s")
params.extend([start_d, end_d])

if mode.startswith("Outcode") and selected_outcode:
    filters.append("LEFT(postcode_nospace, LENGTH(postcode_nospace)-3) = %s")
    params.append(selected_outcode)

if mode == "Full postcode" and selected_postcode_ns:
    filters.append("postcode_nospace = %s")
    params.append(selected_postcode_ns)

if property_type != "(Any)":
    filters.append("property_type = %s")
    params.append(property_type)

where_sql = " AND ".join(filters)


# -----------------------------
# Main layout
# -----------------------------
colA, colB = st.columns([1.25, 1])

with colA:
    st.subheader("Price trends (monthly)")

    df_trend = trend_where(where_sql, params)

    if df_trend.empty:
        st.warning("No results for this selection.")
    else:
        df_plot = df_trend.melt(
            id_vars=["month", "n_sales"],
            value_vars=["mean_price", "median_price"],
            var_name="metric",
            value_name="price",
        )
        fig = px.line(
            df_plot,
            x="month",
            y="price",
            color="metric",
            hover_data=["n_sales"],
        )
        fig.update_layout(yaxis_title="Price (£)", xaxis_title="Month", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Sales in period", f"{int(df_trend['n_sales'].sum()):,}")
        k2.metric("Latest mean (£)", f"{int(df_trend.iloc[-1]['mean_price']):,}")
        k3.metric("Latest median (£)", f"{int(df_trend.iloc[-1]['median_price']):,}")

with colB:
    st.subheader("Recent sales")
    df_recent = recent_sales_where(where_sql, params, limit_n=limit_recent)

    if df_recent.empty:
        st.info("No sales found for this selection.")
    else:
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
        st.download_button(
            "Download recent sales (CSV)",
            data=df_recent.to_csv(index=False).encode("utf-8"),
            file_name="recent_sales.csv",
            mime="text/csv",
        )


# -----------------------------
# Comparison
# -----------------------------
if compare_outcodes:
    st.divider()
    st.subheader("Outcode comparison (median price)")

    comp_filters = []
    comp_params = []

    comp_filters.append("transfer_date BETWEEN %s AND %s")
    comp_params.extend([start_d, end_d])

    if property_type != "(Any)":
        comp_filters.append("property_type = %s")
        comp_params.append(property_type)

    placeholders = ", ".join(["%s"] * len(compare_outcodes))
    comp_filters.append(f"LEFT(postcode_nospace, LENGTH(postcode_nospace)-3) IN ({placeholders})")
    comp_params.extend(compare_outcodes)

    comp_where = " AND ".join(comp_filters)

    df_comp = run_query(f"""
        SELECT
          DATE_TRUNC('month', transfer_date)::date AS month,
          LEFT(postcode_nospace, LENGTH(postcode_nospace)-3) AS outcode,
          percentile_cont(0.5) WITHIN GROUP (ORDER BY price)::bigint AS median_price
        FROM sales
        WHERE {comp_where}
        GROUP BY 1, 2
        ORDER BY 1, 2;
    """, comp_params)

    if df_comp.empty:
        st.warning("No comparison data returned.")
    else:
        fig2 = px.line(df_comp, x="month", y="median_price", color="outcode")
        fig2.update_layout(yaxis_title="Median price (£)", xaxis_title="Month", legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Tip: Outcode mode is faster. Full postcode mode is narrower and can return fewer sales.")
