import duckdb
import os

DB_PATH = "app.duckdb"
PPD_PATH = "data/ppd_sample.csv"
ONSPD_PATH = "data/onspd.csv"

def main():
    if not os.path.exists(PPD_PATH):
        raise FileNotFoundError(f"Missing {PPD_PATH}. Put the Land Registry CSV there.")

    con = duckdb.connect(DB_PATH)

    con.execute("DROP TABLE IF EXISTS sales;")
    con.execute("DROP TABLE IF EXISTS postcodes;")

    print("Loading Price Paid Data... (assuming NO header row)")

    print("Loading Price Paid Data... (assuming NO header row)")

    con.execute(f"""
    CREATE TABLE sales AS
    SELECT
        TRIM(column00) AS tx_id,
        column01 AS price,
        column02 AS transfer_date,
        NULLIF(TRIM(column03), '') AS postcode,
        TRIM(column04) AS property_type,
        TRIM(column05) AS old_new,
        TRIM(column06) AS duration,
        TRIM(column07) AS paon,
        TRIM(column08) AS saon,
        TRIM(column09) AS street,
        TRIM(column10) AS locality,
        TRIM(column11) AS town_city,
        TRIM(column12) AS district,
        TRIM(column13) AS county,
        TRIM(column14) AS category,
        TRIM(column15) AS record_status
    FROM read_csv_auto('{PPD_PATH}', header=False)
    WHERE TRIM(column15) = 'A'
      AND column03 IS NOT NULL
      AND column01 IS NOT NULL;
""")


    con.execute("CREATE INDEX IF NOT EXISTS idx_sales_postcode_date ON sales(postcode, transfer_date);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sales_type_date ON sales(property_type, transfer_date);")

    if os.path.exists(ONSPD_PATH):
        con.execute(f"""
    CREATE TABLE postcodes AS
    SELECT
        UPPER(REPLACE(TRIM(pcds), ' ', '')) AS postcode_nospace,
        TRIM(pcds) AS postcode,
        lat AS lat,
        long AS lon
    FROM read_csv_auto('{ONSPD_PATH}', header=True)
    WHERE pcds IS NOT NULL AND lat IS NOT NULL AND long IS NOT NULL;
""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pc_nospace ON postcodes(postcode_nospace);")

    con.close()
    print(f"Built {DB_PATH} successfully.")

if __name__ == "__main__":
    main()
