"""Quick check of actual categorical values in data."""

import duckdb

con = duckdb.connect(":memory:")
con.execute("CREATE TABLE ci AS SELECT * FROM read_parquet('data/raw/customer_info.parquet')")
con.execute("CREATE TABLE ce AS SELECT * FROM read_csv_auto('data/raw/cease.csv')")
con.execute("CREATE TABLE ca AS SELECT * FROM read_csv_auto('data/raw/calls.csv')")

print("=== contract_status values ===")
for r in con.execute("SELECT DISTINCT contract_status FROM ci ORDER BY 1").fetchall():
    print(f"  {repr(r[0])}")

print("\n=== call_type values ===")
for r in con.execute("SELECT DISTINCT call_type FROM ca ORDER BY 1").fetchall():
    print(f"  {repr(r[0])}")

print("\n=== reason_description_insight values ===")
for r in con.execute("SELECT DISTINCT reason_description_insight FROM ce ORDER BY 1").fetchall():
    print(f"  {repr(r[0])}")

print("\n=== technology values ===")
for r in con.execute("SELECT DISTINCT technology FROM ci ORDER BY 1").fetchall():
    print(f"  {repr(r[0])}")

print("\n=== date ranges ===")
print("customer_info:", con.execute("SELECT MIN(datevalue), MAX(datevalue) FROM ci").fetchone())
print("calls:", con.execute("SELECT MIN(event_date), MAX(event_date) FROM ca").fetchone())
print(
    "cease:",
    con.execute("SELECT MIN(cease_placed_date), MAX(cease_placed_date) FROM ce").fetchone(),
)

print("\n=== usage sample ===")
con.execute("CREATE TABLE usage AS SELECT * FROM read_parquet('data/raw/usage.parquet') LIMIT 5")
print(con.execute("SELECT * FROM usage").fetchdf().to_string())
