"""
Data ingestion layer.
Loads raw CSV / Parquet files into DuckDB and exposes validated DataFrames.
All reads go through DuckDB so calls.csv (>50MB) is never fully loaded into RAM.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from churn.config import cfg

logger = logging.getLogger(__name__)


def get_connection(
    db_path: Path | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Return a persistent DuckDB connection to the processed database."""
    path = str(db_path or cfg.paths.duckdb_path)
    cfg.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path, read_only=read_only)


def ingest_raw_tables(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """
    Read raw source files and create (or replace) permanent tables in DuckDB.
    Safe to re-run — uses CREATE OR REPLACE.
    """
    if con is None:
        con = get_connection()

    cease   = str(cfg.data.cease_csv)
    ci      = str(cfg.data.customer_info_parquet)
    usage   = str(cfg.data.usage_parquet)
    calls   = str(cfg.data.calls_csv)

    logger.info("Ingesting cease.csv …")
    con.execute(f"CREATE OR REPLACE TABLE cease AS SELECT * FROM read_csv_auto('{cease}')")

    logger.info("Ingesting customer_info.parquet …")
    con.execute(f"CREATE OR REPLACE TABLE customer_info AS SELECT * FROM read_parquet('{ci}')")

    logger.info("Ingesting usage.parquet …")
    con.execute(f"CREATE OR REPLACE TABLE usage AS SELECT * FROM read_parquet('{usage}')")

    logger.info("Ingesting calls.csv (large file — streaming via DuckDB) …")
    con.execute(f"CREATE OR REPLACE TABLE calls AS SELECT * FROM read_csv_auto('{calls}')")

    logger.info("All raw tables ingested.")
    _log_table_stats(con)


def _log_table_stats(con: duckdb.DuckDBPyConnection) -> None:
    for table in ["cease", "customer_info", "usage", "calls"]:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # type: ignore[index]
        logger.info("  %-20s %s rows", table, f"{n:,}")


# ─── Convenience query helpers ────────────────────────────────────────────────

def query(sql: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Execute SQL and return a pandas DataFrame."""
    if con is None:
        con = get_connection()
    return con.execute(sql).df()


def load_cease(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    return query("SELECT * FROM cease", con)


def load_customer_info(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    return query("SELECT * FROM customer_info", con)


def load_usage_sample(n: int = 100_000, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Sample n rows from usage for EDA — full table may be large."""
    return query(f"SELECT * FROM usage USING SAMPLE {n}", con)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    con = get_connection()
    ingest_raw_tables(con)
    logger.info("Done — database written to %s", cfg.paths.duckdb_path)
