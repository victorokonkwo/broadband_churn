"""
Broadband usage feature engineering.
Declining usage is a leading churn indicator — customers who stop using the
service heavily are more likely to be considering leaving.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def build_usage_features(
    snapshot_date: str,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Compute broadband usage features for all customers up to snapshot_date.

    Features:
        avg_download_30d           : Mean daily download MB (last 30 days)
        avg_upload_30d             : Mean daily upload MB (last 30 days)
        avg_download_7d            : Mean daily download MB (last 7 days)
        download_trend_7_30        : avg_download_7d / avg_download_30d
                                     < 1 = declining — churn signal
        download_pct_change_30_60d : % change between last 30d and prior 30d
        zero_usage_days_30d        : Days with 0 download in last 30 days
                                     (fault or customer absence)
        usage_volatility_30d       : Std dev of daily download (last 30 days)
        avg_daily_total_mb_30d     : download + upload combined
    """
    logger.info("Building usage features as of %s …", snapshot_date)

    sql = f"""
    WITH usage_clean AS (
        SELECT
            unique_customer_identifier,
            calendar_date,
            TRY_CAST(usage_download_mbs AS DOUBLE) AS usage_download_mbs,
            TRY_CAST(usage_upload_mbs   AS DOUBLE) AS usage_upload_mbs
        FROM usage
        WHERE calendar_date < DATE '{snapshot_date}'
    )
    SELECT
        unique_customer_identifier,

        -- Last 30 days averages
        AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                  AND calendar_date < DATE '{snapshot_date}'
             THEN usage_download_mbs END)                          AS avg_download_30d,

        AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                  AND calendar_date < DATE '{snapshot_date}'
             THEN usage_upload_mbs END)                            AS avg_upload_30d,

        -- Last 7 days average (recent behaviour)
        AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 7 DAY
                  AND calendar_date < DATE '{snapshot_date}'
             THEN usage_download_mbs END)                          AS avg_download_7d,

        -- Trend: 7d vs 30d (< 1 = declining, strong churn signal)
        COALESCE(
            AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 7 DAY
                      AND calendar_date < DATE '{snapshot_date}'
                 THEN usage_download_mbs END)
            / NULLIF(
                AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                          AND calendar_date < DATE '{snapshot_date}'
                     THEN usage_download_mbs END), 0), 1.0
        )                                                          AS download_trend_7_30,

        -- % change between last 30d and prior 30d (day -30 to -60)
        COALESCE(
            (AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                       AND calendar_date < DATE '{snapshot_date}'
                  THEN usage_download_mbs END)
             - AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 60 DAY
                         AND calendar_date < DATE '{snapshot_date}' - INTERVAL 30 DAY
                    THEN usage_download_mbs END))
            / NULLIF(
                AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 60 DAY
                          AND calendar_date < DATE '{snapshot_date}' - INTERVAL 30 DAY
                     THEN usage_download_mbs END), 0), 0.0
        )                                                          AS download_pct_change_30_60d,

        -- Zero usage days (line fault / customer absence signal)
        COUNT_IF(
            calendar_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
            AND calendar_date < DATE '{snapshot_date}'
            AND (usage_download_mbs IS NULL OR usage_download_mbs = 0)
        )                                                          AS zero_usage_days_30d,

        -- Volatility (std dev — high = erratic / unusual pattern)
        STDDEV(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                     AND calendar_date < DATE '{snapshot_date}'
                THEN usage_download_mbs END)                       AS usage_volatility_30d,

        -- Combined throughput
        AVG(CASE WHEN calendar_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                  AND calendar_date < DATE '{snapshot_date}'
             THEN COALESCE(usage_download_mbs, 0) + COALESCE(usage_upload_mbs, 0) END
        )                                                          AS avg_daily_total_mb_30d

    FROM usage_clean
    WHERE calendar_date < DATE '{snapshot_date}'
    GROUP BY unique_customer_identifier
    """

    df = con.execute(sql).df()
    logger.info("  usage features: %s customers", f"{len(df):,}")
    return df
