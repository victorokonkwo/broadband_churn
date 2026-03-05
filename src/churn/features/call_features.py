"""
Call centre feature engineering.
All aggregations run inside DuckDB — calls.csv is >50MB and must NOT be
fully loaded into pandas RAM. Features are computed as-of snapshot_date
to prevent any temporal leakage.
"""
from __future__ import annotations

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Rolling windows to compute call frequency features over
WINDOWS = [7, 14, 30, 90]


def build_call_features(
    snapshot_date: str,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Compute call features for all customers with call activity up to snapshot_date.

    Args:
        snapshot_date : ISO date string 'YYYY-MM-DD' — upper bound for feature computation
        con           : Active DuckDB connection with a 'calls' table

    Returns:
        DataFrame with one row per customer, columns:
            unique_customer_identifier,
            call_count_{W}d  (for W in WINDOWS),
            loyalty_call_flag_30d,
            loyalty_call_count_90d,
            pct_loyalty_calls_90d,
            avg_talk_time_30d,
            avg_hold_time_30d,
            days_since_last_call,
            call_frequency_trend
    """
    logger.info("Building call features as of %s …", snapshot_date)

    window_cols = "\n".join(
        f"    COUNT_IF(event_date >= DATE '{snapshot_date}' - INTERVAL {w} DAY "
        f"AND event_date < DATE '{snapshot_date}')  AS call_count_{w}d,"
        for w in WINDOWS
    )

    sql = f"""
    SELECT
        unique_customer_identifier,

        -- Rolling call counts
        {window_cols}

        -- Loyalty-specific signals (strongest pre-churn indicator)
        MAX(CASE WHEN call_type = 'Loyalty'
                  AND event_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                  AND event_date < DATE '{snapshot_date}'
             THEN 1 ELSE 0 END)                                         AS loyalty_call_flag_30d,

        COUNT_IF(call_type = 'Loyalty'
                AND event_date >= DATE '{snapshot_date}' - INTERVAL 90 DAY
                AND event_date < DATE '{snapshot_date}')                AS loyalty_call_count_90d,

        -- Call type mix (proportion Loyalty calls in last 90 days)
        COALESCE(
            COUNT_IF(call_type = 'Loyalty'
                    AND event_date >= DATE '{snapshot_date}' - INTERVAL 90 DAY
                    AND event_date < DATE '{snapshot_date}')
            / NULLIF(
                COUNT_IF(event_date >= DATE '{snapshot_date}' - INTERVAL 90 DAY
                        AND event_date < DATE '{snapshot_date}'), 0), 0
        )                                                               AS pct_loyalty_calls_90d,

        -- Service experience signals
        AVG(CASE WHEN event_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                  AND event_date < DATE '{snapshot_date}'
             THEN talk_time_seconds END)                                AS avg_talk_time_30d,

        AVG(CASE WHEN event_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                  AND event_date < DATE '{snapshot_date}'
             THEN hold_time_seconds END)                                AS avg_hold_time_30d,

        -- Recency
        DATEDIFF('day',
            MAX(CASE WHEN event_date < DATE '{snapshot_date}'
                 THEN event_date END),
            DATE '{snapshot_date}')                                     AS days_since_last_call,

        -- Trend: call rate last 30d vs last 90d (> 1 = accelerating contact)
        COALESCE(
            COUNT_IF(event_date >= DATE '{snapshot_date}' - INTERVAL 30 DAY
                    AND event_date < DATE '{snapshot_date}')
            / NULLIF(
                COUNT_IF(event_date >= DATE '{snapshot_date}' - INTERVAL 90 DAY
                        AND event_date < DATE '{snapshot_date}') / 3.0, 0), 1.0
        )                                                               AS call_frequency_trend

    FROM calls
    WHERE event_date < DATE '{snapshot_date}'
    GROUP BY unique_customer_identifier
    """

    df = con.execute(sql).df()
    logger.info("  call features: %s customers", f"{len(df):,}")
    return df
