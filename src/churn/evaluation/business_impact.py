"""
Business impact calculator.
Translates model performance into expected financial value.
This is what non-technical stakeholders actually care about.

Key framing:
    - FN cost (missed churner): lost customer → estimated annual revenue foregone
    - FP cost (false alarm): unnecessary retention call → agent time cost
    - Net model value = churners rescued × avg_revenue - false_positives × call_cost
    - Compare to random calling baseline at same capacity
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default business assumptions (override as needed)
AVG_ANNUAL_REVENUE_GBP = 360.0    # £30/month × 12
CALL_COST_GBP          = 8.0      # agent time cost per outbound call (~10 min)
RETENTION_SUCCESS_RATE = 0.25     # 25% of contacted churners are retained
CAPACITY_PCT           = 0.20     # ops team can contact top 20% of scored base


def compute_business_impact(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_customers: int | None = None,
    capacity_pct: float = CAPACITY_PCT,
    avg_revenue: float = AVG_ANNUAL_REVENUE_GBP,
    call_cost: float = CALL_COST_GBP,
    retention_success_rate: float = RETENTION_SUCCESS_RATE,
) -> pd.DataFrame:
    """
    Compare model-guided outreach vs. random calling at the same capacity.

    Returns:
        DataFrame with financial impact at different threshold / capacity points
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    n = n_customers or len(y)
    base_rate = y.mean()

    results = []
    for cap in [0.05, 0.10, 0.15, 0.20, 0.30]:
        k = max(1, int(len(y) * cap))
        top_k_idx  = np.argsort(p)[::-1][:k]
        rand_k_idx = np.random.choice(len(y), k, replace=False)

        def _impact(idx: np.ndarray, label: str) -> dict:
            tp = y[idx].sum()
            fp = k - tp
            rescued = tp * retention_success_rate
            revenue_saved = rescued * avg_revenue
            call_spend = k * call_cost
            net_value = revenue_saved - call_spend
            return {
                "strategy": label,
                "capacity_pct": cap,
                "customers_contacted": k,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "precision": round(tp / k, 4) if k > 0 else 0,
                "capture_rate": round(tp / y.sum(), 4) if y.sum() > 0 else 0,
                "estimated_retained": round(rescued, 1),
                "revenue_saved_gbp": round(revenue_saved, 2),
                "call_spend_gbp": round(call_spend, 2),
                "net_value_gbp": round(net_value, 2),
            }

        results.append(_impact(top_k_idx, "Model-guided"))
        results.append(_impact(rand_k_idx, "Random baseline"))

    df = pd.DataFrame(results)
    _log_headline(df, capacity_pct=CAPACITY_PCT)
    return df


def _log_headline(df: pd.DataFrame, capacity_pct: float) -> None:
    model_row  = df[(df["strategy"] == "Model-guided") & (df["capacity_pct"] == capacity_pct)]
    random_row = df[(df["strategy"] == "Random baseline") & (df["capacity_pct"] == capacity_pct)]

    if model_row.empty or random_row.empty:
        return

    m = model_row.iloc[0]
    r = random_row.iloc[0]
    uplift = m["net_value_gbp"] - r["net_value_gbp"]

    logger.info(
        "Business impact @ top %s%%:  "
        "Model net value = £%s  |  Random = £%s  |  Model uplift = £%s",
        int(capacity_pct * 100),
        f"{m['net_value_gbp']:,.0f}",
        f"{r['net_value_gbp']:,.0f}",
        f"{uplift:,.0f}",
    )
