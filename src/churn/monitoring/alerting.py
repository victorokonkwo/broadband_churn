"""
Alerting module — sends notifications when drift or model degradation
thresholds are breached. Production integration points:
    - Slack webhook
    - PagerDuty (critical drift)
    - Email (weekly summary)
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def send_slack_alert(
    webhook_url: str,
    title: str,
    message: str,
    severity: str = "warning",
) -> None:
    """Send a Slack notification via incoming webhook."""
    color_map = {"info": "#36a64f", "warning": "#ff9900", "critical": "#ff0000"}
    payload = {
        "attachments": [
            {
                "color": color_map.get(severity, "#cccccc"),
                "title": f":rotating_light: {title}",
                "text": message,
                "fields": [
                    {"title": "Severity", "value": severity, "short": True},
                    {"title": "Pipeline", "value": "churn-prediction", "short": True},
                ],
            }
        ]
    }

    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("Slack alert sent: %s", title)
    except Exception as e:
        logger.error("Failed to send Slack alert: %s", e)


def format_drift_alert(monitoring_result: dict[str, Any]) -> tuple[str, str, str]:
    """
    Format monitoring results into a human-readable alert message.

    Returns:
        (title, message, severity)
    """
    drift = monitoring_result.get("drift_detected", False)
    share = monitoring_result.get("drift_share", 0.0)
    psi_alerts = monitoring_result.get("psi_alerts", [])

    if not drift and not psi_alerts:
        return ("No Drift Detected", "All features stable.", "info")

    severity = "critical" if share > 0.3 or len(psi_alerts) > 3 else "warning"
    lines = [
        f"Drift detected: {'Yes' if drift else 'No'}",
        f"Drifted feature share: {share:.1%}",
    ]
    if psi_alerts:
        lines.append(f"Features with PSI > 0.2: {', '.join(psi_alerts)}")
    lines.append("→ **Recommended action**: trigger retraining pipeline")

    title = "Churn Model Drift Alert"
    message = "\n".join(lines)
    return title, message, severity
