"""Quick import smoke test for all churn modules."""

import sys

modules = [
    "churn.config",
    "churn.data.loader",
    "churn.data.validator",
    "churn.data.splitter",
    "churn.features.call_features",
    "churn.features.usage_features",
    "churn.features.contract_features",
    "churn.features.target_encoder",
    "churn.features.feature_store",
    "churn.models.base_model",
    "churn.models.lgbm_model",
    "churn.models.survival_model",
    "churn.models.uplift_model",
    "churn.models.calibrator",
    "churn.training.tuner",
    "churn.training.trainer",
    "churn.evaluation.metrics",
    "churn.evaluation.plots",
    "churn.evaluation.business_impact",
    "churn.explainability.shap_explainer",
    "churn.explainability.local_explainer",
    "churn.explainability.churn_segments",
    "churn.scoring.batch_scorer",
    "churn.scoring.output_formatter",
    "churn.monitoring.drift_detector",
    "churn.monitoring.psi_calculator",
    "churn.monitoring.alerting",
    "churn.api.schemas",
    "churn.api.model_loader",
    "churn.api.main",
]

failed = []
for mod in modules:
    try:
        __import__(mod)
        print(f"  OK  {mod}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}")
        failed.append((mod, str(e)))

print(f"\n{len(modules) - len(failed)}/{len(modules)} modules imported successfully")
if failed:
    print("FAILURES:")
    for m, e in failed:
        print(f"  {m}: {e}")
    sys.exit(1)
