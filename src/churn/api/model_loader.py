"""
Model loader for the API serving layer.
Loads model + calibrator from disk (or MLflow registry) at startup.
Uses a module-level singleton so the model is loaded once, not per request.
"""

from __future__ import annotations

import logging
from pathlib import Path

from churn.config import cfg
from churn.models.calibrator import ChurnCalibrator
from churn.models.lgbm_model import LGBMChurnModel

logger = logging.getLogger(__name__)


def _resolve_artefact_path(
    artefacts_dir: Path,
    preferred_name: str,
    fallback_name: str,
) -> Path:
    preferred = artefacts_dir / preferred_name
    fallback = artefacts_dir / fallback_name
    if preferred.exists():
        return preferred
    return fallback


class ModelRegistry:
    """Global model registry — instantiated once at API startup."""

    def __init__(self) -> None:
        self.model: LGBMChurnModel | None = None
        self.calibrator: ChurnCalibrator | None = None
        self.version: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.calibrator is not None

    def load_from_disk(
        self,
        model_path: Path | None = None,
        calibrator_path: Path | None = None,
    ) -> None:
        artefacts = cfg.paths.model_artefacts_dir
        model_path = model_path or _resolve_artefact_path(
            artefacts,
            preferred_name="lgbm_churn_model.pkl",
            fallback_name="lgbm_model.joblib",
        )
        calibrator_path = calibrator_path or _resolve_artefact_path(
            artefacts,
            preferred_name="calibrator.pkl",
            fallback_name="calibrator.joblib",
        )

        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not calibrator_path.exists():
            logger.error("Calibrator file not found: %s", calibrator_path)
            raise FileNotFoundError(f"Calibrator not found at {calibrator_path}")

        self.model = LGBMChurnModel.load(model_path)
        self.calibrator = ChurnCalibrator.load(calibrator_path)
        self.version = model_path.stem
        logger.info("Model loaded from disk — version=%s", self.version)


# Module-level singleton
registry = ModelRegistry()
