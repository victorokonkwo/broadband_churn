"""
Central configuration module.
Loads conf/config.yaml and exposes a typed Config dataclass.
All other modules import from here — no hardcoded paths anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

# ─── Locate the project root (where conf/ lives) ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONF_PATH = PROJECT_ROOT / "conf" / "config.yaml"


@dataclass
class PathsConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    features_dir: Path
    outputs_dir: Path
    figures_dir: Path
    reports_dir: Path
    model_artefacts_dir: Path
    duckdb_path: Path
    scored_customers_path: Path


@dataclass
class DataFilesConfig:
    calls_csv: Path
    cease_csv: Path
    customer_info_parquet: Path
    usage_parquet: Path


@dataclass
class SplitsConfig:
    train_end_date: str
    val_end_date: str
    test_end_date: str


@dataclass
class LabelConfig:
    prediction_window_days: int
    include_pending_ceases: bool


@dataclass
class ModelConfig:
    type: str
    registry_name: str


@dataclass
class ScoringConfig:
    high_risk_threshold: float
    medium_risk_threshold: float
    top_n_drivers: int


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass
class Config:
    paths: PathsConfig
    data: DataFilesConfig
    splits: SplitsConfig
    label: LabelConfig
    model: ModelConfig
    scoring: ScoringConfig
    mlflow: MLflowConfig


def load_config(conf_path: Path = CONF_PATH) -> Config:
    """Load and parse conf/config.yaml into a typed Config object."""
    raw = OmegaConf.load(conf_path)

    root = PROJECT_ROOT

    def _p(rel: str) -> Path:
        return root / rel

    return Config(
        paths=PathsConfig(
            raw_data_dir=_p(raw.paths.raw_data_dir),
            processed_data_dir=_p(raw.paths.processed_data_dir),
            features_dir=_p(raw.paths.features_dir),
            outputs_dir=_p(raw.paths.outputs_dir),
            figures_dir=_p(raw.paths.figures_dir),
            reports_dir=_p(raw.paths.reports_dir),
            model_artefacts_dir=_p(raw.paths.model_artefacts_dir),
            duckdb_path=_p(raw.paths.duckdb_path),
            scored_customers_path=_p(raw.paths.scored_customers_path),
        ),
        data=DataFilesConfig(
            calls_csv=_p(raw.data.calls_csv),
            cease_csv=_p(raw.data.cease_csv),
            customer_info_parquet=_p(raw.data.customer_info_parquet),
            usage_parquet=_p(raw.data.usage_parquet),
        ),
        splits=SplitsConfig(
            train_end_date=raw.splits.train_end_date,
            val_end_date=raw.splits.val_end_date,
            test_end_date=raw.splits.test_end_date,
        ),
        label=LabelConfig(
            prediction_window_days=raw.label.prediction_window_days,
            include_pending_ceases=raw.label.include_pending_ceases,
        ),
        model=ModelConfig(
            type=raw.model.type,
            registry_name=raw.model.registry_name,
        ),
        scoring=ScoringConfig(
            high_risk_threshold=raw.scoring.high_risk_threshold,
            medium_risk_threshold=raw.scoring.medium_risk_threshold,
            top_n_drivers=raw.scoring.top_n_drivers,
        ),
        mlflow=MLflowConfig(
            tracking_uri=raw.mlflow.tracking_uri,
            experiment_name=raw.mlflow.experiment_name,
        ),
    )


# Module-level singleton — import this in other modules
cfg: Config = load_config()
