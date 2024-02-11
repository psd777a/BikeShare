import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))

from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

import bikeshare_model

# Project Directories
PACKAGE_ROOT = Path(bikeshare_model.__file__).resolve().parent

ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    Same Names as in config.yml
    """

    package_name: str
    training_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    Same Names as in config.yml
    """

    target: str
    features: List[str]
    dteday_var: str
    weekday_var: str
    weathersit_var: str
    temp_var: str
    hum_var: str
    atemp_var: str
    windspeed_var: str
    yr_var: str
    month_var: str
    holiday_var: str
    workingday_var: str
    season_var: str
    hr_var: str

    month_mappings: Dict[str, int]
    year_mappings: Dict[int, int]
    holiday_mappings: Dict[str, int]
    workingday_mappings: Dict[str, int]
    season_mappings: Dict[str, int]
    weathersit_mappings: Dict[str, int]
    hour_mappings: Dict[str, int]

    test_size: float
    time_based: bool
    split_var: str
    penalty: str
    alpha: float


class Config(BaseModel):
    """Master config object."""

    appConfig: AppConfig
    modelConfig: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml().data

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        modelConfig=ModelConfig(**parsed_config),
        appConfig=AppConfig(**parsed_config),
    )

    return _config


config = create_and_validate_config()
