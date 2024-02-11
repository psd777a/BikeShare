import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import (
    year_month,
    WeekdayImputer,
    WeathersitImputer,
    Mapper,
    WeekdayOneHotEncoder,
)


def test_yr_mth(sample_data_pull):
    # Given
    mod = year_month(variable=config.modelConfig.dteday_var)
    assert "Year" not in sample_data_pull[0].columns
    assert "Month" not in sample_data_pull[0].columns

    # When
    out = mod.fit(sample_data_pull[0]).transform(sample_data_pull[0])

    # Then
    assert "Year" in out.columns
    assert "Month" in out.columns


def test_weekday(sample_data_pull):
    # Given
    mod = WeekdayImputer(
        variable=config.modelConfig.weekday_var, date_var=config.modelConfig.dteday_var
    )
    assert sample_data_pull[0][config.modelConfig.weekday_var].isna().sum() != 0

    # When
    sample_data_pull[0][config.modelConfig.dteday_var] = pd.to_datetime(
        sample_data_pull[0][config.modelConfig.dteday_var], format="%Y-%m-%d"
    )
    out = mod.fit(sample_data_pull[0]).transform(sample_data_pull[0])

    # Then
    assert out[config.modelConfig.weekday_var].isna().sum() == 0


def test_weathersit(sample_data_pull):
    # Given
    mod = WeathersitImputer(variable=config.modelConfig.weathersit_var)
    assert sample_data_pull[0][config.modelConfig.weathersit_var].isna().sum() != 0

    # When
    out = mod.fit(sample_data_pull[0]).transform(sample_data_pull[0])

    # Then
    assert out[config.modelConfig.weathersit_var].isna().sum() == 0


def test_map_season(sample_data_pull):
    # Given
    mod = Mapper(config.modelConfig.season_var, config.modelConfig.season_mappings)
    assert sample_data_pull[0].loc[306, "season"] == "fall"

    # When
    out = mod.fit(sample_data_pull[0]).transform(sample_data_pull[0])

    # Then
    assert out.loc[306, "season"] == 1


def test_onehot(sample_data_pull):
    # Given
    mod = WeekdayOneHotEncoder(variable=config.modelConfig.weekday_var)
    assert sample_data_pull[0].loc[1567, "weekday"] == "Thu"

    # When
    out = mod.fit(sample_data_pull[0]).transform(sample_data_pull[0])

    # Then
    assert out.loc[1567, "weekday_Thu"] == 1.0
