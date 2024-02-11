import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(parent))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import (
    year_month,
    WeekdayImputer,
    WeathersitImputer,
    Mapper,
    OutlierHandler,
    WeekdayOneHotEncoder,
)

pipeline = Pipeline(
    [
        ##==========Extract Month and Year======##
        ("dteday", year_month(variable=config.modelConfig.dteday_var)),
        ##==========Imputer======##
        (
            "weekday",
            WeekdayImputer(
                variable=config.modelConfig.weekday_var,
                date_var=config.modelConfig.dteday_var,
            ),
        ),
        ("weathersit", WeathersitImputer(variable=config.modelConfig.weathersit_var)),
        ##==========Mapper======##
        ("map_yr", Mapper(config.modelConfig.yr_var, config.modelConfig.year_mappings)),
        (
            "map_holiday",
            Mapper(config.modelConfig.holiday_var, config.modelConfig.holiday_mappings),
        ),
        (
            "map_workingday",
            Mapper(
                config.modelConfig.workingday_var,
                config.modelConfig.workingday_mappings,
            ),
        ),
        (
            "map_season",
            Mapper(config.modelConfig.season_var, config.modelConfig.season_mappings),
        ),
        (
            "map_weathersit",
            Mapper(
                config.modelConfig.weathersit_var,
                config.modelConfig.weathersit_mappings,
            ),
        ),
        (
            "map_month",
            Mapper(config.modelConfig.month_var, config.modelConfig.month_mappings),
        ),
        ("map_hr", Mapper(config.modelConfig.hr_var, config.modelConfig.hour_mappings)),
        ##=========Outlier======##
        ("out_temp", OutlierHandler(variable=config.modelConfig.temp_var)),
        ("out_atemp", OutlierHandler(variable=config.modelConfig.atemp_var)),
        ("out_hum", OutlierHandler(variable=config.modelConfig.hum_var)),
        ("out_windspeed", OutlierHandler(variable=config.modelConfig.windspeed_var)),
        ##=========OneHot======##
        (
            "onehot_weekday",
            WeekdayOneHotEncoder(variable=config.modelConfig.weekday_var),
        ),
        ##=========Scalling======##
        ("scaler", StandardScaler()),
        ##=========Model======##
        (
            "model",
            SGDRegressor(
                penalty=config.modelConfig.penalty, alpha=config.modelConfig.alpha
            ),
        ),
    ]
)
