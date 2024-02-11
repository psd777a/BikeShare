import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


class DataInputSchema(BaseModel):
    dteday: Optional[Union[str, datetime]]
    season: Optional[str]
    hr: Optional[str]
    holiday: Optional[str]
    weekday: Optional[str]
    workingday: Optional[str]
    weathersit: Optional[str]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]
    casual: Optional[int]
    registered: Optional[int]


class MultipleDataInput(BaseModel):
    inputs: List[DataInputSchema]


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    errors = None
    try:
        # to_dict(orient="records") converts the dataframe into a list where each element is
        # a dictionary of column_name and value in that row

        MultipleDataInput(
            inputs=input_df.replace({np.NaN: None}).to_dict(orient="records")
        )

    except ValidationError as error:
        errors = error.json()

    return input_df, errors
