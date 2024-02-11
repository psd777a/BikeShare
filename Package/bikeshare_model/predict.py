import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import numpy as np

from typing import Union
from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.validation import validate_inputs

pipeline_name = f"{config.appConfig.pipeline_save_file}_{_version}.pkl"
bike_share_pipe = load_pipeline(file_name=pipeline_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Predict Output using Saved Pipeline"""
    validated_data, error = validate_inputs(input_df=pd.DataFrame(input_data))
    validated_data.reindex(columns=config.modelConfig.features)

    results = {"predictions": None, "version": _version, "errors": error}
    if not error:
        pred = bike_share_pipe.predict(validated_data)
        results["predictions"] = np.floor(pred)
        results["version"] = _version
        print(results)

    return results


if __name__ == "__main__":
    data_in = {
        "dteday": ["2012-11-05"],
        "season": ["winter"],
        "hr": ["6am"],
        "holiday": ["No"],
        "weekday": ["Mon"],
        "workingday": ["Yes"],
        "weathersit": ["Mist"],
        "temp": [6.10],
        "atemp": [3.0014],
        "hum": [49.0],
        "windspeed": [19.0012],
        "casual": [4],
        "registered": [135],
    }

    make_prediction(input_data=data_in)
