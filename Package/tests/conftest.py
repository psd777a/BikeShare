"""
    Here we are defining a fixture for pytest and then sample_input_data
    function is passed to each test case. The sample_input_data function returns the
    test set with the target which is used further for feature testing and prediction.
"""

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import numpy as np

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_dataset


@pytest.fixture(scope="session")
def sample_data_pull():
    data = load_dataset(file_name=config.appConfig.training_data_file)

    data.sort_values(by=config.modelConfig.split_var, inplace=True, ascending=True)
    data.reset_index(drop=True, inplace=True)
    _, test = np.split(data, [int((1 - config.modelConfig.test_size) * len(data))])
    test.reset_index(drop=True, inplace=True)
    test_X, test_y = (
        test.drop([config.modelConfig.target], axis=1),
        test[config.modelConfig.target].values,
    )

    return test_X, test_y
