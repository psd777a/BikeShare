import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import pipeline
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import mean_squared_error, r2_score


def run_training() -> None:
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.appConfig.training_data_file)

    # divide train and test
    if config.modelConfig.time_based:
        print("Doing Time Based Split")
        data.sort_values(by=config.modelConfig.split_var, inplace=True, ascending=True)
        data.reset_index(drop=True, inplace=True)
        train, test = np.split(
            data, [int((1 - config.modelConfig.test_size) * len(data))]
        )
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        train_X, train_y = (
            train.drop([config.modelConfig.target], axis=1),
            train[config.modelConfig.target].values,
        )
        test_X, test_y = (
            test.drop([config.modelConfig.target], axis=1),
            test[config.modelConfig.target].values,
        )

    else:
        train_X, test_X, train_y, test_y = train_test_split(
            data[config.modelConfig.features],  # predictors
            data[config.modelConfig.target],
            test_size=config.modelConfig.test_size,
            random_state=3,
        )

    # Pipeline fitting
    pipeline.fit(train_X, train_y)
    y_pred = pipeline.predict(test_X)

    # printing the score
    print("RMSE:", np.sqrt(mean_squared_error(test_y, y_pred)))
    print("R2:", r2_score(test_y, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist=pipeline)


if __name__ == "__main__":
    run_training()
