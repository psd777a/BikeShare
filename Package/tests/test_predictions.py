import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.predict import make_prediction
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

def test_make_prediction(sample_data_pull):
    #Given
    num_pred = 3476
    
    #When
    result = make_prediction(input_data=sample_data_pull[0])
    
    #Then
    pred = result.get('predictions')
    assert isinstance(pred, np.ndarray)
    assert isinstance(pred[0], np.float64)
    assert result.get("errors") is None
    assert len(pred) == num_pred
    
    test_y = sample_data_pull[1]
    y_pred = list(pred)
    rmse = np.sqrt(mean_squared_error(test_y, y_pred))
    r2 =  r2_score(test_y, y_pred)
    
    assert r2>0.8
    assert rmse < 1

