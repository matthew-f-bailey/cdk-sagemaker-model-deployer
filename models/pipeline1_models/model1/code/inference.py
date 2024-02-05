from pathlib import Path
from io import StringIO

import xgboost as xgb
import pandas as pd


def model_fn(model_dir):

    classifier = xgb.XGBClassifier()
    classifier.load_model(Path(model_dir, "xgb-trained.json"))
    return classifier


def input_fn(input_data: str, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """
    header, data = input_data.split("\n", 1)

    # Remove csv encoding stuff
    header = [
        x.replace("#015", "").replace("\r", "") for x in header.split(",") if x != ""
    ]
    data = data.replace("#015", "")

    print("Header:", header)
    print("Data:", data)
    return pd.read_csv(StringIO(data), names=header)


def predict_fn(data: tuple, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.

    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn

    Returns: a prediction
    """

    return model.predict(data[away_model.feature_names_in_])


def output_fn(prediction: pd.DataFrame, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns: output data serialized
    """
    return prediction.to_csv(None, index=False)
