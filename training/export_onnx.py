import numpy as np
import onnx
from xgboost import XGBRegressor, XGBClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType


def export_regressor(model_path: str, onnx_path: str):
    model = XGBRegressor()
    model.load_model(model_path)
    initial_types = [("X", OnnxFloatTensorType([None, 151]))]
    onnx_model = convert_xgboost(model, initial_types=initial_types)
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    import os
    print(f"Exported {onnx_path}: {os.path.getsize(onnx_path)/1e6:.2f} MB")


def export_classifier(model_path: str, onnx_path: str):
    model = XGBClassifier()
    model.load_model(model_path)
    initial_types = [("X", OnnxFloatTensorType([None, 151]))]
    onnx_model = convert_xgboost(model, initial_types=initial_types)
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    import os
    print(f"Exported {onnx_path}: {os.path.getsize(onnx_path)/1e6:.2f} MB")


if __name__ == "__main__":
    export_regressor("models/importance_xgb.json", "models/importance.onnx")
    export_regressor("models/urgency_xgb.json", "models/urgency.onnx")
    export_classifier("models/category_xgb.json", "models/category.onnx")
    print("All models exported successfully.")
