# app/model.py
import joblib
import shap
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.joblib"
COLS_PATH = BASE_DIR / "model.joblib.cols"


def load_model():
    """
    Loads pipeline, model, feature names and SHAP explainer
    """
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # ✅ CORRECT feature names AFTER preprocessing
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)

    print("✅ Pipeline, model & SHAP loaded successfully")

    return pipeline, preprocessor, model, feature_names, explainer


def prepare_input(data):
    """
    Convert Pydantic input to DataFrame
    """
    if hasattr(data, "dict"):
        data = data.dict()

    return pd.DataFrame([data])

def get_shap_values(explainer, X_transformed):
    """
    Always returns a 1D numpy array of floats (n_features,)
    """

    shap_output = explainer.shap_values(X_transformed)

    # Binary classifier → class 1
    if isinstance(shap_output, list):
        shap_vals = shap_output[1]
    else:
        shap_vals = shap_output

    # Remove batch dimension (1, n_features) → (n_features,)
    shap_vals = np.asarray(shap_vals).reshape(-1)

    # FORCE python-safe floats
    return shap_vals.astype(float)