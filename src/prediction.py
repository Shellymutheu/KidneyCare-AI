import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

def load_preprocessor():

    return joblib.load(PREPROCESSOR_PATH)


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(data):

    """
    Convert patient input dictionary into a DataFrame
    using the same feature names used during training.
    """

    feature_order = [
        "age",
        "bp",
        "sg",
        "al",
        "su",
        "rbc",
        "pc",
        "pcc",
        "ba",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc",
        "htn",
        "dm",
        "cad",
        "appet",
        "pe",
        "ane"
    ]

    row = {}

    for feature in feature_order:
        row[feature] = data.get(feature)

    return pd.DataFrame([row])


# ============================================================
# MAKE PREDICTION
# ============================================================

def predict_ckd(data):

    model = load_model()
    preprocessor = load_preprocessor()

    input_df = prepare_input(data)

    # Apply the SAME preprocessing used during training
    processed_input = preprocessor.transform(input_df)

    prediction = model.predict(processed_input)[0]

    probabilities = model.predict_proba(processed_input)[0]

    # Probability of CKD (class 1)
    ckd_probability = float(probabilities[1])

    if prediction == 1:
        result = "CKD"
    else:
        result = "NOT CKD"

    return {
        "prediction": result,
        "prediction_value": int(prediction),
        "probability": ckd_probability,
        "model": "Random Forest"
    }