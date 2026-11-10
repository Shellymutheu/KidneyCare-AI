import joblib
import shap
import pandas as pd
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


# ============================================================
# HUMAN-READABLE FEATURE INFORMATION
# ============================================================

FEATURE_INFO = {
    "age": {
        "name": "Age",
        "unit": "years",
        "description": "The patient's age."
    },

    "bp": {
        "name": "Blood Pressure",
        "unit": "mmHg",
        "description": "The patient's blood pressure measurement."
    },

    "sg": {
        "name": "Urine Specific Gravity",
        "unit": "",
        "description": "A measure of how concentrated the urine is."
    },

    "al": {
        "name": "Urine Albumin",
        "unit": "",
        "description": "The amount of albumin detected in the urine."
    },

    "su": {
        "name": "Urine Sugar",
        "unit": "",
        "description": "The amount of sugar detected in the urine."
    },

    "rbc": {
        "name": "Red Blood Cells in Urine",
        "unit": "",
        "description": "Indicates whether red blood cells were detected in the urine."
    },

    "pc": {
        "name": "Pus Cells in Urine",
        "unit": "",
        "description": "Indicates whether pus cells were detected in the urine."
    },

    "pcc": {
        "name": "Pus Cell Clumps",
        "unit": "",
        "description": "Indicates whether pus cell clumps were detected."
    },

    "ba": {
        "name": "Bacteria in Urine",
        "unit": "",
        "description": "Indicates whether bacteria were detected in the urine."
    },

    "bgr": {
        "name": "Blood Glucose",
        "unit": "mg/dL",
        "description": "The patient's blood glucose level."
    },

    "bu": {
        "name": "Blood Urea",
        "unit": "mg/dL",
        "description": "The amount of urea measured in the blood."
    },

    "sc": {
        "name": "Serum Creatinine",
        "unit": "mg/dL",
        "description": "A blood measurement commonly used when assessing kidney function."
    },

    "sod": {
        "name": "Sodium",
        "unit": "mEq/L",
        "description": "The amount of sodium in the blood."
    },

    "pot": {
        "name": "Potassium",
        "unit": "mEq/L",
        "description": "The amount of potassium in the blood."
    },

    "hemo": {
        "name": "Hemoglobin",
        "unit": "g/dL",
        "description": "A measurement related to the blood's ability to carry oxygen."
    },

    "pcv": {
        "name": "Packed Cell Volume",
        "unit": "%",
        "description": "The percentage of blood made up of red blood cells."
    },

    "wbcc": {
        "name": "White Blood Cell Count",
        "unit": "",
        "description": "The number of white blood cells in the blood."
    },

    "rbcc": {
        "name": "Red Blood Cell Count",
        "unit": "",
        "description": "The number of red blood cells in the blood."
    },

    "htn": {
        "name": "Hypertension",
        "unit": "",
        "description": "Indicates whether the patient has high blood pressure."
    },

    "dm": {
        "name": "Diabetes",
        "unit": "",
        "description": "Indicates whether the patient has diabetes."
    },

    "cad": {
        "name": "Coronary Artery Disease",
        "unit": "",
        "description": "Indicates whether coronary artery disease is present."
    },

    "appet": {
        "name": "Appetite",
        "unit": "",
        "description": "The patient's reported appetite."
    },

    "pe": {
        "name": "Pedal Edema",
        "unit": "",
        "description": "Indicates whether swelling of the feet or legs is present."
    },

    "ane": {
        "name": "Anemia",
        "unit": "",
        "description": "Indicates whether anemia is present."
    }
}


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

def load_explainability_components():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_explanation_input(data):

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
# GET ORIGINAL FEATURE NAME
# ============================================================

def get_original_feature_name(feature_name):

    feature_name = str(feature_name)

    # Remove preprocessing prefix such as:
    # numerical__
    # categorical__
    if "__" in feature_name:
        feature_name = feature_name.split("__", 1)[1]

    numerical_features = [
        "age",
        "bp",
        "sg",
        "al",
        "su",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc"
    ]

    categorical_features = [
        "rbc",
        "pc",
        "pcc",
        "ba",
        "htn",
        "dm",
        "cad",
        "appet",
        "pe",
        "ane"
    ]

    # Handle names such as numerical_sc
    if feature_name.startswith("numerical_"):

        original_name = feature_name.replace(
            "numerical_",
            "",
            1
        )

        if original_name in numerical_features:
            return original_name

    # Handle names such as categorical_htn
    if feature_name.startswith("categorical_"):

        original_name = feature_name.replace(
            "categorical_",
            "",
            1
        )

        if original_name in categorical_features:
            return original_name

    # Direct numerical feature
    if feature_name in numerical_features:
        return feature_name

    # Direct categorical feature
    if feature_name in categorical_features:
        return feature_name

    # Handle one-hot encoded categorical features
    for feature in categorical_features:

        if feature_name.startswith(
            feature + "_"
        ):
            return feature

    # Fallback
    return feature_name


# ============================================================
# HUMAN-READABLE FEATURE NAME
# ============================================================

def get_readable_feature_name(feature_name):

    original_name = get_original_feature_name(
        feature_name
    )

    if original_name in FEATURE_INFO:

        return FEATURE_INFO[
            original_name
        ]["name"]

    return str(
        feature_name
    ).replace(
        "_",
        " "
    ).title()


# ============================================================
# FORMAT PATIENT VALUE
# ============================================================

def format_feature_value(
    feature_name,
    feature_value,
    original_data
):

    original_name = get_original_feature_name(
        feature_name
    )

    value = original_data.get(
        original_name
    )

    if value is not None:

        try:

            if pd.isna(value):

                return "Not available"

        except Exception:

            pass

        if isinstance(
            value,
            float
        ):

            return f"{value:.2f}"

        return str(value)

    return str(feature_value)


# ============================================================
# SHAP INFLUENCE STRENGTH
# ============================================================

def get_influence_strength(
    shap_value
):

    magnitude = abs(
        float(shap_value)
    )

    if magnitude < 0.10:

        return "Very small influence"

    elif magnitude < 0.30:

        return "Small influence"

    elif magnitude < 0.60:

        return "Moderate influence"

    else:

        return "Strong influence"


# ============================================================
# SIMPLE EFFECT
# ============================================================

def get_simple_effect(
    shap_value,
    prediction
):

    value = float(
        shap_value
    )

    strength = get_influence_strength(
        value
    )

    if value > 0:

        if prediction == "CKD":

            direction = (
                "This factor increased the "
                "model's likelihood of CKD."
            )

        else:

            direction = (
                "This factor increased the "
                "model's likelihood of the "
                "predicted result."
            )

    else:

        if prediction == "CKD":

            direction = (
                "This factor reduced the "
                "model's likelihood of CKD."
            )

        else:

            direction = (
                "This factor reduced the "
                "model's likelihood of the "
                "predicted result."
            )

    return (
        f"{strength}. {direction}"
    )


# ============================================================
# FEATURE DESCRIPTION
# ============================================================

def get_feature_description(
    feature_name
):

    original_name = get_original_feature_name(
        feature_name
    )

    if original_name in FEATURE_INFO:

        return FEATURE_INFO[
            original_name
        ]["description"]

    return (
        "This patient factor was considered "
        "by the machine-learning model."
    )


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(
    data,
    prediction="CKD"
):

    model, preprocessor = (
        load_explainability_components()
    )

    input_df = prepare_explanation_input(
        data
    )

    processed_input = (
        preprocessor.transform(
            input_df
        )
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            processed_input
        )
    )

    # Handle different SHAP output formats
    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) > 1:

            shap_for_ckd = (
                shap_values[1][0]
            )

        else:

            shap_for_ckd = (
                shap_values[0]
            )

    else:

        if len(
            shap_values.shape
        ) == 3:

            shap_for_ckd = (
                shap_values[0, :, 1]
            )

        else:

            shap_for_ckd = (
                shap_values[0]
            )

    # Create explanation table
    explanation = pd.DataFrame({

        "feature": feature_names,

        "shap_value": shap_for_ckd,

        "feature_value": processed_input[0]
    })

    # Absolute SHAP importance
    explanation[
        "importance"
    ] = explanation[
        "shap_value"
    ].abs()

    # Contribution
    explanation[
        "contribution"
    ] = explanation[
        "shap_value"
    ].apply(
        lambda value:
        "Increases CKD likelihood"
        if value > 0
        else
        "Decreases CKD likelihood"
    )

    # Human-readable name
    explanation[
        "readable_feature"
    ] = explanation[
        "feature"
    ].apply(
        get_readable_feature_name
    )

    # Original patient value
    explanation[
        "patient_value"
    ] = explanation.apply(
        lambda row:
        format_feature_value(
            row["feature"],
            row["feature_value"],
            data
        ),
        axis=1
    )

    # Influence strength
    explanation[
        "influence"
    ] = explanation[
        "shap_value"
    ].apply(
        get_influence_strength
    )

    # Simple explanation
    explanation[
        "simple_explanation"
    ] = explanation[
        "shap_value"
    ].apply(
        lambda value:
        get_simple_effect(
            value,
            prediction
        )
    )

    # Description
    explanation[
        "description"
    ] = explanation[
        "feature"
    ].apply(
        get_feature_description
    )

    # Sort from most important to least important
    explanation = explanation.sort_values(
        "importance",
        ascending=False
    ).reset_index(
        drop=True
    )

    return explanation