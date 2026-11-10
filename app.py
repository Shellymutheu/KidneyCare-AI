import streamlit as st
import pandas as pd
from src.database import (
    initialize_database,
    create_user,
    authenticate_user,
    create_patient,
    get_patients,
    create_medical_record,
    create_prediction,
    get_prediction_history,
    get_shap_explanations,
    save_shap_explanation,
    create_report,
)

from src.prediction import predict_ckd
from src.explainability import generate_shap_explanation
from src.report_generator import generate_pdf_report

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chronic Kidney Disease Predictor",
    page_icon="🩺",
    layout="wide",
)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_prediction_id" not in st.session_state:
    st.session_state.last_prediction_id = None

if "last_record_id" not in st.session_state:
    st.session_state.last_record_id = None


# ============================================================
# CUSTOM CSS — MEDICAL BLUE AND TEAL THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ckd-teal: #6ba9b9;
        --ckd-deep: #246b7a;
        --ckd-ink: #203b47;
        --ckd-muted: #607783;
        --ckd-border: #deebef;
    }
    .stApp {
        background: linear-gradient(180deg, #f3f9fb 0%, #ffffff 48%, #f7fbfc 100%);
    }
    .block-container {
        max-width: 1380px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #edf6f8;
        border-right: 1px solid #d8e9ee;
    }
    h1, h2, h3 {
        color: var(--ckd-ink) !important;
        letter-spacing: -0.02em;
    }
    .main-title {
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        text-align: center;
        color: var(--ckd-deep);
        line-height: 1.15;
        margin: 8px 0 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: var(--ckd-muted);
        margin-bottom: 26px;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--ckd-border);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 5px 18px rgba(35, 94, 111, 0.06);
    }
    [data-testid="stMetricValue"] { color: var(--ckd-deep) !important; }
    div[data-testid="stForm"] {
        background: #fff;
        border: 1px solid var(--ckd-border);
        border-radius: 16px;
        padding: 1rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border-color: var(--ckd-teal);
        min-height: 2.6rem;
        font-weight: 650;
    }
    .stButton > button[kind="primary"] {
        background: var(--ckd-deep);
        color: #fff;
        border-color: var(--ckd-deep);
    }
    [data-testid="stDataFrame"] { border-radius: 12px; }
    .ckd-hero {
        background: linear-gradient(120deg, #e5f3f7 0%, #f8fcfd 58%, #d5eaf0 100%);
        border: 1px solid #d7e9ee;
        border-radius: 24px;
        padding: 30px 32px;
        margin: 0 0 22px 0;
    }
    .ckd-eyebrow {
        color: #347f91;
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .12em;
        margin-bottom: 10px;
    }
    .ckd-hero h1 {
        color: #1e3d49 !important;
        font-size: clamp(28px, 3.4vw, 42px);
        line-height: 1.15;
        margin: 0 0 12px 0;
    }
    .ckd-hero p { color: #58717b; font-size: 15px; line-height: 1.7; max-width: 800px; }
    .ckd-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
    .ckd-chip {
        display: inline-block;
        padding: 7px 12px;
        background: rgba(255,255,255,.9);
        border: 1px solid #d5e9ee;
        border-radius: 999px;
        color: #356d7b;
        font-size: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOGGED-IN APPLICATION
# ============================================================

if st.session_state.logged_in:

    user = st.session_state.user

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.markdown(
        """
        <div style="display:flex;gap:12px;align-items:center;padding:4px 0 14px;">
          <div style="width:42px;height:42px;border-radius:13px;background:#d8edf2;color:#246b7a;display:flex;align-items:center;justify-content:center;font-size:24px;">✚</div>
          <div>
            <div style="font-weight:800;font-size:15px;line-height:1.2;color:#245f70;">Chronic Kidney<br> Disease Predictor</div>
            <div style="font-size:11px;color:#6a8791;margin-top:4px;">Clinical support workspace</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.success(
        f"Logged in as: {user['full_name']}"
    )

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Patient Registration",
            "Prediction",
            "Prediction History",
            "Reports",
        ],
    )

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "Logout",
        use_container_width=True,
    ):

        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.last_prediction = None
        st.session_state.last_prediction_id = None
        st.session_state.last_record_id = None

        st.rerun()

    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        st.markdown(
            """
            <section class="ckd-hero">
              <div class="ckd-eyebrow">Clinical decision support · Explainable AI</div>
              <h1>Clearer insights for kidney health.</h1>
              <p>
                Review patient information, explore machine-learning predictions,
                and understand the clinical factors that influenced each result.
                This application is an academic prototype for healthcare professionals.
              </p>
              <div class="ckd-chips">
                <span class="ckd-chip">Patient management</span>
                <span class="ckd-chip">CKD prediction</span>
                <span class="ckd-chip">Explainable results</span>
                <span class="ckd-chip">Reports and history</span>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Welcome, **{user['full_name']}**."
        )

        st.info(
            "Chronic Kidney Disease Predictor is an academic prototype that uses "
            "machine learning to estimate CKD likelihood from "
            "clinical and laboratory data."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "ML Models",
                "5",
            )

        with col2:
            st.metric(
                "Dataset Records",
                "400",
            )

        with col3:
            st.metric(
                "Explainability",
                "SHAP",
            )

        st.markdown("---")

        st.subheader("System Workflow")

        st.write(
            """
            1. Register a patient.
            2. Enter clinical and laboratory information.
            3. Generate a CKD prediction.
            4. View the prediction probability.
            5. Examine SHAP-based feature explanations.
            6. Store the prediction and explanation.
            7. Generate a report.
            """
        )

        st.markdown("---")

        st.subheader("Machine Learning Models")

        model_col1, model_col2 = st.columns(2)

        with model_col1:
            st.write("• Logistic Regression")
            st.write("• Random Forest")
            st.write("• Support Vector Machine")

        with model_col2:
            st.write("• K-Nearest Neighbours")
            st.write("• XGBoost")
            st.write("• SHAP Explainability")

        st.warning(
            "This system is an academic decision-support "
            "prototype and is not a certified medical "
            "diagnostic device."
        )

    # ========================================================
    # PATIENT REGISTRATION
    # ========================================================

    elif page == "Patient Registration":

        st.title("👤 Patient Registration")

        st.write(
            "Register a patient before entering clinical data."
        )

        with st.form("patient_registration_form"):

            full_name = st.text_input(
                "Patient Full Name"
            )

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=30,
            )

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female",
                    "Other",
                ],
            )

            phone = st.text_input(
                "Phone Number"
            )

            submitted = st.form_submit_button(
                "Register Patient",
                use_container_width=True,
            )

            if submitted:

                if not full_name.strip():

                    st.error(
                        "Please enter the patient's name."
                    )

                else:

                    try:

                        patient_id = create_patient(
                            user["user_id"],
                            full_name.strip(),
                            age,
                            gender,
                            phone.strip(),
                        )

                        st.success(
                            "Patient registered successfully."
                        )

                        st.code(
                            f"Patient ID: {patient_id}"
                        )

                    except Exception as error:

                        st.error(
                            f"Patient registration error: {error}"
                        )

    # ========================================================
    # PREDICTION
    # ========================================================

    elif page == "Prediction":

        st.title("🔬 CKD Prediction")

        st.write(
            "Enter the patient's clinical and laboratory "
            "information below."
        )

        patients = get_patients(
            user["user_id"]
        )

        if not patients:

            st.warning(
                "No patients have been registered yet. "
                "Please register a patient first."
            )

        else:

            patient_options = {}

            for patient in patients:

                label = (
                    f"{patient['full_name']} "
                    f"({patient['patient_id'][:8]})"
                )

                patient_options[label] = (
                    patient["patient_id"]
                )

            selected_patient = st.selectbox(
                "Select Patient",
                list(patient_options.keys()),
            )

            patient_id = patient_options[
                selected_patient
            ]

            st.markdown("---")

            st.subheader(
                "Clinical & Laboratory Information"
            )

            col1, col2, col3 = st.columns(3)

            # =================================================
            # COLUMN 1
            # =================================================

            with col1:

                age = st.number_input(
                    "Age",
                    min_value=1.0,
                    max_value=120.0,
                    value=45.0,
                )

                bp = st.number_input(
                    "Blood Pressure",
                    min_value=50.0,
                    max_value=250.0,
                    value=80.0,
                )

                sg = st.number_input(
                    "Specific Gravity",
                    min_value=1.000,
                    max_value=1.030,
                    value=1.020,
                    step=0.005,
                    format="%.3f",
                )

                al = st.number_input(
                    "Albumin",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.0,
                )

                su = st.number_input(
                    "Sugar",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.0,
                )

                bgr = st.number_input(
                    "Blood Glucose Random",
                    min_value=50.0,
                    max_value=500.0,
                    value=120.0,
                )

                bu = st.number_input(
                    "Blood Urea",
                    min_value=1.0,
                    max_value=300.0,
                    value=40.0,
                )

            # =================================================
            # COLUMN 2
            # =================================================

            with col2:

                sc = st.number_input(
                    "Serum Creatinine",
                    min_value=0.1,
                    max_value=20.0,
                    value=1.0,
                )

                sod = st.number_input(
                    "Sodium",
                    min_value=80.0,
                    max_value=200.0,
                    value=140.0,
                )

                pot = st.number_input(
                    "Potassium",
                    min_value=1.0,
                    max_value=15.0,
                    value=4.5,
                )

                hemo = st.number_input(
                    "Hemoglobin",
                    min_value=1.0,
                    max_value=25.0,
                    value=13.0,
                )

                pcv = st.number_input(
                    "Packed Cell Volume",
                    min_value=5.0,
                    max_value=70.0,
                    value=40.0,
                )

                wbcc = st.number_input(
                    "White Blood Cell Count",
                    min_value=1000.0,
                    max_value=30000.0,
                    value=8000.0,
                )

                rbcc = st.number_input(
                    "Red Blood Cell Count",
                    min_value=1.0,
                    max_value=10.0,
                    value=5.0,
                )

            # =================================================
            # COLUMN 3
            # =================================================

            with col3:

                rbc = st.selectbox(
                    "Red Blood Cells",
                    [
                        "normal",
                        "abnormal",
                    ],
                )

                pc = st.selectbox(
                    "Pus Cell",
                    [
                        "normal",
                        "abnormal",
                    ],
                )

                pcc = st.selectbox(
                    "Pus Cell Clumps",
                    [
                        "notpresent",
                        "present",
                    ],
                )

                ba = st.selectbox(
                    "Bacteria",
                    [
                        "notpresent",
                        "present",
                    ],
                )

                htn = st.selectbox(
                    "Hypertension",
                    [
                        "no",
                        "yes",
                    ],
                )

                dm = st.selectbox(
                    "Diabetes Mellitus",
                    [
                        "no",
                        "yes",
                    ],
                )

                cad = st.selectbox(
                    "Coronary Artery Disease",
                    [
                        "no",
                        "yes",
                    ],
                )

                appet = st.selectbox(
                    "Appetite",
                    [
                        "good",
                        "poor",
                    ],
                )

                pe = st.selectbox(
                    "Pedal Edema",
                    [
                        "no",
                        "yes",
                    ],
                )

                ane = st.selectbox(
                    "Anemia",
                    [
                        "no",
                        "yes",
                    ],
                )

            st.markdown("---")

            # =================================================
            # PREDICTION BUTTON
            # =================================================

            if st.button(
                "🔍 Generate CKD Prediction",
                type="primary",
                use_container_width=True,
            ):

                input_data = {
                    "age": age,
                    "bp": bp,
                    "sg": sg,
                    "al": al,
                    "su": su,
                    "rbc": rbc,
                    "pc": pc,
                    "pcc": pcc,
                    "ba": ba,
                    "bgr": bgr,
                    "bu": bu,
                    "sc": sc,
                    "sod": sod,
                    "pot": pot,
                    "hemo": hemo,
                    "pcv": pcv,
                    "wbcc": wbcc,
                    "rbcc": rbcc,
                    "htn": htn,
                    "dm": dm,
                    "cad": cad,
                    "appet": appet,
                    "pe": pe,
                    "ane": ane,
                }

                try:

                    # ========================================
                    # RUN MODEL
                    # ========================================

                    result = predict_ckd(
                        input_data
                    )

                    # ========================================
                    # SAVE MEDICAL RECORD
                    # ========================================

                    record_id = create_medical_record(
                        patient_id,
                        input_data,
                    )

                    # ========================================
                    # SAVE PREDICTION
                    # ========================================

                    prediction_id = create_prediction(
                        patient_id=patient_id,
                        record_id=record_id,
                        prediction=result["prediction"],
                        probability=result["probability"],
                        model_name=result["model"],
                    )

                    # ========================================
                    # SESSION STATE
                    # ========================================

                    st.session_state.last_prediction = result

                    st.session_state.last_prediction_id = (
                        prediction_id
                    )

                    st.session_state.last_record_id = (
                        record_id
                    )

                    # ========================================
                    # DISPLAY RESULT
                    # ========================================

                    st.success(
                        "Prediction generated successfully."
                    )

                    st.markdown(
                        "### Prediction Result"
                    )

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        if result["prediction"] == "CKD":

                            st.error(
                                "Prediction: CKD"
                            )

                        else:

                            st.success(
                                "Prediction: NOT CKD"
                            )

                    with result_col2:

                        st.metric(
                            "Estimated CKD Probability",
                            f"{result['probability'] * 100:.2f}%",
                        )

                    st.caption(
                        f"Model used: {result['model']}"
                    )

                    st.info(
                        "This result is generated by an academic "
                        "machine-learning prototype and should "
                        "not be treated as a medical diagnosis."
                    )

                    # ========================================
                    # SHAP EXPLAINABILITY
                    # ========================================

                    st.markdown("---")

                    st.subheader(
                        "🧠 Why did the model make this prediction?"
                    )

                    try:

                        explanation = (
                            generate_shap_explanation(
                                input_data
                            )
                        )

                        top_features = (
                            explanation.head(10).copy()
                        )

                        top_features = top_features[
                            [
                                "readable_feature",
                                "shap_value",
                                "feature_value",
                            ]
                        ]

                        top_features.columns = [
                            "Patient Factor",
                            "SHAP Value",
                            "Patient Value",
                        ]

                        st.dataframe(
                            top_features,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown(
                            "### Feature Interpretation"
                        )

                        for _, row in explanation.head(5).iterrows():

                            feature = row["readable_feature"]
                            value = row["shap_value"]

                            contribution = row[
                                "contribution"
                            ]

                            if value > 0:

                                st.write(
                                    f"🔴 **{feature}** — "
                                    f"{contribution} "
                                    f"(SHAP: {value:.4f})"
                                )

                            else:

                                st.write(
                                    f"🔵 **{feature}** — "
                                    f"{contribution} "
                                    f"(SHAP: {value:.4f})"
                                )

                        # ====================================
                        # SAVE SHAP VALUES
                        # ====================================

                        for _, row in explanation.head(10).iterrows():

                            save_shap_explanation(
                                prediction_id=prediction_id,
                                feature_name=str(
                                    row["readable_feature"]
                                ),
                                shap_value=float(
                                    row["shap_value"]
                                ),
                                feature_value=str(
                                    row["feature_value"]
                                ),
                                contribution=str(
                                    row["contribution"]
                                ),
                        )

                        st.success(
                            "SHAP explanation saved successfully."
                        )

                    except Exception as shap_error:

                        st.error(
                            f"SHAP explanation error: "
                            f"{shap_error}"
                        )

                except Exception as error:

                    st.error(
                        f"Prediction error: {error}"
                    )

    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    elif page == "Prediction History":

        st.title("📊 Prediction History")

        st.write(
            "Previously generated CKD predictions "
            "for your registered patients."
        )

        try:

            history = get_prediction_history(
                user["user_id"]
            )

            if not history:

                st.info(
                    "No predictions have been generated yet."
                )

            else:

                st.subheader(
                    f"Total Predictions: {len(history)}"
                )

                for prediction in history:

                    prediction_date = (
                        prediction["predicted_at"]
                    )

                    probability = (
                        prediction["probability"] * 100
                    )

                    with st.expander(
                        f"{prediction['full_name']} — "
                        f"{prediction['prediction']} — "
                        f"{probability:.2f}%"
                    ):

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.write("**Patient**")

                            st.write(
                                prediction["full_name"]
                            )

                        with col2:

                            st.write("**Prediction**")

                            if prediction["prediction"] == "CKD":

                                st.error("CKD")

                            else:

                                st.success("NOT CKD")

                        with col3:

                            st.write("**Probability**")

                            st.write(
                                f"{probability:.2f}%"
                            )

                        st.write(
                            f"**Model:** "
                            f"{prediction['model_name']}"
                        )

                        st.write(
                            f"**Date:** "
                            f"{prediction_date}"
                        )

                        st.write(
                            f"**Prediction ID:** "
                            f"{prediction['prediction_id']}"
                        )
                    # ========================================
                    # SHAP EXPLAINABILITY
                    # ========================================

                    st.markdown("---")

                    st.subheader(
                        "🧠 Why did the system make this prediction?"
                    )

                    st.write(
                        "The explanation below shows which patient "
                        "factors had the greatest influence on the "
                        "machine-learning prediction."
                    )

                    st.info(
                        "The system translates the technical SHAP "
                        "calculations into simple language. "
                        "A strong influence does not mean that the "
                        "factor alone caused CKD; it only means that "
                        "the factor had a strong influence on this "
                        "model prediction."
                    )

                    try:

                        # ----------------------------------------
                        # GENERATE SHAP EXPLANATION
                        # ----------------------------------------

                        explanation = (
                            generate_shap_explanation(
                                input_data,
                                prediction=result["prediction"]
                            )
                        )

                        # ----------------------------------------
                        # TOP 10 FEATURES
                        # ----------------------------------------

                        top_features = (
                            explanation
                            .head(10)
                            .copy()
                        )

                        display_table = pd.DataFrame({

                            "Patient Factor":
                                top_features[
                                    "readable_feature"
                                ],

                            "Patient Value":
                                top_features[
                                    "patient_value"
                                ],

                            "Influence":
                                top_features[
                                    "influence"
                                ],

                            "Effect on Prediction":
                                top_features[
                                    "contribution"
                                ]
                        })

                        st.markdown(
                            "### 📊 Main Factors Considered by the Model"
                        )

                        st.dataframe(
                            display_table,
                            use_container_width=True,
                            hide_index=True
                        )

                        # ----------------------------------------
                        # SIMPLE EXPLANATIONS
                        # ----------------------------------------

                        st.markdown(
                            "### 💡 Easy-to-Understand Explanation"
                        )

                        st.write(
                            "The following are the most important "
                            "factors behind this prediction:"
                        )

                        for index, row in (
                            explanation
                            .head(5)
                            .iterrows()
                        ):

                            feature_name = (
                                row[
                                    "readable_feature"
                                ]
                            )

                            patient_value = (
                                row[
                                    "patient_value"
                                ]
                            )

                            influence = (
                                row[
                                    "influence"
                                ]
                            )

                            contribution = (
                                row[
                                    "contribution"
                                ]
                            )

                            description = (
                                row[
                                    "description"
                                ]
                            )

                            # ------------------------------------
                            # POSITIVE CONTRIBUTION
                            # ------------------------------------

                            if (
                                row[
                                    "shap_value"
                                ] > 0
                            ):

                                st.markdown(
                                    f"""
                                    <div style="
                                        padding: 15px;
                                        margin-bottom: 12px;
                                        border-radius: 10px;
                                        border-left: 5px solid #d9534f;
                                        background-color: #fff5f5;
                                    ">

                                    <h4>
                                    🔴 {index + 1}. {feature_name}
                                    </h4>

                                    <p>
                                    <strong>Patient value:</strong>
                                    {patient_value}
                                    </p>

                                    <p>
                                    <strong>Model influence:</strong>
                                    {influence}
                                    </p>

                                    <p>
                                    <strong>Effect:</strong>
                                    This factor
                                    <strong>
                                    increased the likelihood of CKD
                                    </strong>
                                    in the model's prediction.
                                    </p>

                                    <p>
                                    <strong>What it means:</strong>
                                    {description}
                                    </p>

                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # ------------------------------------
                            # NEGATIVE CONTRIBUTION
                            # ------------------------------------

                            else:

                                st.markdown(
                                    f"""
                                    <div style="
                                        padding: 15px;
                                        margin-bottom: 12px;
                                        border-radius: 10px;
                                        border-left: 5px solid #198754;
                                        background-color: #f4fff7;
                                    ">

                                    <h4>
                                    🟢 {index + 1}. {feature_name}
                                    </h4>

                                    <p>
                                    <strong>Patient value:</strong>
                                    {patient_value}
                                    </p>

                                    <p>
                                    <strong>Model influence:</strong>
                                    {influence}
                                    </p>

                                    <p>
                                    <strong>Effect:</strong>
                                    This factor
                                    <strong>
                                    decreased the likelihood of CKD
                                    </strong>
                                    in the model's prediction.
                                    </p>

                                    <p>
                                    <strong>What it means:</strong>
                                    {description}
                                    </p>

                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        # ----------------------------------------
                        # TECHNICAL DETAILS
                        # ----------------------------------------

                        with st.expander(
                            "🔬 Technical SHAP Details"
                        ):

                            st.write(
                                "The values below are the original "
                                "SHAP numerical values used internally "
                                "by the system. They are provided for "
                                "technical review and research purposes."
                            )

                            technical_table = (
                                explanation
                                .head(10)
                                [
                                    [
                                        "feature",
                                        "shap_value",
                                        "importance"
                                    ]
                                ]
                                .copy()
                            )

                            technical_table.columns = [
                                "Technical Feature",
                                "SHAP Value",
                                "Absolute Importance"
                            ]

                            st.dataframe(
                                technical_table,
                                use_container_width=True,
                                hide_index=True
                            )

                        # ----------------------------------------
                        # SAVE SHAP VALUES TO DATABASE
                        # ----------------------------------------

                        from src.database import (
                            save_shap_explanation
                        )

                        for _, row in (
                            explanation
                            .head(10)
                            .iterrows()
                        ):

                            save_shap_explanation(

                                prediction_id=prediction_id,

                                feature_name=str(
                                    row[
                                        "readable_feature"
                                    ]
                                ),

                                shap_value=float(
                                    row[
                                        "shap_value"
                                    ]
                                ),

                                feature_value=str(
                                    row[
                                        "patient_value"
                                    ]
                                ),

                                contribution=str(
                                    row[
                                        "contribution"
                                    ]
                                )
                            )

                        st.success(
                            "Explanation saved successfully."
                        )

                    except Exception as shap_error:

                        st.error(
                            f"SHAP explanation error: {shap_error}"
                        )

        except Exception as history_error:

            st.error(
                f"Unable to load prediction history: "
                f"{history_error}"
            )

    # ========================================================
    # REPORTS
    # ========================================================

    elif page == "Reports":

        st.title("📄 Clinical Prediction Reports")

        st.write(
            "Generate and download a report for a "
            "previously generated CKD prediction."
        )

        try:

            history = get_prediction_history(
                user["user_id"]
            )

            if not history:

                st.info(
                    "No predictions are available. "
                    "Generate a prediction first."
                )

            else:

                st.subheader("Select Prediction")

                report_options = {}

                for prediction in history:

                    label = (
                        f"{prediction['full_name']} | "
                        f"{prediction['prediction']} | "
                        f"{prediction['predicted_at']}"
                    )

                    report_options[label] = prediction

                selected_report = st.selectbox(
                    "Choose a prediction",
                    list(report_options.keys()),
                )

                selected_prediction = report_options[
                    selected_report
                ]

                prediction_id = selected_prediction[
                    "prediction_id"
                ]

                explanations = get_shap_explanations(
                    prediction_id
                )

                st.markdown("---")

                # ============================================
                # REPORT INFORMATION
                # ============================================

                st.subheader("📋 Report Preview")

                patient_name = selected_prediction[
                    "full_name"
                ]

                prediction_result = selected_prediction[
                    "prediction"
                ]

                probability = (
                    selected_prediction["probability"] * 100
                )

                model_name = selected_prediction[
                    "model_name"
                ]

                prediction_date = selected_prediction[
                    "predicted_at"
                ]

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        f"**Patient:** {patient_name}"
                    )

                    st.write(
                        f"**Prediction:** "
                        f"{prediction_result}"
                    )

                    st.write(
                        f"**Probability:** "
                        f"{probability:.2f}%"
                    )

                with col2:

                    st.write(
                        f"**Model:** {model_name}"
                    )

                    st.write(
                        f"**Date:** {prediction_date}"
                    )

                    st.write(
                        f"**Prediction ID:** "
                        f"{prediction_id}"
                    )

                # ============================================
                # SHAP SUMMARY
                # ============================================

                st.markdown("---")

                st.subheader(
                    "🧠 Explainability Summary"
                )

                if explanations:

                    for explanation in explanations[:5]:

                        feature = explanation[
                            "feature_name"
                        ]

                        shap_value = explanation[
                            "shap_value"
                        ]

                        contribution = explanation[
                            "contribution"
                        ]

                        st.write(
                            f"• **{feature}** — "
                            f"{contribution} "
                            f"(SHAP value: "
                            f"{shap_value:.4f})"
                        )

                else:

                    st.info(
                        "No SHAP explanation is available."
                    )

                # ============================================
                # CREATE REPORT CONTENT
                # ============================================

                report_lines = [
                    "KIDNEYCARE AI",
                    "=" * 60,
                    "CHRONIC KIDNEY DISEASE "
                    "PREDICTION REPORT",
                    "=" * 60,
                    "",
                    f"Patient Name: {patient_name}",
                    f"Prediction Date: {prediction_date}",
                    f"Prediction: {prediction_result}",
                    f"Estimated CKD Probability: "
                    f"{probability:.2f}%",
                    f"Machine Learning Model: {model_name}",
                    "",
                    "EXPLAINABILITY SUMMARY",
                    "-" * 60,
                ]

                if explanations:

                    for explanation in explanations[:10]:

                        feature = explanation[
                            "feature_name"
                        ]

                        shap_value = explanation[
                            "shap_value"
                        ]

                        contribution = explanation[
                            "contribution"
                        ]

                        report_lines.append(
                            f"{feature}: "
                            f"{contribution} "
                            f"(SHAP value: "
                            f"{shap_value:.4f})"
                        )

                else:

                    report_lines.append(
                        "No SHAP explanation available."
                    )

                report_lines.extend(
                    [
                        "",
                        "DISCLAIMER",
                        "-" * 60,
                        "Chronic Kidney Disease Predictor is an academic "
                        "clinical decision-support prototype.",
                        "The prediction is generated using "
                        "machine-learning methods.",
                        "This report must not be treated as "
                        "a medical diagnosis.",
                        "Clinical decisions should be made "
                        "by qualified healthcare professionals.",
                        "",
                        "=" * 60,
                        "End of Report",
                        "=" * 60,
                    ]
                )

                report_content = "\n".join(
                    report_lines
                )

                # ============================================
                # REPORT DISPLAY
                # ============================================

                st.text_area(
                    "Generated Report",
                    report_content,
                    height=500,
                )

                # ============================================
                # SAVE REPORT
                # ============================================

                if st.button(
                    "💾 Save Report",
                    type="primary",
                    use_container_width=True,
                ):

                    try:

                        report_id = create_report(
                            patient_id=selected_prediction[
                                "patient_id"
                            ],
                            prediction_id=prediction_id,
                            report_title=(
                                "CKD Prediction Report - "
                                f"{patient_name}"
                            ),
                            report_content=report_content,
                        )

                        st.success(
                            "Report saved successfully."
                        )

                        st.code(
                            f"Report ID: {report_id}"
                        )

                    except Exception as report_error:

                        st.error(
                            f"Unable to save report: "
                            f"{report_error}"
                        )

                # ============================================
                # DOWNLOAD REPORT
                # ============================================

                st.download_button(
                    label="⬇️ Download Report",
                    data=report_content,
                    file_name="KidneyCare_AI_Report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as error:

            st.error(
                f"Unable to generate report: {error}"
            )


# ============================================================
# LOGIN / ACCOUNT CREATION
# ============================================================

else:

    st.markdown(
        '<div class="main-title">🩺 Chronic Kidney Disease Predictor</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Explainable AI-Based Clinical Decision Support System'
        '</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(
        [
            "🔐 Login",
            "📝 Create Account",
        ]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with tab1:

        st.subheader(
            "Healthcare Professional Login"
        )

        username = st.text_input(
            "Username",
            key="login_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True,
        ):

            if not username or not password:

                st.error(
                    "Please enter both username and password."
                )

            else:

                try:

                    logged_user = authenticate_user(
                        username,
                        password,
                    )

                    if logged_user:

                        st.session_state.logged_in = True

                        st.session_state.user = logged_user

                        st.success(
                            "Login successful!"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Invalid username or password."
                        )

                except Exception as error:

                    st.error(
                        f"Login error: {error}"
                    )

    # ========================================================
    # CREATE ACCOUNT
    # ========================================================

    with tab2:

        st.subheader(
            "Create Healthcare Professional Account"
        )

        full_name = st.text_input(
            "Full Name",
            key="register_name",
        )

        username = st.text_input(
            "Username",
            key="register_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            key="register_password",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="register_confirm",
        )

        if st.button(
            "Create Account",
            type="primary",
            use_container_width=True,
        ):

            if (
                not full_name.strip()
                or not username.strip()
                or not password
            ):

                st.error(
                    "Please complete all required fields."
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            elif len(password) < 6:

                st.error(
                    "Password must contain at least 6 characters."
                )

            else:

                try:

                    user_id = create_user(
                        username=username.strip(),
                        full_name=full_name.strip(),
                        password=password,
                    )

                    if user_id:

                        st.success(
                            "Account created successfully. "
                            "You can now log in."
                        )

                    else:

                        st.error(
                            "That username already exists."
                        )

                except Exception as error:

                    st.error(
                        f"Account creation error: {error}"
                    )