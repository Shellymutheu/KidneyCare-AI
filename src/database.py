import sqlite3
from pathlib import Path
from datetime import datetime
import hashlib
import uuid


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "kidneycare.db"

DATABASE_DIR.mkdir(exist_ok=True)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """Create and return a SQLite database connection."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():
    """Create all KidneyCare AI database tables."""

    connection = get_connection()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # USERS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Healthcare Professional',
            created_at TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # PATIENTS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            registration_date TEXT NOT NULL,

            FOREIGN KEY (user_id)
                REFERENCES users(user_id)
                ON DELETE CASCADE
        )
    """)

    # --------------------------------------------------------
    # MEDICAL RECORDS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_records (
            record_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,

            age REAL,
            bp REAL,
            sg REAL,
            al REAL,
            su REAL,
            rbc TEXT,
            pc TEXT,
            pcc TEXT,
            ba TEXT,
            bgr REAL,
            bu REAL,
            sc REAL,
            sod REAL,
            pot REAL,
            hemo REAL,
            pcv REAL,
            wbcc REAL,
            rbcc REAL,
            htn TEXT,
            dm TEXT,
            cad TEXT,
            appet TEXT,
            pe TEXT,
            ane TEXT,

            recorded_at TEXT NOT NULL,

            FOREIGN KEY (patient_id)
                REFERENCES patients(patient_id)
                ON DELETE CASCADE
        )
    """)

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL,
            model_name TEXT NOT NULL,
            predicted_at TEXT NOT NULL,

            FOREIGN KEY (patient_id)
                REFERENCES patients(patient_id)
                ON DELETE CASCADE,

            FOREIGN KEY (record_id)
                REFERENCES medical_records(record_id)
                ON DELETE CASCADE
        )
    """)

    # --------------------------------------------------------
    # SHAP EXPLANATIONS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shap_explanations (
            explanation_id TEXT PRIMARY KEY,
            prediction_id TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            shap_value REAL NOT NULL,
            feature_value TEXT,
            contribution TEXT,
            created_at TEXT NOT NULL,

            FOREIGN KEY (prediction_id)
                REFERENCES predictions(prediction_id)
                ON DELETE CASCADE
        )
    """)

    # --------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            prediction_id TEXT NOT NULL,
            report_title TEXT NOT NULL,
            report_content TEXT NOT NULL,
            generated_at TEXT NOT NULL,

            FOREIGN KEY (patient_id)
                REFERENCES patients(patient_id)
                ON DELETE CASCADE,

            FOREIGN KEY (prediction_id)
                REFERENCES predictions(prediction_id)
                ON DELETE CASCADE
        )
    """)

    connection.commit()
    connection.close()


# ============================================================
# USER FUNCTIONS
# ============================================================

def create_user(username, full_name, password, role="Healthcare Professional"):
    """Create a new user account."""

    connection = get_connection()
    cursor = connection.cursor()

    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    created_at = datetime.now().isoformat()

    try:
        cursor.execute("""
            INSERT INTO users (
                user_id,
                username,
                full_name,
                password_hash,
                role,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            username,
            full_name,
            password_hash,
            role,
            created_at
        ))

        connection.commit()

        return user_id

    except sqlite3.IntegrityError:
        return None

    finally:
        connection.close()


def authenticate_user(username, password):
    """Authenticate a user."""

    connection = get_connection()
    cursor = connection.cursor()

    password_hash = hash_password(password)

    cursor.execute("""
        SELECT *
        FROM users
        WHERE username = ?
        AND password_hash = ?
    """, (username, password_hash))

    user = cursor.fetchone()

    connection.close()

    return user


# ============================================================
# PATIENT FUNCTIONS
# ============================================================

def create_patient(
    user_id,
    full_name,
    age,
    gender,
    phone
):
    """Register a new patient."""

    connection = get_connection()
    cursor = connection.cursor()

    patient_id = str(uuid.uuid4())
    registration_date = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO patients (
            patient_id,
            user_id,
            full_name,
            age,
            gender,
            phone,
            registration_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        user_id,
        full_name,
        age,
        gender,
        phone,
        registration_date
    ))

    connection.commit()
    connection.close()

    return patient_id


def get_patients(user_id):
    """Get patients registered by a user."""

    connection = get_connection()

    patients = connection.execute("""
        SELECT *
        FROM patients
        WHERE user_id = ?
        ORDER BY registration_date DESC
    """, (user_id,)).fetchall()

    connection.close()

    return patients


def get_patient(patient_id):
    """Get a single patient."""

    connection = get_connection()

    patient = connection.execute("""
        SELECT *
        FROM patients
        WHERE patient_id = ?
    """, (patient_id,)).fetchone()

    connection.close()

    return patient


# ============================================================
# MEDICAL RECORD FUNCTIONS
# ============================================================

def create_medical_record(patient_id, data):
    """Store clinical and laboratory data."""

    connection = get_connection()
    cursor = connection.cursor()

    record_id = str(uuid.uuid4())
    recorded_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO medical_records (
            record_id,
            patient_id,
            age,
            bp,
            sg,
            al,
            su,
            rbc,
            pc,
            pcc,
            ba,
            bgr,
            bu,
            sc,
            sod,
            pot,
            hemo,
            pcv,
            wbcc,
            rbcc,
            htn,
            dm,
            cad,
            appet,
            pe,
            ane,
            recorded_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        record_id,
        patient_id,
        data.get("age"),
        data.get("bp"),
        data.get("sg"),
        data.get("al"),
        data.get("su"),
        data.get("rbc"),
        data.get("pc"),
        data.get("pcc"),
        data.get("ba"),
        data.get("bgr"),
        data.get("bu"),
        data.get("sc"),
        data.get("sod"),
        data.get("pot"),
        data.get("hemo"),
        data.get("pcv"),
        data.get("wbcc"),
        data.get("rbcc"),
        data.get("htn"),
        data.get("dm"),
        data.get("cad"),
        data.get("appet"),
        data.get("pe"),
        data.get("ane"),
        recorded_at
    ))

    connection.commit()
    connection.close()

    return record_id


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def create_prediction(
    patient_id,
    record_id,
    prediction,
    probability,
    model_name
):
    """Store a prediction."""

    connection = get_connection()
    cursor = connection.cursor()

    prediction_id = str(uuid.uuid4())
    predicted_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO predictions (
            prediction_id,
            patient_id,
            record_id,
            prediction,
            probability,
            model_name,
            predicted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        patient_id,
        record_id,
        prediction,
        probability,
        model_name,
        predicted_at
    ))

    connection.commit()
    connection.close()

    return prediction_id


# ============================================================
# SHAP FUNCTIONS
# ============================================================

def save_shap_explanation(
    prediction_id,
    feature_name,
    shap_value,
    feature_value,
    contribution
):
    """Store one SHAP feature explanation."""

    connection = get_connection()
    cursor = connection.cursor()

    explanation_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO shap_explanations (
            explanation_id,
            prediction_id,
            feature_name,
            shap_value,
            feature_value,
            contribution,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        explanation_id,
        prediction_id,
        feature_name,
        shap_value,
        str(feature_value),
        contribution,
        created_at
    ))

    connection.commit()
    connection.close()

    return explanation_id


# ============================================================
# REPORT FUNCTIONS
# ============================================================

def create_report(
    patient_id,
    prediction_id,
    report_title,
    report_content
):
    """Store a generated report."""

    connection = get_connection()
    cursor = connection.cursor()

    report_id = str(uuid.uuid4())
    generated_at = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO reports (
            report_id,
            patient_id,
            prediction_id,
            report_title,
            report_content,
            generated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        report_id,
        patient_id,
        prediction_id,
        report_title,
        report_content,
        generated_at
    ))

    connection.commit()
    connection.close()

    return report_id


# ============================================================
# INITIALIZE DATABASE WHEN RUN DIRECTLY
# ============================================================
# ============================================================
# PREDICTION HISTORY
# ============================================================

def get_prediction_history(user_id):
    """
    Get all predictions belonging to patients
    registered by the logged-in user.
    """

    connection = get_connection()

    predictions = connection.execute("""
        SELECT
            p.prediction_id,
            p.prediction,
            p.probability,
            p.model_name,
            p.predicted_at,
            pt.patient_id,
            pt.full_name
        FROM predictions p
        JOIN patients pt
            ON p.patient_id = pt.patient_id
        WHERE pt.user_id = ?
        ORDER BY p.predicted_at DESC
    """, (user_id,)).fetchall()

    connection.close()

    return predictions


def get_prediction_details(prediction_id):
    """
    Get detailed information about a prediction.
    """

    connection = get_connection()

    prediction = connection.execute("""
        SELECT
            p.*,
            pt.full_name AS patient_name,
            pt.age AS patient_age,
            pt.gender AS patient_gender
        FROM predictions p
        JOIN patients pt
            ON p.patient_id = pt.patient_id
        WHERE p.prediction_id = ?
    """, (prediction_id,)).fetchone()

    connection.close()

    return prediction


def get_shap_explanations(prediction_id):
    """
    Retrieve SHAP explanations for a prediction.
    """

    connection = get_connection()

    explanations = connection.execute("""
        SELECT
            feature_name,
            shap_value,
            feature_value,
            contribution
        FROM shap_explanations
        WHERE prediction_id = ?
        ORDER BY ABS(shap_value) DESC
    """, (prediction_id,)).fetchall()

    connection.close()

    return explanations

if __name__ == "__main__":

    initialize_database()

    print("KidneyCare AI database initialized successfully.")
    print(f"Database location: {DATABASE_PATH}")