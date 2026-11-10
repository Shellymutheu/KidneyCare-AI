import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath="data/kidney_disease.csv"):
    """
    Load the CKD dataset.
    """

    df = pd.read_csv(filepath)

    # Remove whitespace from column names
    df.columns = df.columns.str.strip()

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):
    """
    Clean inconsistent values and prepare the target variable.
    """

    df = df.copy()

    # Remove leading/trailing whitespace from string values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype(str).str.strip()

    # Replace common missing-value representations
    missing_values = ["?", "", "nan", "None", "null"]

    df = df.replace(missing_values, np.nan)

    # Normalize target labels
    df["class"] = df["class"].str.lower().str.strip()

    # Convert target to binary
    df["class"] = df["class"].replace({
        "ckd": 1,
        "notckd": 0
    })

    # Remove rows where target is still missing
    df = df.dropna(subset=["class"])

    # Convert target to integer
    df["class"] = df["class"].astype(int)

    return df


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

def prepare_features(df):
    """
    Separate predictors from the target variable.
    """

    X = df.drop("class", axis=1)
    y = df["class"]

    return X, y


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

def get_column_types(X):
    """
    Identify numerical and categorical features.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numerical_features, categorical_features


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(X):
    """
    Create preprocessing pipeline for numerical and
    categorical clinical variables.
    """

    numerical_features, categorical_features = get_column_types(X)

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


# ============================================================
# TRAIN / TEST DATA
# ============================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("KIDNEYCARE AI - DATA PREPROCESSING TEST")
    print("=" * 60)

    # Load
    df = load_data()

    print("\nOriginal dataset:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Clean
    df = clean_data(df)

    print("\nAfter cleaning:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Target distribution
    print("\nTarget distribution:")
    print(df["class"].value_counts())

    # Features
    X, y = prepare_features(df)

    print("\nFeature columns:")
    print(X.columns.tolist())

    # Column types
    numerical, categorical = get_column_types(X)

    print("\nNumerical features:")
    print(numerical)

    print("\nCategorical features:")
    print(categorical)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTrain/Test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Preprocessor
    preprocessor = create_preprocessor(X_train)

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    print("\nProcessed data:")
    print(f"Training shape: {X_train_processed.shape}")
    print(f"Testing shape: {X_test_processed.shape}")

    print("\n" + "=" * 60)
    print("PREPROCESSING SUCCESSFUL")
    print("=" * 60)