import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import (
    load_data,
    clean_data,
    prepare_features,
    split_data,
    create_preprocessor
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/kidney_disease.csv"
MODEL_DIR = "models"


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

print("=" * 70)
print("KIDNEYCARE AI - MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = load_data(DATA_PATH)

print(f"Original dataset shape: {df.shape}")

# Clean data
df = clean_data(df)

print(f"Cleaned dataset shape: {df.shape}")

# Separate features and target
X, y = prepare_features(df)

print(f"Number of features: {X.shape[1]}")
print(f"Number of records: {X.shape[0]}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# PREPROCESSING
# ============================================================

print("\nCreating preprocessing pipeline...")

preprocessor = create_preprocessor(X_train)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Training data processed:", X_train_processed.shape)
print("Testing data processed:", X_test_processed.shape)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    os.path.join(MODEL_DIR, "preprocessor.pkl")
)

print("\nPreprocessor saved.")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": SVC(
        probability=True,
        random_state=42,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# TRAIN AND EVALUATE MODELS
# ============================================================

results = []

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(
        X_train_processed,
        y_train
    )

    # Predictions
    y_pred = model.predict(
        X_test_processed
    )

    # Probability predictions
    y_probability = model.predict_proba(
        X_test_processed
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    # Save model
    filename = name.lower().replace(" ", "_").replace("-", "")

    model_path = os.path.join(
        MODEL_DIR,
        f"{filename}.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"{name} completed.")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

# Sort by F1 score
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model_filename = (
    best_model_name
    .lower()
    .replace(" ", "_")
    .replace("-", "")
)

best_model_path = os.path.join(
    MODEL_DIR,
    f"{best_model_filename}.pkl"
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print(f"MODEL FILE: {best_model_path}")
print("=" * 70)

print("\nResults saved to:")
print(results_path)

print("\nAll models successfully trained.")