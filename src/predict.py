
"""
Mini IDS - Network Traffic Prediction
Dataset: UNSW-NB15
Model: Random Forest

Purpose:
1. Accept a new network traffic CSV file.
2. Load the saved preprocessing pipeline.
3. Prepare incoming network traffic features.
4. Predict benign or malicious traffic.
5. Calculate attack probabilities.
6. Export prediction results to CSV.

Run:
    python src/predict.py --input dataset/UNSW_NB15_testing-set.csv

Optional output:
    python src/predict.py --input dataset/UNSW_NB15_testing-set.csv --output results/my_predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_FILE = MODEL_DIR / "random_forest_ids.joblib"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD TRAINED MODEL AND PREPROCESSOR
# ==========================================================

def load_model():

    print("\n" + "=" * 60)
    print("STEP 1: LOADING TRAINED MODEL")
    print("=" * 60)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Run train_model.py first."
        )

    if not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_FILE}\n"
            "Run preprocessing.py first."
        )

    model = joblib.load(MODEL_FILE)

    preprocessor = joblib.load(PREPROCESSOR_FILE)

    print("\nRandom Forest model loaded.")
    print("Preprocessing pipeline loaded.")

    return model, preprocessor


# ==========================================================
# 3. LOAD NEW NETWORK TRAFFIC CSV
# ==========================================================

def load_new_data(input_file):

    print("\n" + "=" * 60)
    print("STEP 2: LOADING NETWORK TRAFFIC DATA")
    print("=" * 60)

    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_file}"
        )

    if input_file.suffix.lower() != ".csv":
        raise ValueError("Please provide a CSV file.")

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    # Remove accidental index columns.
    df = df.loc[
        :, ~df.columns.str.match(r"^Unnamed")
    ]

    if df.empty:
        raise ValueError("The uploaded CSV contains no records.")

    print(f"\nInput file: {input_file.name}")
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    return df


# ==========================================================
# 4. PREPARE FEATURES
# ==========================================================

def prepare_features(df, preprocessor):

    print("\n" + "=" * 60)
    print("STEP 3: PREPARING NETWORK FEATURES")
    print("=" * 60)

    df = df.copy()

    # Preserve an existing target label only for optional comparison.
    actual_labels = None

    if "label" in df.columns:

        actual_labels = pd.to_numeric(
            df["label"],
            errors="coerce"
        )

    # The fitted preprocessing pipeline expects these columns.
    expected_features = list(
        preprocessor.feature_names_in_
    )

    # Remove target and identifier information.
    # These columns must not be used as model inputs.
    df = df.drop(
        columns=["id", "attack_cat", "label"],
        errors="ignore"
    )

    # Identify columns used by the fitted transformers.
    numerical_features = []
    categorical_features = []

    for name, transformer, columns in preprocessor.transformers_:

        if name == "numerical":
            numerical_features = list(columns)

        elif name == "categorical":
            categorical_features = list(columns)

    # Check which expected columns are missing.
    missing_columns = [
        col for col in expected_features
        if col not in df.columns
    ]

    if missing_columns:

        print("\nWarning: Some input features are missing:")

        print(missing_columns)

        print("Missing features will be handled by the imputer.")

    # Add missing columns and maintain training column order.
    X = df.reindex(
        columns=expected_features
    ).copy()

    # Convert numerical features into numeric values.
    for column in numerical_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite numerical values with NaN.
    X[numerical_features] = X[
        numerical_features
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Keep categorical features compatible with the
    # categorical imputation and OneHotEncoder pipeline.
    for column in categorical_features:

        X[column] = X[column].astype("object")

    print("\nFeature preparation completed.")

    print(f"Input records: {len(X):,}")
    print(f"Expected features: {len(expected_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    return X, actual_labels


# ==========================================================
# 5. TRANSFORM INPUT DATA
# ==========================================================

def transform_features(X, preprocessor):

    print("\n" + "=" * 60)
    print("STEP 4: APPLYING PREPROCESSING PIPELINE")
    print("=" * 60)

    # Do not fit again.
    # Use the exact transformations learned during training.
    X_processed = preprocessor.transform(X)

    X_processed = sparse.csr_matrix(
        X_processed,
        dtype=np.float32
    )

    print("\nFeature transformation completed.")

    print(
        f"Transformed data shape: {X_processed.shape}"
    )

    return X_processed


# ==========================================================
# 6. PREDICT NETWORK TRAFFIC
# ==========================================================

def predict_traffic(model, X_processed):

    print("\n" + "=" * 60)
    print("STEP 5: DETECTING SUSPICIOUS NETWORK TRAFFIC")
    print("=" * 60)

    # Predict classes.
    predictions = model.predict(X_processed)

    # Get probabilities for both classes.
    probabilities = model.predict_proba(X_processed)

    # Find probability column for class 1 (Attack).
    attack_index = list(model.classes_).index(1)

    attack_probabilities = probabilities[
        :, attack_index
    ]

    # Maximum probability represents prediction confidence.
    confidence = np.max(
        probabilities,
        axis=1
    )

    print("\nPredictions completed.")

    benign_count = int(np.sum(predictions == 0))
    attack_count = int(np.sum(predictions == 1))

    print(f"\nTotal records: {len(predictions):,}")
    print(f"Benign records: {benign_count:,}")
    print(f"Suspicious records: {attack_count:,}")

    return predictions, attack_probabilities, confidence


# ==========================================================
# 7. GENERATE PREDICTION REPORT
# ==========================================================

def create_prediction_report(
    original_df,
    predictions,
    attack_probabilities,
    confidence,
    actual_labels=None
):

    print("\n" + "=" * 60)
    print("STEP 6: GENERATING PREDICTION REPORT")
    print("=" * 60)

    results = original_df.copy()

    results["predicted_label"] = predictions

    results["prediction"] = np.where(
        predictions == 1,
        "Attack",
        "Benign"
    )

    results["attack_probability"] = (
        attack_probabilities
    )

    results["confidence"] = confidence

    if actual_labels is not None:

        results["actual_label"] = (
            actual_labels.to_numpy()
        )

        results["actual_class"] = (
            actual_labels.map({
                0: "Benign",
                1: "Attack"
            }).to_numpy()
        )

    # Select suspicious records.
    suspicious_records = results[
        results["predicted_label"] == 1
    ].copy()

    print(
        f"\nTotal suspicious records: "
        f"{len(suspicious_records):,}"
    )

    print("\nSample suspicious records:")

    if not suspicious_records.empty:

        print(
            suspicious_records.head(10).to_string(
                index=False
            )
        )

    else:

        print("No suspicious records detected.")

    return results, suspicious_records


# ==========================================================
# 8. SAVE PREDICTION RESULTS
# ==========================================================

def save_predictions(
    results,
    suspicious_records,
    output_file
):

    print("\n" + "=" * 60)
    print("STEP 7: SAVING PREDICTION RESULTS")
    print("=" * 60)

    output_file = Path(output_file)

    if not output_file.is_absolute():
        output_file = PROJECT_DIR / output_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    suspicious_file = output_file.with_name(
        output_file.stem + "_suspicious.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    suspicious_records.to_csv(
        suspicious_file,
        index=False
    )

    print("\nPrediction results saved successfully.")

    print(f"All predictions: {output_file}")

    print(f"Suspicious records: {suspicious_file}")


# ==========================================================
# 9. OPTIONAL EVALUATION IF ACTUAL LABELS EXIST
# ==========================================================

def compare_with_actual_labels(
    actual_labels,
    predictions
):

    if actual_labels is None:
        return

    valid = actual_labels.notna()

    if not valid.any():
        print("\nNo valid actual labels available for comparison.")
        return

    actual = actual_labels[valid].astype(int).to_numpy()
    predicted = predictions[valid.to_numpy()]

    if not set(np.unique(actual)).issubset({0, 1}):
        print("\nActual labels are not binary; skipping comparison.")
        return

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    print("\n" + "=" * 60)
    print("OPTIONAL: COMPARISON WITH ACTUAL LABELS")
    print("=" * 60)

    print(
        f"Accuracy:  {accuracy_score(actual, predicted) * 100:.2f}%"
    )

    print(
        f"Precision: {precision_score(actual, predicted, zero_division=0) * 100:.2f}%"
    )

    print(
        f"Recall:    {recall_score(actual, predicted, zero_division=0) * 100:.2f}%"
    )

    print(
        f"F1-score:  {f1_score(actual, predicted, zero_division=0) * 100:.2f}%"
    )


# ==========================================================
# 10. MAIN EXECUTION
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Mini IDS Network Traffic Prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the network traffic CSV file."
    )

    parser.add_argument(
        "--output",
        default="results/predictions.csv",
        help="Path to save the prediction CSV."
    )

    args = parser.parse_args()

    print("\n")
    print("#" * 60)
    print(" MINI IDS - NETWORK TRAFFIC PREDICTION ")
    print("#" * 60)

    # Step 1: Load saved model and preprocessor.
    model, preprocessor = load_model()

    # Step 2: Load incoming traffic.
    original_df = load_new_data(args.input)

    # Step 3: Prepare features.
    X, actual_labels = prepare_features(
        original_df,
        preprocessor
    )

    # Step 4: Transform features.
    X_processed = transform_features(
        X,
        preprocessor
    )

    # Step 5: Predict traffic.
    predictions, attack_probabilities, confidence = (
        predict_traffic(
            model,
            X_processed
        )
    )

    # Step 6: Generate results.
    results, suspicious_records = create_prediction_report(
        original_df,
        predictions,
        attack_probabilities,
        confidence,
        actual_labels
    )

    # Step 7: Save results.
    save_predictions(
        results,
        suspicious_records,
        args.output
    )

    # Optional comparison if actual labels are present.
    compare_with_actual_labels(
        actual_labels,
        predictions
    )

    print("\n" + "#" * 60)
    print(" NETWORK TRAFFIC PREDICTION COMPLETED ")
    print("#" * 60)


if __name__ == "__main__":
    main()
