
"""
Mini IDS - Model Evaluation
Dataset: UNSW-NB15
Model: Random Forest

Run:
    python src/evaluate_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_FILE = MODEL_DIR / "random_forest_ids.joblib"

X_TEST_FILE = PROCESSED_DIR / "X_test.npz"
Y_TEST_FILE = PROCESSED_DIR / "y_test.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD MODEL AND TEST DATA
# ==========================================================

def load_data():

    print("\n" + "=" * 60)
    print("STEP 1: LOADING TRAINED MODEL AND TEST DATA")
    print("=" * 60)

    required_files = [
        MODEL_FILE,
        X_TEST_FILE,
        Y_TEST_FILE
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                "Run preprocessing.py and train_model.py first."
            )

    model = joblib.load(MODEL_FILE)

    X_test = sparse.load_npz(X_TEST_FILE).tocsr()

    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0].astype(int)

    if X_test.shape[0] != len(y_test):
        raise ValueError("Testing features and labels do not match.")

    if X_test.shape[1] != model.n_features_in_:
        raise ValueError("Testing feature count does not match model.")

    print(f"\nTesting records: {X_test.shape[0]:,}")
    print(f"Features: {X_test.shape[1]}")
    print("Model loaded successfully.")

    return model, X_test, y_test


# ==========================================================
# 3. GENERATE PREDICTIONS
# ==========================================================

def generate_predictions(model, X_test):

    print("\n" + "=" * 60)
    print("STEP 2: GENERATING PREDICTIONS")
    print("=" * 60)

    y_pred = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    attack_index = list(model.classes_).index(1)

    attack_probabilities = probabilities[:, attack_index]

    print("Predictions generated successfully.")

    return y_pred, attack_probabilities


# ==========================================================
# 4. CALCULATE EVALUATION METRICS
# ==========================================================

def calculate_metrics(y_test, y_pred, attack_probabilities):

    print("\n" + "=" * 60)
    print("STEP 3: CALCULATING PERFORMANCE METRICS")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) > 0 else 0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )

    roc_auc = roc_auc_score(
        y_test, attack_probabilities
    )

    pr_auc = average_precision_score(
        y_test, attack_probabilities
    )

    metrics = {
        "model": "Random Forest",
        "dataset": "UNSW-NB15",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "total_testing_records": int(len(y_test))
    }

    return metrics, cm


# ==========================================================
# 5. DISPLAY RESULTS
# ==========================================================

def display_results(metrics, cm, y_test, y_pred):

    print("\n" + "=" * 60)
    print("STEP 4: MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nAccuracy:             {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:            {metrics['precision'] * 100:.2f}%")
    print(f"Recall:               {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:             {metrics['f1_score'] * 100:.2f}%")
    print(f"Specificity:          {metrics['specificity'] * 100:.2f}%")
    print(
        f"False Positive Rate:  "
        f"{metrics['false_positive_rate'] * 100:.2f}%"
    )
    print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:               {metrics['pr_auc']:.4f}")

    print("\nCONFUSION MATRIX")
    print("-" * 40)

    print("                  Predicted")
    print("                Benign  Attack")

    print(
        f"Actual Benign    "
        f"{metrics['true_negative']:6d}  "
        f"{metrics['false_positive']:6d}"
    )

    print(
        f"Actual Attack    "
        f"{metrics['false_negative']:6d}  "
        f"{metrics['true_positive']:6d}"
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 40)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0
    )

    print(report)

    return report


# ==========================================================
# 6. SAVE EVALUATION RESULTS
# ==========================================================

def save_results(metrics, cm, report):

    print("\n" + "=" * 60)
    print("STEP 5: SAVING EVALUATION RESULTS")
    print("=" * 60)

    # Save metrics as JSON.
    with open(
        RESULTS_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    # Save classification report.
    with open(
        RESULTS_DIR / "evaluation_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("RANDOM FOREST IDS EVALUATION\n")
        file.write("=" * 45 + "\n\n")
        file.write(report)

    # Save confusion matrix.
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Attack"],
        columns=["Predicted Benign", "Predicted Attack"]
    )

    cm_df.to_csv(
        RESULTS_DIR / "evaluation_confusion_matrix.csv"
    )

    print("\nFiles saved:")
    print("1. evaluation_metrics.json")
    print("2. evaluation_report.txt")
    print("3. evaluation_confusion_matrix.csv")


# ==========================================================
# 7. MAIN EXECUTION
# ==========================================================

def main():

    print("\n")
    print("#" * 60)
    print(" MINI IDS - MODEL EVALUATION ")
    print("#" * 60)

    model, X_test, y_test = load_data()

    y_pred, attack_probabilities = generate_predictions(
        model, X_test
    )

    metrics, cm = calculate_metrics(
        y_test,
        y_pred,
        attack_probabilities
    )

    report = display_results(
        metrics,
        cm,
        y_test,
        y_pred
    )

    save_results(
        metrics,
        cm,
        report
    )

    print("\n" + "#" * 60)
    print(" MODEL EVALUATION COMPLETED ")
    print("#" * 60)


if __name__ == "__main__":
    main()
