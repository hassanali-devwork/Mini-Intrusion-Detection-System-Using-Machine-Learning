
"""
Mini Intrusion Detection System Using Machine Learning
Model Training: Random Forest Classifier
Dataset: UNSW-NB15

Purpose:
1. Load preprocessed training and testing datasets.
2. Train a Random Forest classifier.
3. Evaluate model performance.
4. Calculate accuracy, precision, recall and F1-score.
5. Calculate false-positive rate and ROC-AUC.
6. Generate confusion matrix and classification report.
7. Save trained model and evaluation results.

Run from project root:
    python src/train_model.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
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

warnings.filterwarnings("ignore")


# ==========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Input files
X_TRAIN_FILE = PROCESSED_DIR / "X_train.npz"
X_TEST_FILE = PROCESSED_DIR / "X_test.npz"

Y_TRAIN_FILE = PROCESSED_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DIR / "y_test.csv"

FEATURE_NAMES_FILE = PROCESSED_DIR / "feature_names.json"

# Output files
MODEL_FILE = MODEL_DIR / "random_forest_ids.joblib"

METRICS_FILE = RESULTS_DIR / "model_metrics.json"

REPORT_FILE = RESULTS_DIR / "classification_report.txt"

CONFUSION_FILE = RESULTS_DIR / "confusion_matrix.csv"

IMPORTANCE_FILE = RESULTS_DIR / "feature_importance.csv"

PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"


# ==========================================================
# 2. LOAD PREPROCESSED DATA
# ==========================================================

def load_processed_data():

    print("\n" + "=" * 65)
    print("STEP 1: LOADING PREPROCESSED DATA")
    print("=" * 65)

    required_files = [
        X_TRAIN_FILE,
        X_TEST_FILE,
        Y_TRAIN_FILE,
        Y_TEST_FILE,
        FEATURE_NAMES_FILE
    ]

    for file in required_files:

        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}\n"
                "Please run preprocessing.py first."
            )

    # Load sparse feature matrices.
    X_train = sparse.load_npz(X_TRAIN_FILE).tocsr()
    X_test = sparse.load_npz(X_TEST_FILE).tocsr()

    # Load target labels.
    y_train = pd.read_csv(Y_TRAIN_FILE).iloc[:, 0]
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]

    # Convert labels to integers.
    y_train = y_train.astype("int8")
    y_test = y_test.astype("int8")

    # Load feature names.
    with open(
        FEATURE_NAMES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        feature_names = json.load(file)

    # Verify data consistency.
    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "Training feature and target row counts do not match."
        )

    if X_test.shape[0] != len(y_test):
        raise ValueError(
            "Testing feature and target row counts do not match."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training and testing feature counts do not match."
        )

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            "Feature names and transformed feature count do not match."
        )

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must contain only 0 and 1.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Testing labels must contain only 0 and 1.")

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    print(f"\nTraining samples: {len(y_train):,}")
    print(f"Testing samples: {len(y_test):,}")

    print(f"Total features: {len(feature_names)}")

    print("\nTraining class distribution:")
    print(
        y_train.value_counts().sort_index()
        .rename(index={0: "Benign", 1: "Attack"})
    )

    print("\nTesting class distribution:")
    print(
        y_test.value_counts().sort_index()
        .rename(index={0: "Benign", 1: "Attack"})
    )

    return X_train, X_test, y_train, y_test, feature_names


# ==========================================================
# 3. CREATE RANDOM FOREST CLASSIFIER
# ==========================================================

def create_model():

    print("\n" + "=" * 65)
    print("STEP 2: CREATING RANDOM FOREST MODEL")
    print("=" * 65)

    model = RandomForestClassifier(

        # Number of decision trees.
        n_estimators=200,

        # Maximum number of features considered per split.
        max_features="sqrt",

        # Control class imbalance.
        class_weight="balanced_subsample",

        # Minimum samples required to split a node.
        min_samples_split=2,

        # Minimum samples required at a leaf.
        min_samples_leaf=1,

        # Make results reproducible.
        random_state=42,

        # Use all available CPU cores.
        n_jobs=-1,

        # Use Gini impurity to find the best splits.
        criterion="gini",

        # Grow trees until stopping conditions are met.
        bootstrap=True,

        verbose=1
    )

    print("\nRandom Forest configuration:")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max features: {model.max_features}")
    print(f"Class weight: {model.class_weight}")
    print(f"Random state: {model.random_state}")
    print(f"CPU workers: {model.n_jobs}")

    return model


# ==========================================================
# 4. TRAIN RANDOM FOREST MODEL
# ==========================================================

def train_model(model, X_train, y_train):

    print("\n" + "=" * 65)
    print("STEP 3: TRAINING RANDOM FOREST MODEL")
    print("=" * 65)

    print("\nTraining started...")
    print("Please wait. This may take some time.")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print("\nTraining completed successfully!")

    print(
        f"Training time: {training_time:.2f} seconds"
    )

    print(
        f"Number of trees trained: {len(model.estimators_)}"
    )

    return model, training_time


# ==========================================================
# 5. GENERATE PREDICTIONS
# ==========================================================

def generate_predictions(model, X_test):

    print("\n" + "=" * 65)
    print("STEP 4: GENERATING TEST PREDICTIONS")
    print("=" * 65)

    start_time = time.time()

    # Predict binary classes.
    y_pred = model.predict(X_test)

    # Get probability of attack (class 1).
    probabilities = model.predict_proba(X_test)

    attack_class_index = list(model.classes_).index(1)

    attack_probabilities = probabilities[
        :, attack_class_index
    ]

    prediction_time = time.time() - start_time

    print("\nPredictions generated successfully!")

    print(f"Prediction time: {prediction_time:.2f} seconds")

    print("\nPredicted class distribution:")

    unique, counts = np.unique(
        y_pred,
        return_counts=True
    )

    for label, count in zip(unique, counts):

        name = "Benign" if label == 0 else "Attack"

        print(f"{name}: {count:,}")

    return y_pred, attack_probabilities, prediction_time


# ==========================================================
# 6. EVALUATE MODEL PERFORMANCE
# ==========================================================

def evaluate_model(
    y_test,
    y_pred,
    attack_probabilities,
    training_time,
    prediction_time
):

    print("\n" + "=" * 65)
    print("STEP 5: EVALUATING MODEL PERFORMANCE")
    print("=" * 65)

    # Calculate classification metrics.
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # Confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # False positive rate.
    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    # ROC-AUC measures how well probabilities rank classes.
    roc_auc = roc_auc_score(
        y_test,
        attack_probabilities
    )

    # Precision-recall AUC (average precision).
    pr_auc = average_precision_score(
        y_test,
        attack_probabilities
    )

    # Specificity: correctly identifying benign traffic.
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # Display results.
    print("\nMODEL PERFORMANCE RESULTS")
    print("-" * 45)

    print(f"Accuracy:             {accuracy * 100:.2f}%")
    print(f"Precision:            {precision * 100:.2f}%")
    print(f"Recall:               {recall * 100:.2f}%")
    print(f"F1-Score:             {f1 * 100:.2f}%")
    print(f"Specificity:          {specificity * 100:.2f}%")
    print(f"False Positive Rate:  {false_positive_rate * 100:.2f}%")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"PR-AUC:               {pr_auc:.4f}")

    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Prediction time: {prediction_time:.2f} seconds")

    # Display confusion matrix.
    print("\nCONFUSION MATRIX")
    print("-" * 45)

    print("                  Predicted")
    print("                Benign  Attack")
    print(f"Actual Benign    {tn:6d}  {fp:6d}")
    print(f"Actual Attack    {fn:6d}  {tp:6d}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 45)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0
    )

    print(report)

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
        "training_time_seconds": float(training_time),
        "prediction_time_seconds": float(prediction_time),
        "testing_records": int(len(y_test))
    }

    return metrics, cm, report


# ==========================================================
# 7. SAVE TRAINED MODEL
# ==========================================================

def save_model(model):

    print("\n" + "=" * 65)
    print("STEP 6: SAVING TRAINED MODEL")
    print("=" * 65)

    joblib.dump(
        model,
        MODEL_FILE,
        compress=3
    )

    print("\nModel saved successfully!")

    print(f"Location: {MODEL_FILE}")

    print(
        f"File size: "
        f"{MODEL_FILE.stat().st_size / (1024**2):.2f} MB"
    )


# ==========================================================
# 8. SAVE EVALUATION RESULTS
# ==========================================================

def save_results(
    metrics,
    cm,
    report,
    model,
    feature_names,
    y_test,
    y_pred,
    attack_probabilities
):

    print("\n" + "=" * 65)
    print("STEP 7: SAVING EVALUATION RESULTS")
    print("=" * 65)

    # Save metrics in JSON format.
    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save classification report.
    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RANDOM FOREST IDS - CLASSIFICATION REPORT\n"
        )

        file.write("=" * 50 + "\n\n")

        file.write(report)

    # Save confusion matrix.
    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Attack"],
        columns=["Predicted Benign", "Predicted Attack"]
    )

    confusion_df.to_csv(CONFUSION_FILE)

    # Save feature importance.
    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    # Save individual test predictions.
    predictions_df = pd.DataFrame({

        "actual_label": np.asarray(y_test),

        "predicted_label": np.asarray(y_pred),

        "attack_probability": attack_probabilities,

        "prediction": [
            "Attack" if value == 1 else "Benign"
            for value in y_pred
        ]

    })

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print("\nSaved evaluation files:")

    print("1. model_metrics.json")
    print("2. classification_report.txt")
    print("3. confusion_matrix.csv")
    print("4. feature_importance.csv")
    print("5. test_predictions.csv")


# ==========================================================
# 9. DISPLAY TOP IMPORTANT FEATURES
# ==========================================================

def display_feature_importance(model, feature_names):

    print("\n" + "=" * 65)
    print("STEP 8: TOP IMPORTANT FEATURES")
    print("=" * 65)

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 15 features used by Random Forest:\n")

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )


# ==========================================================
# 10. MAIN EXECUTION
# ==========================================================

def main():

    print("\n")
    print("#" * 65)
    print(" MINI IDS - RANDOM FOREST MODEL TRAINING ")
    print("#" * 65)

    # Step 1: Load data.
    X_train, X_test, y_train, y_test, feature_names = (
        load_processed_data()
    )

    # Step 2: Create classifier.
    model = create_model()

    # Step 3: Train classifier.
    model, training_time = train_model(
        model,
        X_train,
        y_train
    )

    # Step 4: Generate predictions.
    y_pred, attack_probabilities, prediction_time = (
        generate_predictions(
            model,
            X_test
        )
    )

    # Step 5: Evaluate classifier.
    metrics, cm, report = evaluate_model(
        y_test,
        y_pred,
        attack_probabilities,
        training_time,
        prediction_time
    )

    # Step 6: Save trained model.
    save_model(model)

    # Step 7: Save results.
    save_results(
        metrics,
        cm,
        report,
        model,
        feature_names,
        y_test,
        y_pred,
        attack_probabilities
    )

    # Step 8: Display important features.
    display_feature_importance(
        model,
        feature_names
    )

    print("\n" + "#" * 65)
    print(" RANDOM FOREST TRAINING COMPLETED SUCCESSFULLY! ")
    print("#" * 65)

    print("\nYour IDS machine learning model is ready.")

    print(f"\nModel: {MODEL_FILE}")
    print(f"Results: {RESULTS_DIR}")

    print("\nNext step: Model testing and Streamlit dashboard.")


if __name__ == "__main__":
    main()
