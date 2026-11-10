
"""
Mini Intrusion Detection System Using Machine Learning
Dataset: UNSW-NB15

Purpose:
1. Load training and testing datasets.
2. Inspect and clean network traffic data.
3. Handle missing and infinite values.
4. Remove unnecessary columns and duplicate training records.
5. Separate features and binary target.
6. Encode categorical features.
7. Impute missing numerical and categorical values.
8. Save processed datasets and preprocessing pipeline.

Run from the project root:
    python src/data_preprocessing.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
PROCESSED_DIR = DATASET_DIR / "processed"

TRAIN_FILE = DATASET_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATASET_DIR / "UNSW_NB15_testing-set.csv"

REMOVE_TRAIN_DUPLICATES = False

RANDOM_STATE = 42

# Create output folders if they do not exist.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD DATASETS
# ==========================================================

def load_datasets():

    print("\n" + "=" * 65)
    print("STEP 1: LOADING UNSW-NB15 DATASETS")
    print("=" * 65)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_FILE}"
        )

    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    test_df = pd.read_csv(TEST_FILE, low_memory=False)

    # Remove accidental whitespace from column names.
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Remove automatically generated CSV index columns.
    train_df = train_df.loc[
        :, ~train_df.columns.str.match(r"^Unnamed")
    ]

    test_df = test_df.loc[
        :, ~test_df.columns.str.match(r"^Unnamed")
    ]

    print(f"\nTraining records: {train_df.shape[0]:,}")
    print(f"Training columns: {train_df.shape[1]}")

    print(f"\nTesting records: {test_df.shape[0]:,}")
    print(f"Testing columns: {test_df.shape[1]}")

    return train_df, test_df


# ==========================================================
# 3. INITIAL DATASET INSPECTION
# ==========================================================

def inspect_dataset(df, dataset_name):

    print("\n" + "=" * 65)
    print(f"STEP 2: {dataset_name.upper()} DATASET INSPECTION")
    print("=" * 65)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.value_counts())

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False))

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if "label" in df.columns:
        print("\nTarget distribution:")

        print(
            df["label"]
            .value_counts(dropna=False)
            .sort_index()
        )

    if "attack_cat" in df.columns:
        print("\nAttack category distribution:")

        print(
            df["attack_cat"]
            .value_counts(dropna=False)
        )


# ==========================================================
# 4. DATA CLEANING
# ==========================================================

def clean_dataset(df, dataset_name):

    print("\n" + "=" * 65)
    print(f"STEP 3: CLEANING {dataset_name.upper()} DATA")
    print("=" * 65)

    df = df.copy()

    # Remove accidental index columns.
    df = df.loc[
        :, ~df.columns.str.match(r"^Unnamed")
    ]

    # Replace positive and negative infinity with NaN.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows with missing target labels.
    if "label" not in df.columns:
        raise ValueError(
            f"'label' target column is missing in {dataset_name}."
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(subset=["label"])

    print(
        f"Rows removed due to missing/invalid labels: "
        f"{before - len(df):,}"
    )

    # Validate binary labels.
    invalid_labels = ~df["label"].isin([0, 1])

    if invalid_labels.any():
        raise ValueError(
            f"Unexpected target labels found in {dataset_name}: "
            f"{df.loc[invalid_labels, 'label'].unique()}"
        )

    df["label"] = df["label"].astype("int8")

    # Remove duplicate training records.
    # Ignore ID because IDs do not describe network behavior.
    if (
        dataset_name.lower() == "training"
        and REMOVE_TRAIN_DUPLICATES
    ):

        before = len(df)

        duplicate_columns = [
            col for col in df.columns
            if col.lower() != "id"
        ]

        df = df.drop_duplicates(
            subset=duplicate_columns
        ).copy()

        print(
            f"Duplicate training records removed: "
            f"{before - len(df):,}"
        )

    print(f"Records after cleaning: {len(df):,}")

    return df


# ==========================================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================================

def separate_features_target(train_df, test_df):

    print("\n" + "=" * 65)
    print("STEP 4: SEPARATING FEATURES AND TARGET")
    print("=" * 65)

    TARGET_COLUMN = "label"

    # attack_cat is target-related information.
    # It must NOT be given to the binary classifier.
    DROP_COLUMNS = [
        "id",
        "attack_cat"
    ]

    # Remove columns only if they exist.
    drop_columns = [
        col for col in DROP_COLUMNS
        if col in train_df.columns
    ]

    print("\nUnnecessary columns removed:")
    print(drop_columns)

    # Separate binary targets.
    y_train = train_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # Remove target and unnecessary columns.
    X_train = train_df.drop(
        columns=[TARGET_COLUMN] + drop_columns,
        errors="ignore"
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN] + drop_columns,
        errors="ignore"
    )

    # Align test features with training feature columns.
    # Any missing test features are filled later by imputers.
    X_test = X_test.reindex(
        columns=X_train.columns
    )

    # Keep column order identical.
    X_test = X_test[X_train.columns]

    print(f"\nTraining features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")

    print(f"\nTraining targets: {y_train.shape}")
    print(f"Testing targets: {y_test.shape}")

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# ==========================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================================

def identify_feature_types(X_train):

    print("\n" + "=" * 65)
    print("STEP 5: IDENTIFYING FEATURE TYPES")
    print("=" * 65)

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    print(
        f"\nTotal numerical features: "
        f"{len(numerical_features)}"
    )

    print(
        f"Total categorical features: "
        f"{len(categorical_features)}"
    )

    return numerical_features, categorical_features


# ==========================================================
# 7. BUILD PREPROCESSING PIPELINE
# ==========================================================

def build_preprocessor(
    numerical_features,
    categorical_features
):

    print("\n" + "=" * 65)
    print("STEP 6: BUILDING PREPROCESSING PIPELINE")
    print("=" * 65)

    # Numerical features:
    # Replace missing values with the median.
    numerical_pipeline = SimpleImputer(
        strategy="median"
    )

    # Categorical features:
    # Replace missing values with the most frequent category.
    # Convert categories into numerical binary columns.
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )

    categorical_pipeline = PipelineForCategorical(
        encoder
    )

    # Combine numerical and categorical preprocessing.
    transformers = []

    if numerical_features:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0
    )

    print("\nPreprocessing pipeline created successfully.")

    return preprocessor


# ==========================================================
# HELPER: CATEGORICAL IMPUTATION + ENCODING
# ==========================================================

from sklearn.pipeline import Pipeline


def PipelineForCategorical(encoder):

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing"
                )
            ),
            (
                "onehot",
                encoder
            )
        ]
    )


# ==========================================================
# 8. TRANSFORM TRAINING AND TESTING DATA
# ==========================================================

def transform_data(preprocessor, X_train, X_test):

    print("\n" + "=" * 65)
    print("STEP 7: TRANSFORMING DATA")
    print("=" * 65)

    print("\nFitting preprocessing on training data...")

    # Fit ONLY on training data.
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    print("Transforming testing data...")

    # Apply the already-fitted transformations to test data.
    X_test_processed = preprocessor.transform(
        X_test
    )

    # Ensure sparse matrix format for efficient storage.
    X_train_processed = sparse.csr_matrix(
        X_train_processed,
        dtype=np.float32
    )

    X_test_processed = sparse.csr_matrix(
        X_test_processed,
        dtype=np.float32
    )

    print("\nTransformation completed.")

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing shape: "
        f"{X_test_processed.shape}"
    )

    print(
        f"\nTraining matrix memory: "
        f"{X_train_processed.data.nbytes / (1024**2):.2f} MB"
    )

    print(
        f"Testing matrix memory: "
        f"{X_test_processed.data.nbytes / (1024**2):.2f} MB"
    )

    return X_train_processed, X_test_processed


# ==========================================================
# 9. SAVE PROCESSED DATA AND PIPELINE
# ==========================================================

def save_processed_data(
    preprocessor,
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    numerical_features,
    categorical_features
):

    print("\n" + "=" * 65)
    print("STEP 8: SAVING PROCESSED DATA")
    print("=" * 65)

    # Save sparse feature matrices.
    sparse.save_npz(
        PROCESSED_DIR / "X_train.npz",
        X_train_processed
    )

    sparse.save_npz(
        PROCESSED_DIR / "X_test.npz",
        X_test_processed
    )

    # Save target labels.
    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    # Save fitted preprocessing pipeline.
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.joblib"
    )

    # Get transformed feature names.
    feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )

    with open(
        PROCESSED_DIR / "feature_names.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_names,
            file,
            indent=4
        )

    # Save metadata.
    metadata = {
        "dataset": "UNSW-NB15",
        "target_column": "label",
        "target_mapping": {
            "0": "Benign",
            "1": "Attack"
        },
        "training_records": int(len(y_train)),
        "testing_records": int(len(y_test)),
        "original_numerical_features": numerical_features,
        "original_categorical_features": categorical_features,
        "transformed_feature_count": len(feature_names),
        "duplicate_removal": REMOVE_TRAIN_DUPLICATES
    }

    with open(
        PROCESSED_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nAll files saved successfully.")

    print(f"\nProcessed data location: {PROCESSED_DIR}")
    print(f"Model location: {MODEL_DIR}")

    print("\nSaved files:")
    print("1. X_train.npz")
    print("2. X_test.npz")
    print("3. y_train.csv")
    print("4. y_test.csv")
    print("5. feature_names.json")
    print("6. metadata.json")
    print("7. preprocessor.joblib")


# ==========================================================
# 10. MAIN EXECUTION
# ==========================================================

def main():

    print("\n")
    print("#" * 65)
    print(" MINI IDS - UNSW-NB15 DATA PREPROCESSING ")
    print("#" * 65)

    # Step 1: Load datasets.
    train_df, test_df = load_datasets()

    # Step 2: Inspect original datasets.
    inspect_dataset(train_df, "Training")
    inspect_dataset(test_df, "Testing")

    # Step 3: Clean datasets.
    train_df = clean_dataset(
        train_df,
        "Training"
    )

    test_df = clean_dataset(
        test_df,
        "Testing"
    )

    # Step 4: Separate features and targets.
    X_train, X_test, y_train, y_test = (
        separate_features_target(
            train_df,
            test_df
        )
    )

    # Step 5: Identify feature types.
    numerical_features, categorical_features = (
        identify_feature_types(X_train)
    )

    # Step 6: Build preprocessing pipeline.
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    # Step 7: Transform data.
    X_train_processed, X_test_processed = (
        transform_data(
            preprocessor,
            X_train,
            X_test
        )
    )

    # Step 8: Save everything.
    save_processed_data(
        preprocessor,
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        numerical_features,
        categorical_features
    )

    print("\n" + "#" * 65)
    print(" PREPROCESSING COMPLETED SUCCESSFULLY! ")
    print("#" * 65)

    print("\nYour data is now ready for machine learning.")
    print("Next step: Train the Random Forest IDS classifier.")


if __name__ == "__main__":
    main()
