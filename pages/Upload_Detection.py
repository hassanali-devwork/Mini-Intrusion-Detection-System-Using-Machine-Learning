
"""
Upload & Detection Page
Network Intrusion Detection System

Premium minimal cybersecurity interface.
No logos or brand graphics.
Original ML detection pipeline preserved.
"""

from pathlib import Path
import textwrap

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy import sparse


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"


# ============================================================
# CUSTOM CSS
# ============================================================

def apply_design():

    st.markdown("""
    <style>

    /* GLOBAL */

    .stApp {
        background:
            radial-gradient(
                circle at 5% 0%,
                rgba(79, 70, 229, 0.12),
                transparent 35%
            ),
            radial-gradient(
                circle at 95% 10%,
                rgba(6, 182, 212, 0.08),
                transparent 35%
            ),
            #080d1a;

        color: #e2e8f0;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* PAGE HEADER */

    .page-header {
        padding: 32px 0 25px 0;
        border-bottom: 1px solid rgba(148,163,184,0.15);
        margin-bottom: 30px;
    }

    .page-eyebrow {
        color: #818cf8;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .page-title {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 850;
        letter-spacing: -1.5px;
        line-height: 1.2;
        margin-bottom: 12px;
    }

    .page-description {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 750px;
    }

    /* SECTION HEADINGS */

    .section-heading {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f1f5f9;
        margin-bottom: 7px;
    }

    .section-subheading {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-bottom: 20px;
    }

    /* INSTRUCTION PANEL */

    .instruction-panel {
        background:
            linear-gradient(
                120deg,
                rgba(30,41,80,0.65),
                rgba(15,23,42,0.85)
            );

        border: 1px solid rgba(129,140,248,0.23);
        border-radius: 17px;
        padding: 24px;
        margin-bottom: 28px;
    }

    .instruction-title {
        color: #c7d2fe;
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 15px;
    }

    .instruction-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
    }

    .instruction-step {
        color: #a5b4c8;
        font-size: 0.84rem;
        line-height: 1.6;
        padding: 12px;
        border-radius: 10px;
        background: rgba(255,255,255,0.035);
    }

    .instruction-step strong {
        display: block;
        color: #e2e8f0;
        margin-bottom: 5px;
    }

    /* UPLOAD PANEL */

    .upload-panel {
        background:
            linear-gradient(
                145deg,
                rgba(20,30,53,0.95),
                rgba(12,19,35,0.98)
            );

        border: 1px solid rgba(129,140,248,0.24);
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 15px 45px rgba(0,0,0,0.18);
    }

    .upload-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .upload-description {
        color: #94a3b8;
        font-size: 0.87rem;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(99,102,241,0.045);
        border: 1px dashed rgba(129,140,248,0.55);
        border-radius: 15px;
        padding: 18px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #67e8f9;
        background: rgba(99,102,241,0.08);
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stFileUploader"] label {
        color: #e2e8f0 !important;
    }

    /* FILE INFORMATION */

    .file-card {
        background: rgba(17,27,48,0.9);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 15px;
        padding: 20px;
        min-height: 110px;
    }

    .file-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .file-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    /* METRIC CARDS */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(22,32,55,0.96),
                rgba(12,19,35,0.98)
            );

        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 17px;
        padding: 23px 20px;
        min-height: 140px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.16);
    }

    .metric-card-label {
        color: #94a3b8;
        font-size: 0.84rem;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .metric-card-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 850;
        letter-spacing: -1px;
    }

    .metric-card-note {
        color: #8493aa;
        font-size: 0.75rem;
        margin-top: 8px;
    }

    .metric-purple {
        border-left: 4px solid #818cf8;
    }

    .metric-green {
        border-left: 4px solid #34d399;
    }

    .metric-red {
        border-left: 4px solid #fb7185;
    }

    .metric-cyan {
        border-left: 4px solid #22d3ee;
    }

    /* RESULT PANEL */

    .success-panel {
        background:
            linear-gradient(
                120deg,
                rgba(6,78,59,0.22),
                rgba(15,23,42,0.92)
            );

        border: 1px solid rgba(52,211,153,0.35);
        border-radius: 18px;
        padding: 25px;
        margin: 25px 0;
    }

    .success-title {
        color: #6ee7b7;
        font-size: 1.5rem;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .success-description {
        color: #a7f3d0;
        font-size: 0.88rem;
    }

    /* EMPTY STATE */

    .empty-panel {
        background: rgba(17,27,48,0.75);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 35px 25px;
        text-align: center;
    }

    .empty-title {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .empty-description {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.7;
    }

    /* EXPECTED FORMAT */

    .format-panel {
        background: rgba(17,27,48,0.85);
        border: 1px solid rgba(148,163,184,0.13);
        border-radius: 16px;
        padding: 23px;
    }

    .format-code {
        color: #a5b4fc;
        font-family: monospace;
        font-size: 0.83rem;
        line-height: 2;
        overflow-wrap: anywhere;
    }

    /* STREAMLIT BUTTONS */

    .stButton button {
        min-height: 49px;
        border-radius: 11px;
        font-weight: 750;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(99,102,241,0.2);
    }

    /* DATAFRAME */

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ALERTS */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* FOOTER */

    .page-footer {
        border-top: 1px solid rgba(148,163,184,0.13);
        margin-top: 45px;
        padding-top: 22px;
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        line-height: 1.8;
    }

    @media(max-width: 800px) {
        .instruction-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .upload-panel {
            padding: 20px;
        }
    }

    @media(max-width: 500px) {
        .instruction-grid {
            grid-template-columns: 1fr;
        }
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessing pipeline."""

    try:

        model_file = MODEL_DIR / "random_forest_ids.joblib"
        preprocessor_file = MODEL_DIR / "preprocessor.joblib"

        if not model_file.exists():
            st.error(f"Model file not found: {model_file}")
            return None, None

        if not preprocessor_file.exists():
            st.error(f"Preprocessor file not found: {preprocessor_file}")
            return None, None

        model = joblib.load(model_file)
        preprocessor = joblib.load(preprocessor_file)

        return model, preprocessor

    except Exception as error:

        st.error(f"Error loading model: {error}")
        return None, None


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df, preprocessor):
    """Prepare uploaded data using the training preprocessing pipeline."""

    df = df.copy()

    # Preserve actual labels

    actual_labels = None

    if "label" in df.columns:

        actual_labels = pd.to_numeric(
            df["label"],
            errors="coerce"
        )

    # Expected training features

    expected_features = list(
        preprocessor.feature_names_in_
    )

    # Remove target and identifier columns

    df = df.drop(
        columns=["id", "attack_cat", "label"],
        errors="ignore"
    )

    numerical_features = []
    categorical_features = []

    for name, transformer, columns in preprocessor.transformers_:

        if name == "numerical":
            numerical_features = list(columns)

        elif name == "categorical":
            categorical_features = list(columns)

    # Missing features

    missing_columns = [
        col for col in expected_features
        if col not in df.columns
    ]

    if missing_columns:

        st.warning(
            f"{len(missing_columns)} expected features are missing. "
            "The preprocessing pipeline will handle missing values."
        )

    # Match training feature order

    X = df.reindex(
        columns=expected_features
    ).copy()

    # Numerical conversion

    for column in numerical_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite values

    X[numerical_features] = X[
        numerical_features
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Categorical conversion

    for column in categorical_features:

        X[column] = X[column].astype("object")

    return X, actual_labels


# ============================================================
# TRANSFORM FEATURES
# ============================================================

def transform_features(X, preprocessor):
    """Transform features with the fitted preprocessing pipeline."""

    X_processed = preprocessor.transform(X)

    X_processed = sparse.csr_matrix(
        X_processed,
        dtype=np.float32
    )

    return X_processed


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_traffic(model, X_processed):
    """Predict benign and malicious network traffic."""

    predictions = model.predict(X_processed)

    probabilities = model.predict_proba(
        X_processed
    )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The loaded model does not contain the attack class (1)."
        )

    attack_index = classes.index(1)

    attack_probabilities = probabilities[
        :, attack_index
    ]

    confidence = np.max(
        probabilities,
        axis=1
    )

    return (
        predictions,
        attack_probabilities,
        confidence
    )


# ============================================================
# PREDICTION REPORT
# ============================================================

def create_prediction_report(
    original_df,
    predictions,
    attack_probabilities,
    confidence,
    actual_labels=None
):
    """Create detailed prediction results."""

    results = original_df.copy()

    results["predicted_label"] = predictions

    results["prediction"] = np.where(
        predictions == 1,
        "Attack",
        "Benign"
    )

    results["attack_probability"] = attack_probabilities

    results["confidence"] = confidence

    if actual_labels is not None:

        results["actual_label"] = actual_labels.to_numpy()

        results["actual_class"] = actual_labels.map(
            {
                0: "Benign",
                1: "Attack"
            }
        ).to_numpy()

    return results


# ============================================================
# HTML COMPONENTS
# ============================================================

def show_page_header():

    st.html(textwrap.dedent("""
        <div class="page-header">

            <div class="page-eyebrow">
                NETWORK SECURITY / TRAFFIC ANALYSIS
            </div>

            <div class="page-title">
                Upload & Detection
            </div>

            <div class="page-description">
                Upload your network traffic dataset and let the
                machine learning detection engine identify
                suspicious activities and potential threats.
            </div>

        </div>
    """))


def show_instructions():

    st.html(textwrap.dedent("""
        <div class="instruction-panel">

            <div class="instruction-title">
                How detection works
            </div>

            <div class="instruction-grid">

                <div class="instruction-step">
                    <strong>01 / Upload</strong>
                    Select your network traffic CSV dataset.
                </div>

                <div class="instruction-step">
                    <strong>02 / Validate</strong>
                    Review the uploaded records and features.
                </div>

                <div class="instruction-step">
                    <strong>03 / Analyze</strong>
                    Run the trained Random Forest model.
                </div>

                <div class="instruction-step">
                    <strong>04 / Results</strong>
                    Explore predictions and download reports.
                </div>

            </div>

        </div>
    """))


def show_metric_card(title, value, color, description):

    st.html(textwrap.dedent(f"""
        <div class="metric-card {color}">

            <div class="metric-card-label">
                {title}
            </div>

            <div class="metric-card-value">
                {value}
            </div>

            <div class="metric-card-note">
                {description}
            </div>

        </div>
    """))


# ============================================================
# UPLOADED FILE INFORMATION
# ============================================================

def show_file_information(uploaded_file, df):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.html(textwrap.dedent(f"""
            <div class="file-card">

                <div class="file-label">
                    FILE NAME
                </div>

                <div class="file-value">
                    {uploaded_file.name}
                </div>

            </div>
        """))

    with col2:

        st.html(textwrap.dedent(f"""
            <div class="file-card">

                <div class="file-label">
                    TOTAL RECORDS
                </div>

                <div class="file-value">
                    {len(df):,}
                </div>

            </div>
        """))

    with col3:

        st.html(textwrap.dedent(f"""
            <div class="file-card">

                <div class="file-label">
                    TOTAL FEATURES
                </div>

                <div class="file-value">
                    {len(df.columns):,}
                </div>

            </div>
        """))


# ============================================================
# DETECTION SUMMARY
# ============================================================

def show_detection_summary(predictions):

    total_records = len(predictions)

    benign_count = int(
        np.sum(predictions == 0)
    )

    attack_count = int(
        np.sum(predictions == 1)
    )

    benign_percentage = (
        benign_count / total_records * 100
        if total_records else 0
    )

    attack_percentage = (
        attack_count / total_records * 100
        if total_records else 0
    )

    st.html(textwrap.dedent("""
        <div class="success-panel">

            <div class="success-title">
                Detection completed successfully
            </div>

            <div class="success-description">
                Your network traffic has been analyzed.
                Review the classification summary below.
            </div>

        </div>
    """))

    col1, col2, col3 = st.columns(3)

    with col1:

        show_metric_card(
            "TOTAL RECORDS",
            f"{total_records:,}",
            "metric-purple",
            "Records successfully analyzed"
        )

    with col2:

        show_metric_card(
            "BENIGN TRAFFIC",
            f"{benign_count:,}",
            "metric-green",
            f"{benign_percentage:.2f}% of all records"
        )

    with col3:

        show_metric_card(
            "SUSPICIOUS TRAFFIC",
            f"{attack_count:,}",
            "metric-red",
            f"{attack_percentage:.2f}% of all records"
        )

    # Distribution bar

    st.markdown("#### Traffic classification distribution")

    st.progress(
        benign_count / total_records if total_records else 0,
        text=f"Benign traffic: {benign_percentage:.2f}%"
    )

    st.progress(
        attack_count / total_records if total_records else 0,
        text=f"Malicious traffic: {attack_percentage:.2f}%"
    )


# ============================================================
# EXPECTED CSV FORMAT
# ============================================================

def show_expected_format():

    st.html(textwrap.dedent("""
        <div class="format-panel">

            <div class="section-heading">
                Expected CSV structure
            </div>

            <div class="section-subheading">
                Your dataset should contain features compatible
                with the UNSW-NB15 training pipeline.
            </div>

            <div class="format-code">

                dur, spkts, dpkts, sbytes, dbytes<br>
                proto, service, state<br>
                ...additional network traffic features

            </div>

        </div>
    """))

    with st.expander("View supported feature descriptions"):

        st.markdown("""
        | Feature | Description |
        |---|---|
        | dur | Network flow duration |
        | spkts | Source packets |
        | dpkts | Destination packets |
        | sbytes | Source bytes |
        | dbytes | Destination bytes |
        | proto | Network protocol |
        | service | Network service |
        | state | Connection state |

        **Optional columns:**

        - `label`: Actual class (0 = Benign, 1 = Attack).
        - `attack_cat`: Attack category, used for reference.
        - `id`: Record identifier, ignored during prediction.
        """)


# ============================================================
# MAIN PAGE
# ============================================================

def show():

    apply_design()

    show_page_header()

    show_instructions()

    # --------------------------------------------------------
    # UPLOAD SECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Dataset upload</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subheading">'
        'Choose a CSV file containing network traffic records.'
        '</div>',
        unsafe_allow_html=True
    )

    st.html(textwrap.dedent("""
        <div class="upload-panel">

            <div class="upload-title">
                Upload network traffic dataset
            </div>

            <div class="upload-description">
                Select a CSV file from your computer.
                Your file will be processed using the
                saved preprocessing pipeline and trained model.
            </div>

        </div>
    """))

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="Upload a CSV containing network traffic features.",
        key="network_csv_upload"
    )

    # --------------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.html(textwrap.dedent("""
            <div class="empty-panel">

                <div class="empty-title">
                    No dataset selected
                </div>

                <div class="empty-description">
                    Upload your network traffic CSV file to
                    begin the intrusion detection process.
                    <br><br>
                    Supported format: CSV
                </div>

            </div>
        """))

        st.markdown("---")

        show_expected_format()

        st.markdown("""
        <div class="page-footer">
            Network Intrusion Detection System<br>
            Machine Learning Based Network Traffic Analysis
        </div>
        """, unsafe_allow_html=True)

        return

    # --------------------------------------------------------
    # READ UPLOADED FILE
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            uploaded_file,
            low_memory=False
        )

        df.columns = df.columns.str.strip()

        # Remove unnamed columns

        df = df.loc[
            :,
            ~df.columns.str.match(r"^Unnamed")
        ]

        if df.empty:

            st.error(
                "The uploaded CSV contains no records."
            )

            return

        if len(df.columns) == 0:

            st.error(
                "The uploaded CSV contains no usable columns."
            )

            return

        # ----------------------------------------------------
        # FILE INFORMATION
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">File information</div>',
            unsafe_allow_html=True
        )

        show_file_information(
            uploaded_file,
            df
        )

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">Dataset preview</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subheading">'
            'Review the first records before running detection.'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            df.head(10),
            width="stretch",
            height=350
        )

        # ----------------------------------------------------
        # FEATURE VALIDATION
        # ----------------------------------------------------

        required_cols = [
            "dur",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:

            st.warning(
                "Some common UNSW-NB15 features are missing: "
                + ", ".join(missing_cols)
                + ". The preprocessing pipeline will align "
                  "the dataset with the model's expected features."
            )

        else:

            st.success(
                "The common network traffic features are present."
            )

        # ----------------------------------------------------
        # DETECTION BUTTON
        # ----------------------------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">Run intrusion detection</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subheading">'
            'Start the machine learning analysis of your dataset.'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Run Network Detection",
            type="primary",
            width="stretch",
            key="run_network_detection"
        ):

            try:

                # Load model

                with st.spinner(
                    "Loading trained model and preprocessing pipeline..."
                ):

                    model, preprocessor = (
                        load_model_and_preprocessor()
                    )

                    if model is None or preprocessor is None:

                        st.error(
                            "Unable to load the trained model. "
                            "Please check the models directory."
                        )

                        return

                # Prepare data

                with st.spinner(
                    "Preparing network traffic features..."
                ):

                    X, actual_labels = prepare_features(
                        df,
                        preprocessor
                    )

                # Transform

                with st.spinner(
                    "Transforming dataset..."
                ):

                    X_processed = transform_features(
                        X,
                        preprocessor
                    )

                # Predict

                with st.spinner(
                    "Analyzing network traffic..."
                ):

                    (
                        predictions,
                        attack_probabilities,
                        confidence
                    ) = predict_traffic(
                        model,
                        X_processed
                    )

                # Generate report

                with st.spinner(
                    "Generating prediction report..."
                ):

                    results = create_prediction_report(
                        df,
                        predictions,
                        attack_probabilities,
                        confidence,
                        actual_labels
                    )

                # Store results

                st.session_state.uploaded_data = df

                st.session_state.predictions = predictions

                st.session_state.prediction_results = results

                st.session_state.actual_labels_available = (
                    actual_labels is not None
                )

                st.session_state.detection_filename = (
                    uploaded_file.name
                )

                st.session_state.detection_completed = True

                st.rerun()

            except Exception as error:

                st.error(
                    f"Detection failed: {str(error)}"
                )

                st.info(
                    "Please verify that your CSV file contains "
                    "compatible network traffic features and "
                    "that the trained model is available."
                )

        # ----------------------------------------------------
        # DISPLAY COMPLETED RESULTS
        # ----------------------------------------------------

        if st.session_state.get(
            "detection_completed",
            False
        ):

            results = st.session_state.get(
                "prediction_results"
            )

            predictions = st.session_state.get(
                "predictions"
            )

            if (
                results is not None
                and predictions is not None
                and st.session_state.get("detection_filename")
                == uploaded_file.name
            ):

                st.markdown("---")

                show_detection_summary(
                    predictions
                )

                # Navigate to Results

                st.markdown("<br>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(
                    [1, 1.5, 1]
                )

                with col2:

                    if st.button(
                        "View Detailed Analytics",
                        type="primary",
                        width="stretch",
                        key="goto_results"
                    ):

                        st.session_state.current_page = "Results"

                        st.rerun()

    except Exception as error:

        st.error(
            f"Unable to process uploaded file: {str(error)}"
        )

        st.info(
            "Please upload a valid CSV file."
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown("""
    <div class="page-footer">

        Network Intrusion Detection System

        <br>

        Secure Analysis | Machine Learning | Network Intelligence

    </div>
    """, unsafe_allow_html=True)
