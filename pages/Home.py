import json
from pathlib import Path
from textwrap import dedent

import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"
PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
MODEL_DIR = PROJECT_DIR / "models"


# ============================================================
# DATA LOADING
# ============================================================

def load_json_file(file_path):
    """Safely load a JSON file."""

    if not file_path.exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        st.warning(f"Unable to load {file_path.name}: {error}")
        return None


def load_model_metrics():
    """Load model performance metrics."""

    return load_json_file(
        RESULTS_DIR / "model_metrics.json"
    )


def load_metadata():
    """Load dataset metadata."""

    return load_json_file(
        PROCESSED_DIR / "metadata.json"
    )


# ============================================================
# CUSTOM DESIGN
# ============================================================

def apply_home_design():

    render_html(dedent("""
    <style>

    .stApp {
        background: radial-gradient(ellipse at 10% 0%, rgba(79, 70, 229, 0.14), transparent 40%),
                    radial-gradient(ellipse at 95% 15%, rgba(6, 182, 212, 0.10), transparent 40%),
                    #080d1a;
        color: #e2e8f0;
    }

    .block-container {
        max-width: 1500px;
        padding: 1.8rem 1rem 3rem;
    }

    .hero-container {
        position: relative;
        overflow: hidden;
        padding: 65px 35px;
        border-radius: 26px;
        text-align: center;
        background: linear-gradient(135deg, rgba(49, 46, 129, 0.35), rgba(15, 23, 42, 0.95), rgba(8, 145, 178, 0.16));
        border: 1px solid rgba(129, 140, 248, 0.25);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 25px;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 17px;
        border-radius: 50px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(129, 140, 248, 0.35);
        color: #c7d2fe;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 23px;
    }

    .hero-title {
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        font-weight: 900;
        line-height: 1.15;
        letter-spacing: -2px;
        margin-bottom: 18px;
        background:
            linear-gradient(
                90deg,
                #ffffff,
                #a5b4fc,
                #67e8f9
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-description {
        max-width: 780px;
        margin: auto;
        color: #a5b4c8;
        font-size: 1.12rem;
        line-height: 1.8;
    }

    .hero-bottom {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 28px;
    }

    .hero-feature {
        padding: 9px 16px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 800;
        margin: 12px 0 5px;
    }

    .section-description {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 22px;
    }

    .stat-card {
        background: linear-gradient(145deg, rgba(22, 32, 55, 0.95), rgba(13, 20, 37, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 17px;
        padding: 23px 20px;
        min-height: 145px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.17);
        transition: all 0.25s ease;
    }

    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(129, 140, 248, 0.5);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
    }

    .stat-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .stat-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 850;
        letter-spacing: -1px;
    }

    .stat-footer {
        color: #8493aa;
        font-size: 0.77rem;
        margin-top: 7px;
    }

    .purple {
        border-left: 4px solid #818cf8;
    }

    .cyan {
        border-left: 4px solid #22d3ee;
    }

    .green {
        border-left: 4px solid #34d399;
    }

    .pink {
        border-left: 4px solid #f472b6;
    }

    .glass-card {
        background: rgba(17, 26, 46, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 18px;
        padding: 23px;
        margin-bottom: 12px;
    }

    .glass-card-title {
        color: #f1f5f9;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .glass-card-text {
        color: #a5b4c8;
        font-size: 0.88rem;
        line-height: 1.8;
    }

    .workflow-card {
        background: linear-gradient(145deg, rgba(22, 32, 55, 0.95), rgba(13, 20, 37, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 22px 18px;
        min-height: 185px;
        transition: all 0.25s ease;
    }

    .workflow-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.55);
    }

    .workflow-number {
        color: #818cf8;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }


    .workflow-title {
        color: #f8fafc;
        font-weight: 750;
        font-size: 1rem;
        margin-bottom: 9px;
    }

    .workflow-description {
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    .config-item {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        padding: 13px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        font-size: 0.87rem;
    }

    .config-label {
        color: #94a3b8;
    }

    .config-value {
        color: #c7d2fe;
        font-weight: 650;
        text-align: right;
    }

    .home-footer {
        text-align: center;
        padding: 28px 10px;
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.8;
    }

    [data-testid="stMetric"] {
        background: rgba(22, 32, 55, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 14px;
        padding: 17px;
    }

    [data-testid="stMetricLabel"] {
        color: #a5b4c8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    .stButton button {
        border-radius: 12px;
        font-weight: 750;
        min-height: 50px;
        transition: all 0.25s ease;
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);
    }

    hr {
        border-color: rgba(148, 163, 184, 0.15);
        margin-top: 28px;
        margin-bottom: 28px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """))


# ============================================================
# REUSABLE COMPONENTS
# ============================================================

def render_html(markup):
    """Render an HTML snippet (dedent is pre-applied in the function calls now)."""
    st.html(markup)


def section_header(title, description=""):

    render_html(dedent(f"""
        <div class="section-title">{title}</div>
        <div class="section-description">{description}</div>
    """))


def create_stat_card(
    title,
    value,
    color,
    description
):

    render_html(dedent(f"""
        <div class="stat-card {color}">
            <div class="stat-label">
                {title}
            </div>
            <div class="stat-value">
                {value}
            </div>
            <div class="stat-footer">
                {description}
            </div>
        </div>
    """))


def format_percentage(value):

    try:
        return f"{float(value) * 100:.2f}%"

    except (TypeError, ValueError):
        return "N/A"


def format_number(value):

    try:
        return f"{int(value):,}"

    except (TypeError, ValueError):
        return "N/A"


# ============================================================
# PERFORMANCE CONFUSION MATRIX
# ============================================================

def show_confusion_matrix(metrics):

    tn = metrics.get("true_negative", 0)
    fp = metrics.get("false_positive", 0)
    fn = metrics.get("false_negative", 0)
    tp = metrics.get("true_positive", 0)

    cm = [
        [tn, fp],
        [fn, tp]
    ]

    fig = go.Figure(

        data=go.Heatmap(

            z=cm,

            x=[
                "Predicted Benign",
                "Predicted Attack"
            ],

            y=[
                "Actual Benign",
                "Actual Attack"
            ],

            colorscale=[
                [0, "#111c35"],
                [0.5, "#4338ca"],
                [1, "#22d3ee"]
            ],

            text=cm,

            texttemplate="%{text}",

            textfont=dict(
                size=22,
                color="white"
            ),

            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}<br>"
                "Records: %{z:,}"
                "<extra></extra>"
            ),

            showscale=True
        )
    )

    fig.update_layout(

        height=390,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#cbd5e1",
            size=12
        ),

        xaxis_title="Predicted Classification",

        yaxis_title="Actual Classification",

        margin=dict(
            l=20,
            r=20,
            t=25,
            b=30
        )
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False}
    )


# ============================================================
# HERO SECTION
# ============================================================

def show_hero():

    render_html(dedent("""
        <div class="hero-container">
            <div class="hero-badge">
                INTELLIGENT NETWORK SECURITY
            </div>
            <div class="hero-title">
                Network Intrusion<br>
                Detection System
            </div>
            <div class="hero-description">
                Protect your network with intelligent
                machine learning-powered intrusion detection.
                <br><br>
                Analyze network traffic, identify malicious
                activities, and gain actionable security insights
                through advanced AI-driven analytics.
            </div>
            <div class="hero-bottom">
                <div class="hero-feature">
                    Random Forest AI
                </div>
                <div class="hero-feature">
                    Real-Time Detection
                </div>
                <div class="hero-feature">
                    Advanced Analytics
                </div>
                <div class="hero-feature">
                    Network Protection
                </div>
            </div>
        </div>
    """))

    col1, col2, col3 = st.columns([1, 1.6, 1])

    with col2:
        if st.button(
            "Analyze Your Network Traffic",
            key="cta_upload",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.current_page = "Upload"
            st.rerun()


# ============================================================
# DATASET OVERVIEW
# ============================================================

def show_dataset_overview(metadata):

    section_header(
        "Dataset Overview",
        "An overview of the data used to train and evaluate the intrusion detection model."
    )

    if not metadata:

        st.warning(
            "Dataset metadata was not found. "
            "Please check dataset/processed/metadata.json."
        )

        return

    col1, col2, col3 = st.columns(3)

    with col1:

        create_stat_card(
            "Training Records",
            format_number(
                metadata.get("training_records", 0)
            ),
            "purple",
            "Records used for model training"
        )

    with col2:

        create_stat_card(
            "Testing Records",
            format_number(
                metadata.get("testing_records", 0)
            ),
            "cyan",
            "Records used for model evaluation"
        )

    with col3:

        create_stat_card(
            "Total Features",
            format_number(
                metadata.get("transformed_feature_count", 0)
            ),
            "green",
            "Features after preprocessing"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def show_model_performance(metrics):

    section_header(
        "Model Performance",
        "Evaluate how effectively the machine learning model identifies network threats."
    )

    if not metrics:

        st.warning(
            "Model metrics were not found. "
            "Please train the model first."
        )

        return

    # PRIMARY METRICS

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        create_stat_card(
            "Accuracy",
            format_percentage(
                metrics.get("accuracy")
            ),
            "purple",
            "Overall classification correctness"
        )

    with col2:

        create_stat_card(
            "Precision",
            format_percentage(
                metrics.get("precision")
            ),
            "green",
            "Correctness of attack predictions"
        )

    with col3:

        create_stat_card(
            "Recall",
            format_percentage(
                metrics.get("recall")
            ),
            "green",
            "Ability to detect actual attacks"
        )

    with col4:

        create_stat_card(
            "F1-Score",
            format_percentage(
                metrics.get("f1_score")
            ),
            "purple",
            "Balance between precision and recall"
        )

    st.html("<div style=\"height: 12px;\"></div>")

    # SECONDARY METRICS

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "ROC-AUC",
            (
                f"{float(metrics['roc_auc']):.4f}"
                if metrics.get("roc_auc") is not None
                else "N/A"
            )
        )

    with col2:

        st.metric(
            "Specificity",
            format_percentage(
                metrics.get("specificity")
            )
        )

    with col3:

        st.metric(
            "False Positive Rate",
            format_percentage(
                metrics.get("false_positive_rate")
            )
        )

    # CONFUSION MATRIX

    st.markdown("---")

    section_header(
        "Confusion Matrix",
        "Visual representation of correct and incorrect model classifications."
    )

    if all(
        key in metrics
        for key in [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive"
        ]
    ):

        show_confusion_matrix(metrics)

    else:

        st.info(
            "Confusion matrix values are unavailable."
        )


# ============================================================
# TRAINING INFORMATION
# ============================================================

def show_training_information(metrics):

    section_header(
        "Training Information",
        "Details about model training and prediction performance."
    )

    if not metrics:

        st.info(
            "Training information is not available."
        )

        return

    col1, col2 = st.columns(2)

    with col1:

        training_time = metrics.get(
            "training_time_seconds"
        )

        st.metric(
            "Training Time",

            (
                f"{float(training_time):.2f} seconds"
                if training_time is not None
                else "N/A"
            )
        )

    with col2:

        prediction_time = metrics.get(
            "prediction_time_seconds"
        )

        st.metric(
            "Prediction Time",

            (
                f"{float(prediction_time):.2f} seconds"
                if prediction_time is not None
                else "N/A"
            )
        )


# ============================================================
# DATASET DETAILS
# ============================================================

def show_dataset_details(metadata):

    section_header(
        "Dataset Information",
        "Explore the dataset structure and preprocessing configuration."
    )

    if not metadata:

        st.info("Dataset information is unavailable.")

        return

    col1, col2 = st.columns(2)

    with col1:

        numerical_features = metadata.get(
            "original_numerical_features",
            []
        )

        categorical_features = metadata.get(
            "original_categorical_features",
            []
        )

        render_html(dedent(f"""
        <div class="glass-card">
            <div class="glass-card-title">
                Dataset Configuration
            </div>
            <div class="config-item">
                <span class="config-label">Dataset</span>
                <span class="config-value">
                    {metadata.get("dataset", "N/A")}
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Target Column</span>
                <span class="config-value">
                    {metadata.get("target_column", "N/A")}
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Numerical Features</span>
                <span class="config-value">
                    {len(numerical_features)}
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Categorical Features</span>
                <span class="config-value">
                    {len(categorical_features)}
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Transformed Features</span>
                <span class="config-value">
                    {metadata.get("transformed_feature_count", "N/A")}
                </span>
            </div>
        </div>
        """))

    with col2:

        target_mapping = metadata.get(
            "target_mapping",
            {}
        )

        benign_label = target_mapping.get(
            "0",
            "N/A"
        )

        attack_label = target_mapping.get(
            "1",
            "N/A"
        )

        duplicate_removal = metadata.get(
            "duplicate_removal"
        )

        duplicate_status = (
            "Enabled"
            if duplicate_removal is True
            else "Disabled"
            if duplicate_removal is False
            else "N/A"
        )

        render_html(dedent(f"""
        <div class="glass-card">
            <div class="glass-card-title">
                Target & Preprocessing
            </div>
            <div class="config-item">
                <span class="config-label">Class 0</span>
                <span class="config-value">
                    {benign_label}
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Class 1</span>
                <span class="config-value">
                    {attack_label}
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Duplicate Removal</span>
                <span class="config-value">
                    {duplicate_status}
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Model Type</span>
                <span class="config-value">
                    Random Forest
                </span>
            </div>
            <div class="config-item">
                <span class="config-label">Preprocessing</span>
                <span class="config-value">
                    Saved Pipeline
                </span>
            </div>
        </div>
        """))


# ============================================================
# SYSTEM WORKFLOW
# ============================================================

def show_workflow():

    section_header(
        "How It Works",
        "A simple four-step process for identifying malicious network traffic."
    )

    col1, col2, col3, col4 = st.columns(4)

    workflow = [
        (
            col1,
            "01",
            "Upload Dataset",
            "Upload your network traffic CSV file through the detection interface."
        ),

        (
            col2,
            "02",
            "Preprocessing",
            "Clean and transform network data using the saved preprocessing pipeline."
        ),

        (
            col3,
            "03",
            "AI Detection",
            "The Random Forest classifier analyzes traffic and predicts potential attacks."
        ),

        (
            col4,
            "04",
            "View Analytics",
            "Explore predictions, investigate suspicious records, and export reports."
        )
    ]

    for column, number, title, description in workflow:

        with column:

            render_html(dedent(f"""
            <div class="workflow-card">
                <div class="workflow-number">
                    STEP {number}
                </div>
                <div class="workflow-title">
                    {title}
                </div>
                <div class="workflow-description">
                    {description}
                </div>
            </div>
            """))


# ============================================================
# MODEL DETAILS
# ============================================================

def show_model_details():

    section_header(
        "Machine Learning Model",
        "Technical configuration of the intrusion detection engine."
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:

        render_html(dedent("""
        <div class="glass-card">
            <div class="glass-card-title">
                Random Forest Classifier
            </div>
            <div class="glass-card-text">
                The system uses a Random Forest classification
                algorithm to distinguish between normal and
                malicious network traffic.
                <br><br>
                Random Forest combines multiple decision trees
                to improve classification reliability and
                reduce overfitting compared with a single tree.
                <br><br>
                The model is trained using the UNSW-NB15
                network intrusion detection dataset.
            </div>
        </div>
        """))

    with col2:

        render_html(dedent("""
        <div class="glass-card">
            <div class="glass-card-title">
                Model Configuration
            </div>
            <div class="config-item">
                <span class="config-label">Algorithm</span>
                <span class="config-value">Random Forest</span>
            </div>
            <div class="config-item">
                <span class="config-label">Number of Trees</span>
                <span class="config-value">200</span>
            </div>
            <div class="config-item">
                <span class="config-label">Max Features</span>
                <span class="config-value">sqrt</span>
            </div>
            <div class="config-item">
                <span class="config-label">Class Weight</span>
                <span class="config-value">balanced_subsample</span>
            </div>
            <div class="config-item">
                <span class="config-label">Random State</span>
                <span class="config-value">42</span>
            </div>
            <div class="config-item">
                <span class="config-label">Criterion</span>
                <span class="config-value">Gini</span>
            </div>
            <div class="config-item">
                <span class="config-label">Bootstrap</span>
                <span class="config-value">Enabled</span>
            </div>
        </div>
        """))

    # MODEL FILES

    st.markdown("#### Saved Model Files")

    model_file = MODEL_DIR / "random_forest_ids.joblib"

    preprocessor_file = MODEL_DIR / "preprocessor.joblib"

    col1, col2 = st.columns(2)

    with col1:

        if model_file.exists():

            st.success(
                "random_forest_ids.joblib — Available"
            )

        else:

            st.warning(
                "random_forest_ids.joblib — Not found"
            )

    with col2:

        if preprocessor_file.exists():

            st.success(
                "preprocessor.joblib — Available"
            )

        else:

            st.warning(
                "preprocessor.joblib — Not found"
            )


# ============================================================
# MAIN HOME PAGE
# ============================================================

def show():

    # Apply theme
    apply_home_design()

    # Load information
    metrics = load_model_metrics()
    metadata = load_metadata()

    # 1. HERO
    show_hero()

    st.markdown("---")

    # 2. INTRODUCTION
    section_header(
        "Intelligent Network Protection",
        "Machine learning-powered network intrusion detection and threat analysis."
    )

    render_html(dedent("""
    <div class="glass-card">
        <div class="glass-card-title">
            What Does This System Do?
        </div>
        <div class="glass-card-text">
            Network Intrusion Detection System (NIDS) is an
            intelligent cybersecurity application designed to
            identify suspicious and malicious network traffic.
            <br><br>
            Using a trained <b>Random Forest classifier</b>
            and the <b>UNSW-NB15 dataset</b>, the system
            analyzes network traffic patterns and classifies
            each record as either benign or malicious.
            <br><br>
            Upload a CSV file, run the detection engine,
            explore detailed analytics, and download
            suspicious traffic reports.
        </div>
    </div>
    """))

    st.markdown("---")

    # 3. DATASET OVERVIEW
    show_dataset_overview(metadata)

    st.markdown("---")

    # 4. MODEL PERFORMANCE
    show_model_performance(metrics)

    st.markdown("---")

    # 5. TRAINING INFORMATION
    show_training_information(metrics)

    st.markdown("---")

    # 6. DATASET DETAILS
    show_dataset_details(metadata)

    st.markdown("---")

    # 7. WORKFLOW
    show_workflow()

    st.markdown("---")

    # 8. MODEL DETAILS
    show_model_details()

    st.markdown("---")

    # 9. FINAL CTA
    section_header(
        "Ready to Secure Your Network?",
        "Start analyzing network traffic and discover potential security threats."
    )

    col1, col2, col3 = st.columns([1, 1.6, 1])

    with col2:

        if st.button(
            "Start Network Intrusion Detection",
            key="home_bottom_cta",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.current_page = "Upload"
            st.rerun()

    # 10. FOOTER
    render_html(dedent("""
    <div class="home-footer">
        <b>Network Intrusion Detection System</b>
        <br>
        Intelligent Security • Machine Learning • Network Analytics
        <br><br>
        Built with Python, Streamlit, Scikit-learn and Plotly
    </div>
    """))