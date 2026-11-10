
"""
Premium Results & Analytics Dashboard
Network Intrusion Detection System (NIDS)

Features:
- Modern cybersecurity dashboard UI
- Premium dark glassmorphism design
- Interactive analytics charts
- Prediction summary cards
- Model evaluation metrics
- Confusion matrix
- Classification report
- Searchable and filterable results
- CSV export functionality
- Prediction error analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from io import BytesIO
import textwrap
from html import escape

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)




# ============================================================
# SAFE HTML RENDERING
# ============================================================

def render_html(content):
    """Render HTML safely through Streamlit Markdown after removing indentation."""
    html_content = textwrap.dedent(str(content)).strip()
    st.markdown(html_content, unsafe_allow_html=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

def apply_custom_design():

    st.markdown(textwrap.dedent("""
    <style>

    /* Main application background */
    .stApp {
        background:
            radial-gradient(
                ellipse at 10% 0%,
                rgba(79, 70, 229, 0.12),
                transparent 40%
            ),
            radial-gradient(
                ellipse at 90% 20%,
                rgba(6, 182, 212, 0.08),
                transparent 40%
            ),
            #080d1a;

        color: #e5e7eb;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Main title */
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #f1f5f9;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(
            145deg,
            rgba(22, 32, 55, 0.95),
            rgba(13, 20, 37, 0.95)
        );

        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 18px;

        padding: 23px 20px;

        min-height: 145px;

        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);

        transition: all 0.25s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(129, 140, 248, 0.5);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
    }

    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 17px;
    }

    .kpi-label {
        color: #a5b4c8;
        overflow-wrap: anywhere;
        word-break: break-word;
        font-size: 0.86rem;
        font-weight: 600;
    }

    .kpi-icon {
        font-size: 1.4rem;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 2rem;
        overflow-wrap: anywhere;
        word-break: break-word;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .kpi-footer {
        color: #8493aa;
        font-size: 0.77rem;
        margin-top: 7px;
    }

    /* Accent colors */
    .purple {
        border-left: 4px solid #818cf8;
    }

    .green {
        border-left: 4px solid #34d399;
    }

    .red {
        border-left: 4px solid #fb7185;
    }

    .cyan {
        border-left: 4px solid #22d3ee;
    }

    /* Dashboard panels */
    .dashboard-panel {
        background: rgba(17, 26, 46, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 18px;
        padding: 20px;
    }

    /* Status labels */
    .status-normal {
        color: #34d399;
        font-weight: 700;
    }

    .status-attack {
        color: #fb7185;
        font-weight: 700;
    }

    /* Information banner */
    .info-banner {
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 12px;
        padding: 15px 18px;
        color: #c7d2fe;
        margin-bottom: 20px;
    }

    /* Download cards */
    .download-card {
        background: linear-gradient(
            145deg,
            rgba(22, 32, 55, 0.95),
            rgba(13, 20, 37, 0.95)
        );

        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 15px;
        padding: 20px;
        min-height: 120px;
    }

    .download-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .download-description {
        font-size: 0.83rem;
        color: #94a3b8;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(22, 32, 55, 0.8);
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

    /* Input controls */
    .stSelectbox label,
    .stTextInput label,
    .stMultiSelect label {
        color: #cbd5e1 !important;
        font-weight: 600;
    }

    /* Buttons */
    .stDownloadButton button {
        border-radius: 10px;
        min-height: 43px;
        font-weight: 650;
        transition: all 0.2s ease;
    }

    .stDownloadButton button:hover {
        transform: translateY(-2px);
    }

    /* Divider */
    hr {
        border-color: rgba(148, 163, 184, 0.15);
        margin-top: 25px;
        margin-bottom: 25px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #a5b4fc !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hide unnecessary decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """), unsafe_allow_html=True)


# ============================================================
# PLOTLY THEME
# ============================================================

def style_chart(fig, height=380):

    fig.update_layout(
        height=height,

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            family="Arial",
            color="#cbd5e1",
            size=12
        ),

        title_font=dict(
            size=16,
            color="#f1f5f9"
        ),

        margin=dict(
            l=20,
            r=20,
            t=55,
            b=30
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        ),

        hoverlabel=dict(
            bgcolor="#172033",
            bordercolor="#6366f1",
            font=dict(color="white")
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.15)"
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.15)"
    )

    return fig


# ============================================================
# KPI CARD
# ============================================================

def create_kpi_card(title, value, icon, color, description):

    render_html(f"""
    <div class="kpi-card {color}">

        <div class="kpi-header">

            <div class="kpi-label">
                {escape(str(title))}
            </div>

            <div class="kpi-icon">
                {icon}
            </div>

        </div>

        <div class="kpi-value">
            {escape(str(value))}
        </div>

        <div class="kpi-footer">
            {escape(str(description))}
        </div>

    </div>
    """)


# ============================================================
# SECTION HEADER
# ============================================================

def section_header(title, description=""):

    render_html(
        f"""
        <div class="section-title">{escape(str(title))}</div>
        <div class="section-description">{escape(str(description))}</div>
        """,
    )


# ============================================================
# CSV EXPORT
# ============================================================

def dataframe_to_csv(df):

    buffer = BytesIO()

    buffer.write(
        df.to_csv(index=False).encode("utf-8")
    )

    buffer.seek(0)

    return buffer


# ============================================================
# MAIN PAGE
# ============================================================

def show():

    apply_custom_design()

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    render_html("""
    <div class="main-title">
        Security Analytics
    </div>

    <div class="main-subtitle">
        Network Intrusion Detection System
        <br>
        Monitor network traffic, analyze threats, and evaluate
        intrusion detection performance.
    </div>
    """)

    # --------------------------------------------------------
    # CHECK PREDICTIONS
    # --------------------------------------------------------

    if (
        "prediction_results" not in st.session_state
        or st.session_state.prediction_results is None
    ):

        render_html("""
        <div class="info-banner">
            <b>📡 No prediction data available</b>
            <br><br>
            Upload a network traffic dataset and run intrusion
            detection from the Upload & Detection page to view
            analytics here.
        </div>
        """)

        return

    results = st.session_state.prediction_results.copy()

    if results.empty:

        st.warning("The prediction results dataset is empty.")
        return

    if "predicted_label" not in results.columns:

        st.error("Missing required column: predicted_label")
        return

    predictions = results["predicted_label"].values

    total_records = len(results)

    benign_count = int(np.sum(predictions == 0))
    attack_count = int(np.sum(predictions == 1))

    other_count = total_records - benign_count - attack_count

    attack_percentage = (
        attack_count / total_records * 100
        if total_records > 0 else 0
    )

    benign_percentage = (
        benign_count / total_records * 100
        if total_records > 0 else 0
    )

    # --------------------------------------------------------
    # PREDICTION OVERVIEW
    # --------------------------------------------------------

    section_header(
        "📈 Prediction Overview",
        "A real-time snapshot of the analyzed network traffic."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        create_kpi_card(
            "Total Records",
            f"{total_records:,}",
            "📁",
            "purple",
            "Total analyzed network connections"
        )

    with col2:

        create_kpi_card(
            "Normal Traffic",
            f"{benign_count:,}",
            "🟢",
            "green",
            f"{benign_percentage:.2f}% of total traffic"
        )

    with col3:

        create_kpi_card(
            "Malicious Traffic",
            f"{attack_count:,}",
            "🚨",
            "red",
            f"{attack_percentage:.2f}% of total traffic"
        )

    with col4:

        create_kpi_card(
            "Attack Detection",
            f"{attack_percentage:.2f}%",
            "🎯",
            "cyan",
            "Percentage classified as malicious"
        )

    if other_count > 0:

        st.caption(
            f"Note: {other_count:,} records contain labels "
            "other than 0 or 1."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # TRAFFIC ANALYTICS
    # --------------------------------------------------------

    section_header(
        "📊 Traffic Analytics",
        "Visual insights into normal and malicious network activity."
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # PIE CHART
    # --------------------------------------------------------

    with col1:

        st.markdown("#### 🎯 Traffic Classification")

        labels = ["Normal Traffic", "Malicious Traffic"]
        values = [benign_count, attack_count]

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,

                    hole=0.68,

                    marker=dict(
                        colors=["#34d399", "#fb7185"],
                        line=dict(
                            color="#101827",
                            width=3
                        )
                    ),

                    textinfo="percent",
                    textfont=dict(
                        size=14,
                        color="white"
                    ),

                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Records: %{value:,}<br>"
                        "Percentage: %{percent}"
                        "<extra></extra>"
                    )
                )
            ]
        )

        fig_pie.add_annotation(
            text=(
                f"<b>{total_records:,}</b>"
                "<br><span style='font-size:11px'>Records</span>"
            ),

            x=0.5,
            y=0.5,

            showarrow=False,

            font=dict(
                size=21,
                color="#f8fafc"
            )
        )

        fig_pie = style_chart(fig_pie, 370)

        st.plotly_chart(
            fig_pie,
            width="stretch",
            config={"displayModeBar": False}
        )

    # --------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    with col2:

        st.markdown("#### 📉 Attack Probability Distribution")

        if "attack_probability" in results.columns:

            probability = pd.to_numeric(
                results["attack_probability"],
                errors="coerce"
            ).dropna()

            if not probability.empty:

                fig_hist = px.histogram(
                    x=probability,

                    nbins=40,

                    labels={
                        "x": "Attack Probability",
                        "count": "Number of Records"
                    },

                    color_discrete_sequence=["#818cf8"]
                )

                fig_hist.update_traces(
                    marker_line_color="#c4b5fd",
                    marker_line_width=1,
                    opacity=0.85
                )

                fig_hist.update_layout(
                    bargap=0.08,
                    xaxis_range=[0, 1]
                )

                fig_hist = style_chart(fig_hist, 370)

                st.plotly_chart(
                    fig_hist,
                    width="stretch",
                    config={"displayModeBar": False}
                )

            else:

                st.info("No valid probability values available.")

        else:

            st.info(
                "Attack probability information is not available."
            )

    # --------------------------------------------------------
    # TRAFFIC DISTRIBUTION BAR CHART
    # --------------------------------------------------------

    st.markdown("#### 📊 Network Traffic Breakdown")

    traffic_df = pd.DataFrame({
        "Traffic Type": [
            "Normal Traffic",
            "Malicious Traffic"
        ],

        "Records": [
            benign_count,
            attack_count
        ]
    })

    fig_bar = px.bar(
        traffic_df,

        x="Traffic Type",
        y="Records",

        color="Traffic Type",

        color_discrete_map={
            "Normal Traffic": "#34d399",
            "Malicious Traffic": "#fb7185"
        },

        text="Records"
    )

    fig_bar.update_traces(
        textposition="outside",
        textfont=dict(
            color="#f8fafc",
            size=13
        ),

        marker_line_width=0
    )

    fig_bar = style_chart(fig_bar, 320)

    fig_bar.update_layout(
        showlegend=False,
        yaxis_title="Number of Records",
        xaxis_title=""
    )

    st.plotly_chart(
        fig_bar,
        width="stretch",
        config={"displayModeBar": False}
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    actual_labels_available = (
        st.session_state.get(
            "actual_labels_available",
            False
        )
    )

    if actual_labels_available:

        section_header(
            "🎯 Model Performance",
            "Evaluate the intrusion detection model against actual labels."
        )

        if "actual_label" not in results.columns:

            st.warning("Actual label column is missing.")

        else:

            valid_mask = results["actual_label"].notna()

            actual = results.loc[
                valid_mask,
                "actual_label"
            ]

            actual_numeric = pd.to_numeric(
                actual,
                errors="coerce"
            )

            predicted_numeric = pd.to_numeric(
                results["predicted_label"],
                errors="coerce"
            )

            # Evaluate only rows where both actual and predicted labels
            # are valid binary values. Keep the original row indexes aligned.
            evaluation_mask = (
                results["actual_label"].notna()
                & actual_numeric.isin([0, 1])
                & predicted_numeric.isin([0, 1])
            )

            actual = actual_numeric.loc[evaluation_mask].astype(int).to_numpy()
            predicted = predicted_numeric.loc[evaluation_mask].astype(int).to_numpy()

            if len(actual) > 0:

                accuracy = accuracy_score(
                    actual,
                    predicted
                )

                precision = precision_score(
                    actual,
                    predicted,
                    zero_division=0
                )

                recall = recall_score(
                    actual,
                    predicted,
                    zero_division=0
                )

                f1 = f1_score(
                    actual,
                    predicted,
                    zero_division=0
                )

                st.caption(
                    f"Evaluation based on {len(actual):,} "
                    "valid labeled records."
                )

                # Performance metrics

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Accuracy",
                        f"{accuracy * 100:.2f}%",
                        help="Percentage of correctly classified records."
                    )

                with col2:

                    st.metric(
                        "Precision",
                        f"{precision * 100:.2f}%",
                        help="Percentage of predicted attacks that were actual attacks."
                    )

                with col3:

                    st.metric(
                        "Recall",
                        f"{recall * 100:.2f}%",
                        help="Percentage of actual attacks detected."
                    )

                with col4:

                    st.metric(
                        "F1-Score",
                        f"{f1 * 100:.2f}%",
                        help="Harmonic mean of precision and recall."
                    )

                # ------------------------------------------------
                # PERFORMANCE COMPARISON
                # ------------------------------------------------

                st.markdown("#### 📈 Performance Metrics Comparison")

                metrics_df = pd.DataFrame({
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1-Score"
                    ],

                    "Score": [
                        accuracy,
                        precision,
                        recall,
                        f1
                    ]
                })

                fig_metrics = px.bar(
                    metrics_df,

                    x="Metric",
                    y="Score",

                    text=metrics_df["Score"].apply(
                        lambda x: f"{x * 100:.2f}%"
                    ),

                    color="Score",

                    color_continuous_scale=[
                        "#6366f1",
                        "#22d3ee",
                        "#34d399"
                    ]
                )

                fig_metrics.update_traces(
                    textposition="outside",
                    textfont=dict(color="#f8fafc")
                )

                fig_metrics.update_layout(
                    yaxis_range=[0, 1.15],
                    coloraxis_showscale=False,
                    yaxis_title="Performance Score",
                    xaxis_title=""
                )

                fig_metrics = style_chart(
                    fig_metrics,
                    350
                )

                st.plotly_chart(
                    fig_metrics,
                    width="stretch",
                    config={"displayModeBar": False}
                )

                st.markdown("---")

                # ------------------------------------------------
                # CONFUSION MATRIX
                # ------------------------------------------------

                section_header(
                    "🔍 Confusion Matrix",
                    "Understand how accurately the model classifies each traffic category."
                )

                cm = confusion_matrix(
                    actual,
                    predicted,
                    labels=[0, 1]
                )

                cm_fig = go.Figure(
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

                        showscale=True,

                        hovertemplate=(
                            "%{y}<br>"
                            "%{x}<br>"
                            "Records: %{z:,}"
                            "<extra></extra>"
                        )
                    )
                )

                cm_fig.update_layout(
                    height=400,
                    xaxis_title="Predicted Classification",
                    yaxis_title="Actual Classification"
                )

                cm_fig = style_chart(cm_fig, 400)

                st.plotly_chart(
                    cm_fig,
                    width="stretch",
                    config={"displayModeBar": False}
                )

                # ------------------------------------------------
                # CLASSIFICATION REPORT
                # ------------------------------------------------

                section_header(
                    "📋 Classification Report",
                    "Detailed precision, recall, and F1-score for each class."
                )

                report = classification_report(
                    actual,
                    predicted,

                    labels=[0, 1],

                    target_names=[
                        "Benign",
                        "Attack"
                    ],

                    zero_division=0,
                    output_dict=True
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df.style.format(
                        "{:.4f}",
                        na_rep="-"
                    ),

                    width="stretch"
                )

            else:

                st.warning(
                    "No valid actual labels are available "
                    "for model evaluation."
                )

    else:

        render_html("""
        <div class="info-banner">
            <b>ℹ️ Model evaluation is currently unavailable</b>
            <br><br>
            Actual ground-truth labels were not provided.
            Prediction analytics are available, but accuracy,
            precision, recall, and F1-score require actual labels.
        </div>
        """)

    st.markdown("---")

    # --------------------------------------------------------
    # DETAILED PREDICTION RESULTS
    # --------------------------------------------------------

    section_header(
        "📋 Detailed Prediction Results",
        "Explore individual network traffic predictions and investigate suspicious activity."
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    filter_col1, filter_col2, filter_col3 = st.columns(
        [1.2, 1.5, 1]
    )

    with filter_col1:

        filter_option = st.selectbox(
            "Prediction Category",

            [
                "All Records",
                "Normal Traffic Only",
                "Malicious Traffic Only"
            ],

            key="analytics_prediction_filter"
        )

    with filter_col2:

        search_term = st.text_input(
            "🔎 Search Records",

            placeholder="Search values or column names...",

            key="analytics_search"
        )

    with filter_col3:

        rows_to_display = st.selectbox(
            "Rows to Display",

            [20, 50, 100, 250, 500],

            index=2,

            key="analytics_rows"
        )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_results = results.copy()

    if filter_option == "Normal Traffic Only":

        filtered_results = filtered_results[
            filtered_results["predicted_label"] == 0
        ]

    elif filter_option == "Malicious Traffic Only":

        filtered_results = filtered_results[
            filtered_results["predicted_label"] == 1
        ]

    if search_term.strip():

        search_mask = filtered_results.astype(
            str
        ).apply(
            lambda row: row.str.contains(
                search_term.strip(),
                case=False,
                na=False,
                regex=False
            )
        ).any(axis=1)

        filtered_results = filtered_results[
            search_mask
        ]

    # --------------------------------------------------------
    # RESULTS COUNTER
    # --------------------------------------------------------

    render_html(
        f"""
        <div class="info-banner">
            <b>📊 Search Results</b>
            <br>
            Showing <b>{len(filtered_results):,}</b>
            out of <b>{total_records:,}</b> records.
        </div>
        """,
    )

    # --------------------------------------------------------
    # SELECT DISPLAY COLUMNS
    # --------------------------------------------------------

    preferred_columns = [
        "prediction",
        "actual_class",
        "attack_probability",
        "confidence"
    ]

    display_columns = [
        col for col in preferred_columns
        if col in filtered_results.columns
    ]

    excluded_columns = [
        "predicted_label",
        "actual_label"
    ]

    feature_columns = [
        col for col in filtered_results.columns

        if col not in display_columns
        and col not in excluded_columns
    ]

    display_columns.extend(
        feature_columns[:5]
    )

    # Remove duplicate columns

    display_columns = list(
        dict.fromkeys(display_columns)
    )

    # --------------------------------------------------------
    # DISPLAY TABLE
    # --------------------------------------------------------

    if not filtered_results.empty:

        display_df = filtered_results[
            display_columns
        ].head(rows_to_display).copy()

        def highlight_prediction(row):

            prediction = row.get(
                "prediction",
                ""
            )

            if prediction == "Benign":

                return [
                    "background-color: rgba(52, 211, 153, 0.12); "
                    "color: #6ee7b7;"
                ] * len(row)

            elif prediction == "Attack":

                return [
                    "background-color: rgba(251, 113, 133, 0.12); "
                    "color: #fda4af;"
                ] * len(row)

            return [""] * len(row)

        if "prediction" in display_df.columns:

            styled_df = display_df.style.apply(
                highlight_prediction,
                axis=1
            )

            st.dataframe(
                styled_df,

                width="stretch",

                height=430
            )

        else:

            st.dataframe(
                display_df,

                width="stretch",

                height=430
            )

        if len(filtered_results) > rows_to_display:

            st.caption(
                f"Showing first {rows_to_display:,} records. "
                "Use the CSV download below to access all matching records."
            )

    else:

        st.warning(
            "No records match your current filters or search."
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DOWNLOAD CENTER
    # --------------------------------------------------------

    section_header(
        "💾 Export Center",
        "Download prediction results for reporting, investigation, and further analysis."
    )

    col1, col2 = st.columns(2)

    with col1:

        render_html("""
        <div class="download-card">

            <div class="download-title">
                📁 Complete Prediction Report
            </div>

            <div class="download-description">
                Export all analyzed network traffic records,
                prediction labels, probabilities, and confidence scores.
            </div>

        </div>
        """)

        st.download_button(

            label="⬇️ Download Full Results CSV",

            data=dataframe_to_csv(results),

            file_name="prediction_results.csv",

            mime="text/csv",

            width="stretch",

            key="download_full_results"
        )

    with col2:

        suspicious = results[
            results["predicted_label"] == 1
        ].copy()

        render_html(f"""
        <div class="download-card">

            <div class="download-title">
                🚨 Suspicious Traffic Report
            </div>

            <div class="download-description">
                Export only malicious network traffic records
                for security investigation.
                <br>
                <b>{len(suspicious):,} suspicious records found.</b>
            </div>

        </div>
        """)

        if not suspicious.empty:

            st.download_button(

                label="⬇️ Download Suspicious Records",

                data=dataframe_to_csv(suspicious),

                file_name="suspicious_records.csv",

                mime="text/csv",

                width="stretch",

                key="download_suspicious"
            )

        else:

            st.info(
                "No suspicious records are available."
            )

    # --------------------------------------------------------
    # ADDITIONAL STATISTICS
    # --------------------------------------------------------

    st.markdown("---")

    section_header(
        "📌 Additional Statistics",
        "Confidence analysis and further insights into the prediction results."
    )

    col1, col2, col3 = st.columns(3)

    if "confidence" in results.columns:

        confidence = pd.to_numeric(
            results["confidence"],
            errors="coerce"
        )

        valid_confidence = confidence.dropna()

        high_confidence = int(
            (valid_confidence > 0.90).sum()
        )

        low_confidence = int(
            (valid_confidence < 0.50).sum()
        )

        average_confidence = (
            valid_confidence.mean() * 100
            if not valid_confidence.empty
            else 0
        )

        with col1:

            st.metric(
                "High Confidence (>90%)",
                f"{high_confidence:,}"
            )

        with col2:

            st.metric(
                "Low Confidence (<50%)",
                f"{low_confidence:,}"
            )

        with col3:

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2f}%"
            )

    else:

        st.info(
            "Confidence statistics are unavailable because "
            "the confidence column is missing."
        )

    # --------------------------------------------------------
    # PREDICTION ERRORS
    # --------------------------------------------------------

    if (
        actual_labels_available
        and "actual_label" in results.columns
    ):

        st.markdown("---")

        section_header(
            "⚠️ Prediction Error Analysis",
            "Identify incorrectly classified network traffic records."
        )

        valid_mask = results["actual_label"].notna()

        actual_numeric = pd.to_numeric(
            results["actual_label"],
            errors="coerce"
        )

        predicted_numeric = pd.to_numeric(
            results["predicted_label"],
            errors="coerce"
        )

        valid_mask = (
            valid_mask
            & actual_numeric.isin([0, 1])
            & predicted_numeric.isin([0, 1])
        )

        errors = results[
            valid_mask
            & (
                actual_numeric != predicted_numeric
            )
        ].copy()

        correct_count = int(
            valid_mask.sum() - len(errors)
        )

        error_count = len(errors)

        total_valid = int(valid_mask.sum())

        error_rate = (
            error_count / total_valid * 100
            if total_valid > 0 else 0
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Correct Predictions",
                f"{correct_count:,}"
            )

        with col2:

            st.metric(
                "Misclassified Records",
                f"{error_count:,}"
            )

        with col3:

            st.metric(
                "Error Rate",
                f"{error_rate:.2f}%"
            )

        if error_count > 0:

            st.warning(
                f"⚠️ Found {error_count:,} incorrectly "
                "classified network traffic records."
            )

            error_columns = [
                col for col in [
                    "prediction",
                    "actual_class",
                    "actual_label",
                    "predicted_label",
                    "attack_probability",
                    "confidence"
                ]

                if col in errors.columns
            ]

            error_columns.extend([
                col for col in feature_columns[:3]
                if col in errors.columns
                and col not in error_columns
            ])

            st.dataframe(
                errors[error_columns].head(50),

                width="stretch",

                height=350
            )

            st.download_button(

                label="⬇️ Download Misclassified Records",

                data=dataframe_to_csv(errors),

                file_name="misclassified_records.csv",

                mime="text/csv",

                width="stretch",

                key="download_errors"
            )

        elif total_valid > 0:

            st.success(
                "✅ No prediction errors found in the valid labeled records!"
            )

        else:

            st.info(
                "No valid labeled records are available for error analysis."
            )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown("---")

    render_html("""
    <div style="
        text-align: center;
        padding: 18px;
        color: #64748b;
        font-size: 0.82rem;
    ">

         <b>Network Intrusion Detection System</b>

        <br><br>

        Security Analytics Dashboard
        <br>
        Monitor • Detect • Analyze • Protect

    </div>
    """)
