"""
Network Intrusion Detection System - Streamlit Web Application
Dataset: UNSW-NB15
Model: Random Forest

Main application entry point.
Run: streamlit run app.py
"""

import streamlit as st
from textwrap import dedent

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CYBERSECURITY PREMIUM THEME
# ============================================================
st.markdown(dedent("""
    <style>
    /* Hide the default sidebar and its toggle button */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Global Dark Glassmorphism Background */
    .stApp {
        background-color: #080d1a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(30, 27, 75, 0.4) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(8, 145, 178, 0.15) 0px, transparent 50%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Adjust top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Premium Gradient Headings */
    h1, h2, h3 {
        font-weight: 700;
        margin-bottom: 1rem;
    }
    h1 {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #ffffff, #a5b4fc, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #818cf8;
    }

    /* Sleek Button Styling (Applies to Nav and standard buttons) */
    .stButton > button {
        background: rgba(30, 41, 59, 0.6);
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(129, 140, 248, 0.6);
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.2);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Helper Classes for text */
    .success-text {
        color: #10b981;
        font-weight: 600;
    }
    .danger-text {
        color: #ef4444;
        font-weight: 600;
    }
    .highlight-text {
        color: #38bdf8;
        font-weight: 600;
    }

    /* Custom Divider */
    hr {
        border-color: rgba(148, 163, 184, 0.15);
        margin: 1.5rem 0;
    }
    </style>
"""), unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'actual_labels_available' not in st.session_state:
    st.session_state.actual_labels_available = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"


# ============================================================
# MODERN NAVIGATION BAR
# ============================================================
st.markdown(dedent("""
    <div style='display: flex; align-items: center; margin-bottom: 5px;'>
        <span style='font-size: 1.2rem; font-weight: 700; color: #818cf8; margin-right: 20px;'>
            🛡️ NIDS Platform
        </span>
    </div>
"""), unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        st.session_state.current_page = "Home"
        st.rerun()

with col2:
    if st.button("📤 Upload & Detect", key="nav_upload", use_container_width=True):
        st.session_state.current_page = "Upload"
        st.rerun()

with col3:
    if st.button("📊 Analytics & Results", key="nav_results", use_container_width=True):
        st.session_state.current_page = "Results"
        st.rerun()

with col4:
    if st.button("ℹ️ About System", key="nav_about", use_container_width=True):
        st.session_state.current_page = "About"
        st.rerun()

st.markdown("---")


# ============================================================
# PAGE ROUTING
# ============================================================
if st.session_state.current_page == "Home":
    from pages import Home
    Home.show()

elif st.session_state.current_page == "Upload":
    from pages import Upload_Detection
    Upload_Detection.show()

elif st.session_state.current_page == "Results":
    from pages import Results_Analytics
    Results_Analytics.show()

elif st.session_state.current_page == "About":
    
    st.title("ℹ️ About This System")
    
    st.markdown(dedent("""
        <div style="font-size: 1.1rem; color: #94a3b8; margin-bottom: 2rem;">
            This system is designed for enterprise-grade network threat detection, 
            leveraging advanced Machine Learning to identify anomalies and malicious patterns.
        </div>
    """), unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("🧠 **Model Architecture**\n\n"
                "The core engine utilizes a **Random Forest classifier** comprising 200 independent decision trees. "
                "This ensemble method ensures high precision, resistance to overfitting, and rapid processing times "
                "suitable for large network datasets.")
                
        st.success("📊 **Performance Metrics**\n\n"
                   "- **Accuracy:** 87.11%\n"
                   "- **Precision:** 81.67%\n"
                   "- **Recall:** 98.77%\n"
                   "- **F1-Score:** 89.41%")

    with col_info2:
        st.warning("🗂️ **Dataset Information**\n\n"
                   "Trained on the comprehensive **UNSW-NB15 dataset**. This dataset encompasses a hybrid of real modern "
                   "normal activities and synthetic contemporary attack behaviors, providing a robust baseline for modern threat detection.")
        
        st.markdown(dedent("""
            ### 🔄 How to Use
            1. Navigate to the **Upload & Detect** tab.
            2. Securely upload your network traffic CSV file.
            3. The AI engine will preprocess and classify the traffic in real-time.
            4. Review comprehensive visualizations and download reports in the **Analytics & Results** tab.
        """))