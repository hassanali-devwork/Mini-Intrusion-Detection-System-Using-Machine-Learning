# Network Intrusion Detection System (NIDS)

A machine learning-based Network Intrusion Detection System using the UNSW-NB15 dataset and Random Forest classifier. This project includes a professional three-page web application built with Streamlit for real-time network traffic analysis.

## 📋 Project Overview

This system uses a trained Random Forest model to detect malicious network traffic patterns. It can:
- Analyze network traffic CSV files
- Classify traffic as benign or malicious
- Provide detailed prediction results and analytics
- Display model performance metrics
- Generate downloadable reports

## 🏗️ Project Structure

```
project/
│
├── app.py                          # Main Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── pages/                          # Streamlit pages
│   ├── __init__.py
│   ├── Home.py                     # Page 1: Dashboard & Model Info
│   ├── Upload_Detection.py         # Page 2: Upload & Detection
│   └── Results_Analytics.py        # Page 3: Results & Analytics
│
├── src/                            # Original source files (unchanged)
│   ├── model.py                    # Model evaluation script
│   ├── predict.py                  # Prediction script
│   ├── preprocessing.py            # Data preprocessing script
│   └── trainmodel.py               # Model training script
│
├── models/                         # Trained model artifacts
│   ├── random_forest_ids.joblib    # Trained Random Forest model
│   └── preprocessor.joblib         # Fitted preprocessing pipeline
│
├── dataset/                        # Dataset files
│   ├── processed/                  # Preprocessed data
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
│
└── results/                        # Training results and metrics
    ├── model_metrics.json
    ├── classification_report.txt
    ├── confusion_matrix.csv
    └── feature_importance.csv
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Step 2: Verify Model Files

Ensure the following files exist in the `models/` directory:
- `random_forest_ids.joblib` - Trained model
- `preprocessor.joblib` - Preprocessing pipeline

These files should already exist if you have trained the model using `trainmodel.py`.

## 🎯 Running the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at:
```
http://localhost:8501
```

## 📱 Application Pages

### Page 1: Home / Dashboard

- Project overview and introduction
- Model performance metrics (accuracy, precision, recall, F1-score)
- Confusion matrix display
- Training and testing record counts
- Dataset information (UNSW-NB15)
- System workflow explanation
- Model configuration details

### Page 2: Upload & Detection

- Upload CSV files containing network traffic data
- Preview uploaded dataset
- Validate file format and features
- Apply preprocessing pipeline (same as training)
- Run real ML predictions using the trained model
- Display prediction summary (benign vs malicious counts)
- Store results in session state for analysis

### Page 3: Results & Analytics

- Detailed prediction statistics
- Interactive charts:
  - Normal vs Attack pie chart
  - Attack probability distribution histogram
- Evaluation metrics (if actual labels available):
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix visualization
  - Classification report
- Detailed prediction table with:
  - Search and filtering functionality
  - Filter by Normal/Attack traffic
  - Confidence scores
- Download options:
  - Full prediction results (CSV)
  - Suspicious records only (CSV)
- Additional statistics and error analysis

## 📊 Model Performance

The trained Random Forest model achieves the following performance on the test set:

- **Accuracy:** 87.11%
- **Precision:** 81.67%
- **Recall:** 98.77%
- **F1-Score:** 89.41%
- **ROC-AUC:** 0.9804
- **PR-AUC:** 0.9839

## 📁 Supported CSV Format

The uploaded CSV file should contain network traffic features compatible with the UNSW-NB15 dataset. Expected features include:

**Numerical Features:**
- dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl
- sload, dload, sloss, dloss, sinpkt, dinpkt
- sjit, djit, swin, stcpb, dtcpb, dwin
- tcprtt, synack, ackdat, smean, dmean
- trans_depth, response_body_len
- ct_srv_src, ct_state_ttl, ct_dst_ltm
- ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm
- is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd
- ct_src_ltm, ct_srv_dst, is_sm_ips_ports

**Categorical Features:**
- proto (protocol)
- service
- state

**Optional Columns:**
- label: Actual label (0 = Benign, 1 = Attack) - for evaluation
- attack_cat: Attack category - for reference
- id: Record identifier - will be ignored during prediction

## 🔧 Technical Details

### Model Configuration

- **Algorithm:** Random Forest Classifier
- **Number of Trees:** 200
- **Max Features:** sqrt
- **Class Weight:** balanced_subsample
- **Random State:** 42
- **Criterion:** Gini impurity
- **Bootstrap:** Enabled

### Preprocessing Pipeline

- Numerical features: Median imputation
- Categorical features: Most frequent imputation + One-Hot Encoding
- Missing values: Handled by imputers
- Infinite values: Replaced with NaN

## 🛡️ Security Notes

- The model is trained on the UNSW-NB15 dataset
- Do not use this system for production security without additional validation
- Regular model retraining is recommended for evolving threat landscapes
- This system is for educational and research purposes

## 📝 Original Training Scripts

The original training scripts are preserved in the `src/` directory:

- `preprocessing.py` - Data preprocessing and pipeline creation
- `trainmodel.py` - Model training and evaluation
- `predict.py` - Command-line prediction script
- `model.py` - Model evaluation script

These scripts can be run independently for model training and evaluation:

```bash
python src/preprocessing.py
python src/trainmodel.py
python src/predict.py --input dataset/UNSW_NB15_testing-set.csv
```

## 🐛 Troubleshooting

### Model not found error
Ensure the model files exist in the `models/` directory:
- `random_forest_ids.joblib`
- `preprocessor.joblib`

If missing, run the training scripts first:
```bash
python src/preprocessing.py
python src/trainmodel.py
```

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CSV upload errors
Verify that your CSV file contains the expected features. The system will handle missing features through imputation, but core network features should be present.

## 📄 License

This project uses the UNSW-NB15 dataset. Please refer to the dataset's license for usage terms.

## 👥 Acknowledgments

- UNSW-NB15 Dataset creators
- Scikit-learn library
- Streamlit framework
- Plotly for interactive visualizations

## 📧 Contact

For questions or issues, please refer to the project documentation or create an issue in the project repository.
