# 🆔 Aadhaar Risk Intelligence

An AI-powered early-warning system that predicts **service stress and digital exclusion risk** in Aadhaar enrolment centers across India.

This project analyzes enrolment, demographic, and biometric patterns to help administrators **act before citizens face service disruption**.

Built as a hackathon-ready prototype with:
- End-to-end data pipeline
- Machine Learning risk model
- Interactive Streamlit dashboard
- Explainable AI outputs

---

## 🚀 What It Does

For any **State → District → Date**, the system:

1. Aggregates Aadhaar activity signals  
2. Extracts intelligent features:
   - Enrolment load  
   - Child / Youth / Adult share  
   - Biometric pressure  
   - Log-scaled demand  
3. Uses a trained ML model to predict:
   - **Low / Medium / High** service risk  
4. Shows:
   - Risk level  
   - Key indicators  
   - AI decision factors  
   - Trends (when data exists)  
   - Human-readable insights  
   - Operational recommendations  

This turns raw government data into **actionable intelligence**.

---

## 🧠 Architecture


```
Aadhaar-Risk-Intelligence/
│
├── app.py # Streamlit dashboard (UI + ML inference)
│
├── Note_books/
│ ├── 01_exploration.ipynb # Data understanding & EDA
│ ├── 02_feature_engineering.ipynb # Cleaning, merging, feature creation
│ └── 03_model_training.ipynb # ML model training & evaluation
│
├── data/
│ ├── api_data_aadhar_biometric/ # Raw biometric datasets
│ ├── api_data_aadhar_demographic/ # Raw demographic datasets
│ ├── api_data_aadhar_enrolment/ # Raw enrolment datasets
│ ├── clean_biometric.csv # Cleaned biometric data
│ ├── clean_demographic.csv # Cleaned demographic data
│ ├── clean_enrolment.csv # Cleaned enrolment data
│ └── final_ml_dataset.csv # Final ML-ready dataset
│
├── reference/
│ ├── canonical_states.py # Standard state names
│ └── canonical_districts.py # Standard district names
│
└── model/
├── risk_model.pkl # Trained ML model
└── label_encoder.pkl # Risk label encoder
```
## 📸 Screenshots

### Dashboard Overview
![Dashboard](screenshots/dashboard_home.png)

### High Risk Example
![High Risk](screenshots/risk_high.png)

### Trends & Insights
![Trends](screenshots/trends_view.png)


### Flow

1. Raw Aadhaar datasets → `data/api_data_*`
2. Cleaning & merging → `02_feature_engineering.ipynb`
3. Feature engineering → `final_ml_dataset.csv`
4. Model training → `03_model_training.ipynb`
5. Inference & visualization → `app.py` (Streamlit)

This shows a **full production-style ML pipeline**:
`Raw Data → Processing → Features → Model → Interactive Dashboard`



---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib rapidfuzz
```
### 2. Generate data & train model:
```
Run notebooks in order:
- 01_exploration.ipynb
- 02_feature_engineering.ipynb
- 03_model_training.ipynb

This creates:
- data/final_ml_dataset.csv
- model/risk_model.pkl
- model/label_encoder.pkl
```
### 3. Launch the app:
```
streamlit run app.py 
```
### 4. Open in browser:
```
http://localhost:8501
```
---
### HACKATHON PITCH

Aadhaar Risk Intelligence is an AI-powered early-warning system for digital governance.
It predicts service stress and exclusion risk at district level using enrolment load,
population structure, and biometric pressure—helping administrators deploy resources
before citizens are affected.


### HIGHLIGHTS

- Automatic cleaning of messy government geography
- Explainable AI (shows why a risk is predicted)
- Honest trend handling (no fake charts)
- Production-style project structure
- Real-world governance use case


### AUTHOR

Prince Patel  
AI/ML & Data Science Enthusiast  
Built for hackathons, research, and real-world impact.
