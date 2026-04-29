🩺 Diabetes Clinical Decision Support System (MLOps + AI Product)

An end-to-end clinical decision support system for diabetes risk prediction, built with a production-ready machine learning pipeline, real-time monitoring, calibration, and mobile integration.

🚀 Project Overview

This project is not just a machine learning model — it is a complete AI-powered clinical decision support system designed for real-world healthcare scenarios.

It combines:

Machine Learning (Ensemble models)
Clinical decision logic (risk scoring + threshold optimization)
Model calibration (probability reliability)
MLOps monitoring (drift detection + logging)
API backend (FastAPI-style architecture)
Flutter mobile integration
🧠 Key Features
🔬 Machine Learning Pipeline
Voting Ensemble (Logistic Regression + Random Forest + XGBoost)
SMOTE for class imbalance handling
Feature engineering:
Glucose × Age interaction
BMI categorization (WHO-based)
Winsorization for outlier control
Standard scaling pipeline
🎯 Clinical Optimization
PR-curve based threshold selection
Recall-constrained optimization (Recall ≥ 0.80)
Class-weighted learning for imbalance handling
Feature selection (e.g., SkinThickness removal improved performance)
📊 Model Performance
Model	ROC-AUC	Recall	Precision	F1
Voting (Clinical Optimized)	0.80	0.82	0.53	0.65
False Negatives significantly reduced (critical for healthcare use case)
Optimized for clinical safety (high recall priority)
⚖️ Model Calibration
Isotonic calibration (CalibratedClassifierCV)
Improved probability reliability
Brier score reduction (better uncertainty estimation)
📡 MLOps & Monitoring Layer

Production-grade monitoring system including:

🔍 Drift Detection:
ROC-AUC drift monitoring
Brier score drift tracking
Prediction distribution shift analysis
🚨 Alert System:
OK / WARNING / CRITICAL classification
Automatic logging in JSONL format
📊 Risk Distribution Monitoring:
Low / Medium / High risk population tracking
🔁 Single Inference Pipeline

All inference flows through a unified pipeline:

Preprocessing
Model prediction
Calibration
Threshold decision
Risk scoring
Monitoring & logging
⚙️ API Layer
/predict → real-time prediction + risk score
/model-health → system health metrics
/threshold-config → clinician-controlled threshold override

Supports:

Backward compatibility
Clinical override (0.35 – 0.50 range)
Real-time drift status reporting
📱 Flutter Integration

Mobile-ready outputs:

Risk Score (0.0 – 1.0)
Risk Categories:
Low Risk
Medium Risk
High Risk
Model Health Dashboard:
ROC-AUC
Recall
Threshold
Drift status
🏗️ Architecture
Data → Preprocessing → Feature Engineering
      → ML Ensemble Model
      → Calibration Layer
      → Threshold Decision Engine
      → Risk Scoring System
      → Monitoring Layer
      → API → Flutter UI
📦 Tech Stack

Machine Learning

Python
Scikit-learn
XGBoost
Imbalanced-learn (SMOTE)
Optuna

MLOps

Custom drift monitoring
JSONL logging system
Calibration (Isotonic)

Backend

Python API layer
Modular inference pipeline

Frontend

Flutter (mobile UI integration)
📈 Clinical Impact
Reduced False Negatives → improved patient safety
Optimized for high recall (medical priority)
Transparent risk scoring system
Clinician-controlled threshold adjustment
Interpretable AI outputs
🔐 Key Design Principles
No black-box predictions → calibrated probabilities
Safety-first optimization (recall > precision priority)
Production-ready modular architecture
Fully extensible monitoring system
📊 Example Output
{
  "prediction": 1,
  "risk_score": 0.82,
  "risk_category": "High Risk",
  "model_info": {
    "roc_auc": 0.80,
    "recall": 0.82,
    "threshold": 0.42
  }
}
🧩 Future Improvements
Dockerization + CI/CD pipeline
MLflow model registry integration
Real-time streaming inference
Advanced SHAP explainability dashboard
Hospital EMR integration
👨‍💻 Author Notes

This project demonstrates:

End-to-end ML system design
Clinical decision-making optimization
Production-grade MLOps architecture
Mobile + backend integration
Real-world constraint-aware AI design
⭐ Summary

A production-ready, clinically optimized AI system for diabetes risk prediction with full MLOps lifecycle support and mobile integration.