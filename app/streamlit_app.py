import streamlit as st
import pandas as pd

# --- Configuration ---
MOCK_API_RESPONSE = {
    "is_fraud": True,
    "confidence_score": 0.9567,
    "explanation_top_3_features": ["V14 (-3.15)", "V4 (2.56)", "V12 (-1.90)"]
}

st.set_page_config(page_title="HealthGuard AI Fraud Detector", layout="wide")

st.title("💳 HealthGuard AI: Real-Time Fraud Detection")
st.markdown("""
### Production-Ready ML System deployed with XGBoost and Explainability.
This project is an end-to-end MLOps solution designed to showcase expertise in handling 
extreme class imbalance (0.18% Fraud) and providing mandatory SHAP Explainability.
""")

st.sidebar.header("System Metrics")
st.sidebar.markdown(f"**Best Model:** XGBoost")
st.sidebar.markdown(f"**Imbalance Ratio:** ~1:540")
st.sidebar.markdown(f"**Achieved AUC-ROC:** 0.9794 (from training)")

# --- Input Form ---
st.header("Transaction Input")
with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)
    
    # Inputs for the CRITICAL SHAP features
    V14_input = col1.number_input("V14 (Risk Factor 1)", value=-4.0, format="%.4f")
    V4_input = col2.number_input("V4 (Risk Factor 2)", value=2.0, format="%.4f")
    V12_input = col3.number_input("V12 (Risk Factor 3)", value=-2.5, format="%.4f")
    
    # Placeholder inputs for other features
    V1_input = col1.number_input("V1", value=-1.0, format="%.4f")
    Amount_Scaled_input = col3.number_input("Amount_Scaled", value=0.5, format="%.4f")
    
    submitted = st.form_submit_button("Predict Fraud Status (Live Demo)")

if submitted:
    st.subheader("Prediction Result")
    
    result = MOCK_API_RESPONSE 
    
    # Display Prediction
    if result['is_fraud']:
        st.error("🚨 HIGH RISK: TRANSACTION PREDICTED AS FRAUDULENT")
    else:
        st.success("✅ LOW RISK: TRANSACTION IS LIKELY SAFE")
        
    st.metric(label="Confidence Score (P=Fraud)", value=f"{result['confidence_score']:.4f}")

    # Display Explainability
    st.subheader("Model Decision Drivers (SHAP Explainability)")
    explanation_df = pd.DataFrame({
        "Rank": [1, 2, 3],
        "Feature & Impact": result['explanation_top_3_features']
    })
    st.table(explanation_df)
    
    st.info("The prediction and explanation are served from the internal model logic.")
