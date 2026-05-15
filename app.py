# app.py
import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
import os

st.set_page_config(page_title="Diabetes Diagnostics Platform", page_icon="🩺", layout="centered")

MODEL_PATH = 'models/diabetes_ann_model.onnx'
SCALER_PATH = 'models/scaler.pkl'

@st.cache_resource
def load_production_assets():
    """Thread-safe asset caching for high-speed concurrent serving."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    session = ort.InferenceSession(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return session, scaler

session, scaler = load_production_assets()

st.title(" 🩺 Deep Learning Diabetes Risk Predictor")
st.write("Input patient physiological data below to evaluate target diabetes risk via your production ANN model.")

if session is None or scaler is None:
    st.error("🚨 Critical Error: Production artifacts missing. Execute 'python src/train.py' locally first.")
else:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=1.0, max_value=300.0, value=115.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=1.0, max_value=200.0, value=72.0)
        skin_thickness = st.number_input("Triceps Skin Thickness (mm)", min_value=1.0, max_value=100.0, value=20.0)
        
    with col2:
        insulin = st.number_input("Two-Hour Serum Insulin (mu U/ml)", min_value=1.0, max_value=900.0, value=80.0)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=1.0, max_value=70.0, value=32.0)
        dpf = st.number_input("Diabetes Pedigree Function Value", min_value=0.01, max_value=3.0, value=0.35, format="%.3f")
        age = st.number_input("Patient Age", min_value=1, max_value=120, value=30, step=1)

    if st.button("Run Diagnostic Evaluation", type="primary"):
        # Compile inputs to raw matrix array shape (1, 8)
        raw_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], dtype=np.float32)
        
        # Scale inputs using our standardized profile
        scaled_features = scaler.transform(raw_features).astype(np.float32)
        
        # Extract precise operational layer parameters matching your log tracking
        input_name = session.get_inputs()[0].name   # 'input_layer'
        output_name = session.get_outputs()[0].name # 'output_0'
        
        # Compute forward pass via ONNX Inference Node
        prediction_raw = session.run([output_name], {input_name: scaled_features})
        prediction_probability = prediction_raw[0][0][0]
        prediction_class = 1 if prediction_probability > 0.5 else 0
        
        st.markdown("---")
        st.subheader("Diagnostic Risk Analysis")
        
        if prediction_class == 1:
            st.error(f"⚠️ **High Risk Alert**: The patient is classified as **Diabetic**. Confidence Level: {prediction_probability * 100:.2f}%")
        else:
            st.success(f"✅ **Normal Assessment**: The patient is classified as **Non-Diabetic**. Confidence Level: {(1 - prediction_probability) * 100:.2f}%")
            
        st.info("💡 *Notice: This is an AI assessment platform. Outputs should be cross-referenced with empirical clinical diagnostics.*")