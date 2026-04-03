# ==========================================
# FRONTEND UI - PROFESSIONAL HEALTHCARE DASHBOARD
# ==========================================
import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import time

import os
import gdown

# create Models folder
os.makedirs("Models", exist_ok=True)

model_path = "Models/pneumonia_cnn_model.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1UBzpEFB4rJ5X3nu8enYOkd0gLwNEw1SZ"
    gdown.download(url, model_path, quiet=False)
    
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Clinical Assistant", page_icon="⚕️", layout="wide")

# --- 2. BULLETPROOF CSS OVERRIDES ---
st.markdown("""
    <style>
    /* ==========================================================
       1. UNIFIED APP BACKGROUND & TOP HEADER FIX 
       ========================================================== */
    
    /* Make the whole app background gray */
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #E2E8F0 !important; 
    }
    
    /* Fix the ugly white bar at the top (Streamlit's default header) */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        box-shadow: none !important;
    }

    /* Remove the massive blank space at the very top of the page */
    .block-container {
        padding-top: 3rem !important; 
        padding-bottom: 2rem !important;
    }
    
    /* ==========================================================
       2. FORCE SIDEBAR TO BE NAVY BLUE (Fixes bleed-through)
       ========================================================== */
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div:first-child {
        background-color: #0F172A !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.2) !important;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] h1 {
        color: #F8FAFC !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* ==========================================================
       3. FLOATING 3D CARDS (The main containers)
       ========================================================== */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #F1F5F9 !important;
        padding: 15px !important;
        transition: transform 0.2s ease-in-out;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
    }

    /* ==========================================================
       4. 3D "PUNCHED-IN" INPUT BOXES (Unified +/- buttons)
       ========================================================== */
    div[data-baseweb="select"],
    [data-testid="stNumberInput"] label + div {
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        background-color: #F8FAFC !important;
        box-shadow: inset 0 3px 6px rgba(0,0,0,0.06) !important;
        overflow: hidden !important; 
        transition: all 0.2s;
    }
    
    /* Strip the broken inner borders from Streamlit's text field */
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }

    /* Glow effect when user clicks/hovers an input */
    [data-testid="stNumberInput"] label + div:hover, 
    div[data-baseweb="select"]:hover,
    div[data-baseweb="select"]:focus-within,
    [data-testid="stNumberInput"] label + div:focus-within {
        border-color: #0EA5E9 !important;
        background-color: #FFFFFF !important;
        box-shadow: inset 0 2px 4px rgba(14, 165, 233, 0.1), 0 0 0 3px rgba(14, 165, 233, 0.2) !important;
    }

    /* ==========================================================
       5. TACTILE 3D PRESSABLE BUTTONS
       ========================================================== */
    div.stButton > button {
        background: linear-gradient(180deg, #0284C7 0%, #0369A1 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 6px 0 #075985, 0 10px 10px rgba(0,0,0,0.2) !important;
        padding: 16px 24px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.1s ease-in-out !important;
        width: 100% !important;
        margin-bottom: 6px; 
    }
    
    div.stButton > button:active {
        transform: translateY(6px) !important;
        box-shadow: 0 0px 0 #075985, 0 2px 2px rgba(0,0,0,0.2) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(180deg, #0369A1 0%, #075985 100%) !important;
    }

    /* ==========================================================
       6. TYPOGRAPHY & ALERTS
       ========================================================== */
    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        border-bottom: 2px solid #CBD5E1;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h2, h3, h4 {
        color: #0284C7 !important;
        font-weight: 700 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("Clinical AI System")
    st.markdown("---")
    app_mode = st.radio("🏥 Select Diagnostic Tool:", ["Home Dashboard", "🫀 Heart Risk Assessment", "🫁 X-Ray Analysis (Pneumonia)"])
    st.markdown("---")
    st.markdown("<p style='font-size: 12px; color: #94A3B8; font-style: italic; background: #1E293B; padding: 10px; border-radius: 5px;'>⚠️ <b>Disclaimer:</b> This AI is for research and assistive purposes only. It is not a substitute for professional medical diagnosis.</p>", unsafe_allow_html=True)

# --- 4. LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('Models/heart_disease_model.pkl')
        scaler = joblib.load('Models/scaler.pkl')
        cnn_model = tf.keras.models.load_model('Models/pneumonia_cnn_model.h5')
        return rf_model, scaler, cnn_model
    except Exception as e:
        return None, None, None

rf_model, scaler, cnn_model = load_models()

# ==========================================
# 5. APP MODULES
# ==========================================

# --- HOME DASHBOARD ---
if app_mode == "Home Dashboard":
    st.title("Welcome to the Clinical AI Assistant 🏥")
    st.markdown("### Empowering doctors with rapid, AI-driven diagnostics.")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.info("#### 🫀 Cardiovascular Risk Assessment\nAnalyzes 13 distinct clinical markers (vitals, lab results, patient history) using a Random Forest algorithm to predict the likelihood of severe coronary artery disease.")
    with col2:
        with st.container(border=True):
            st.success("#### 🫁 Pulmonary Radiography Analysis\nUtilizes a Deep Convolutional Neural Network (CNN) to scan patient Chest X-Rays for lung opacities indicative of bacterial or viral pneumonia.")
    
    st.markdown("<br><center><img src='https://images.unsplash.com/photo-1576091160550-2173ff9e5ee5?q=80&w=1000' width='80%' style='border-radius:15px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'></center>", unsafe_allow_html=True)

# --- HEART DISEASE ASSESSOR ---
elif app_mode == "🫀 Heart Risk Assessment":
    st.title("Cardiovascular Risk Assessment")
    st.markdown("Enter the patient's clinical profile below to generate an AI risk analysis.")

    with st.container(border=True):
        st.subheader("👤 Patient Demographics & Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (Years)", 1, 120, 50)
        with col2:
            sex = st.selectbox("Biological Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        with col3:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 200, 120)

    with st.container(border=True):
        st.subheader("🩸 Lab Results & Symptoms")
        col4, col5, col6 = st.columns(3)
        with col4:
            chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes (Diabetic)" if x == 1 else "No")
        with col5:
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col6:
            cp_dict = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: cp_dict[x])

    with st.container(border=True):
        st.subheader("📈 ECG & Advanced Cardiology")
        col7, col8, col9 = st.columns(3)
        with col7:
            restecg_dict = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
            restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: restecg_dict[x])
        with col8:
            slope_dict = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
            slope = st.selectbox("ST Segment Slope", [0, 1, 2], format_func=lambda x: slope_dict[x])
        with col9:
            thal_dict = {0: "Unknown", 1: "Normal", 2: "Fixed Defect", 3: "Reversable Defect"}
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: thal_dict[x])

        col10, col11 = st.columns(2)
        with col10:
            oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
        with col11:
            ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3, 4])

    st.markdown("<br>", unsafe_allow_html=True)
    
    _, btn_col, _ = st.columns([1, 2, 1]) 
    
    if btn_col.button("Generate AI Risk Assessment", type="primary"):
        if rf_model is not None and scaler is not None:
            with st.spinner('Analyzing clinical data...'):
                time.sleep(1) 
                
                base_features =[age, sex, trestbps, chol, fbs, thalach, exang, oldpeak, ca]
                cp_dummies =[1 if cp == 1 else 0, 1 if cp == 2 else 0, 1 if cp == 3 else 0]
                restecg_dummies =[1 if restecg == 1 else 0, 1 if restecg == 2 else 0]
                slope_dummies =[1 if slope == 1 else 0, 1 if slope == 2 else 0]
                thal_dummies =[1 if thal == 1 else 0, 1 if thal == 2 else 0, 1 if thal == 3 else 0]
                
                final_features = base_features + cp_dummies + restecg_dummies + slope_dummies + thal_dummies
                input_scaled = scaler.transform(np.array([final_features]))
                
                prediction = rf_model.predict(input_scaled)[0]
                probability = rf_model.predict_proba(input_scaled)[0][1] * 100 
                
                st.divider()
                
                with st.container(border=True):
                    st.subheader("📑 Diagnostic Report")
                    if prediction == 1:
                        st.error(f"⚠️ HIGH RISK DETECTED: {probability:.1f}% Probability of Cardiovascular Disease.")
                        st.progress(int(probability))
                        st.markdown("**Recommendation:** Immediate consultation with a cardiologist is advised. Consider further imaging or angiography.")
                    else:
                        st.success(f"✅ LOW RISK: Only {(probability):.1f}% Probability of Cardiovascular Disease.")
                        st.progress(int(probability))
                        st.markdown("**Recommendation:** Patient vitals look stable. Continue routine checkups and healthy lifestyle choices.")
        else:
            st.error("Model files not found. Cannot generate assessment.")

# --- PNEUMONIA DETECTOR ---
elif app_mode == "🫁 X-Ray Analysis (Pneumonia)":
    st.title("Automated Radiography Analysis")
    st.markdown("Upload a standard AP/PA Chest X-Ray. The AI will scan for pulmonary opacities.")

    with st.container(border=True):
        uploaded_file = st.file_uploader("📂 Drop X-Ray Image Here (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("❌ File too large! Please upload an image under 50MB.")
        else:
            col1, col2 = st.columns([1, 1.5], gap="large") 
            
            with col1:
                with st.container(border=True):
                    st.markdown("#### 📷 Source Image")
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, use_container_width=True)

            with col2:
                with st.container(border=True):
                    st.markdown("#### ⚙️ AI Analysis")
                    if st.button("Run Diagnostic Scan 🔍", type="primary"):
                        if cnn_model is not None:
                            with st.spinner("Scanning for lung anomalies..."):
                                time.sleep(1.5)
                                
                                img_resized = image.resize((150, 150)) 
                                img_array = img_to_array(img_resized)
                                img_array = np.expand_dims(img_array, axis=0) / 255.0 

                                confidence = cnn_model.predict(img_array)[0][0] * 100

                                st.divider()
                                if confidence > 50:
                                    st.error(f"🚨 **PNEUMONIA DETECTED**")
                                    st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
                                    st.progress(int(confidence))
                                    st.markdown("📌 **Radiology Notes:** High probability of fluid buildup or viral/bacterial opacities. Antibiotic intervention or further respiratory panel recommended.")
                                else:
                                    st.success(f"✅ **LUNGS CLEAR**")
                                    st.metric(label="Normalcy Confidence", value=f"{(100 - confidence):.2f}%")
                                    st.progress(int(100 - confidence))
                                    st.markdown("📌 **Radiology Notes:** No significant signs of opacification. Lungs appear visually normal.")
                        else:
                            st.error("CNN Model not loaded. Cannot run scan.")
