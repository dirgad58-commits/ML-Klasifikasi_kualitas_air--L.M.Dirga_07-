import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Water Quality Intelligence", 
    page_icon="🌊", 
    layout="wide"
)

# CSS untuk mempercantik antarmuka
st.markdown("""
<style>
    .main-title { font-size: 40px; font-weight: 800; color: #0F172A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #475569; text-align: center; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 Dashboard Klasifikasi Kualitas Air</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Cerdas Berbasis Ensemble Learning (Hard Voting)</div>', unsafe_allow_html=True)

# 2. Load Model (Hanya merujuk pada all_models_components.pkl)
@st.cache_resource
def load_models():
    # Mengambil langsung dari root directory
    return joblib.load('all_models_components.pkl')

try:
    components = load_models()
    rf_pipe = components['random_forest']
    xgb_pipe = components['xgboost']
    gb_pipe = components['gradient_boosting']
    le = components['label_encoder']
    
    st.sidebar.success("🟢 AI System Online!")
    st.sidebar.write(f"**Model Terbaik:** {components.get('best_model_name', 'Gradient Boosting')}")
except Exception as e:
    st.sidebar.error(f"🔴 Error: File 'all_models_components.pkl' tidak ditemukan di root directory.")
    st.stop()

# --- INPUT DATA ---
st.subheader("📥 Input Parameter Sampel Air")
col_kiri, col_kanan = st.columns(2, gap="large")

with col_kiri:
    ammonia = st.number_input("Ammonia (mg/l)", value=0.05)
    bod = st.number_input("BOD (mg/l)", value=1.3)
    do = st.number_input("Dissolved Oxygen (mg/l)", value=8.15)
    ortho = st.number_input("Orthophosphate (mg/l)", value=0.01)

with col_kanan:
    ph = st.number_input("pH Level", value=7.0)
    temp = st.number_input("Temperature (°C)", value=10.0)
    nitrogen = st.number_input("Total Nitrogen (mg/l)", value=0.34)
    nitrate = st.number_input("Nitrate (mg/l)", value=1.17)

# Mapping input sesuai nama kolom saat training
input_data = pd.DataFrame([{
    'Ammonia (mg/l)': ammonia,
    'Biochemical Oxygen Demand (mg/l)': bod,
    'Dissolved Oxygen (mg/l)': do,
    'Orthophosphate (mg/l)': ortho,
    'pH (ph units)': ph,
    'Temperature (cel)': temp,
    'Nitrogen (mg/l)': nitrogen,
    'Nitrate (mg/l)': nitrate
}])

st.write("")
if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
    # Prediksi dari 3 model
    p_rf = rf_pipe.predict(input_data)
    p_xgb = xgb_pipe.predict(input_data)
    p_gb = gb_pipe.predict(input_data)
    
    # Konversi ke label teks
    labels = le.inverse_transform([p_rf[0], p_xgb[0], p_gb[0]])
    
    # Voting Terbanyak
    final_vote = Counter(labels).most_common(1)[0][0]
    
    # --- DISPLAY HASIL ---
    st.markdown("---")
    theme = {
        "Excellent": "#059669", "Good": "#2563EB", "Fair": "#D97706", 
        "Marginal": "#DC2626", "Poor": "#7F1D1D"
    }
    color = theme.get(final_vote, "#475569")
    
    st.markdown(f"""
    <div style="background-color:{color}; padding:30px; border-radius:15px; text-align:center; color:white;">
        <h2 style="margin:0;">HASIL KLASIFIKASI: {final_vote.upper()}</h2>
        <p style="opacity:0.9;">Keputusan diambil berdasarkan konsensus 3 algoritma AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detail Voting
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest", labels[0])
    c2.metric("XGBoost", labels[1])
    c3.metric("Gradient Boosting", labels[2])
