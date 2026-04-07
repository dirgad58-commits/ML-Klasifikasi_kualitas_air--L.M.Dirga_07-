import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="WaterQuality AI | Sistem Klasifikasi",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #f0f8ff; }
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 10px;
        width: 100%; height: 3em; font-weight: bold;
    }
    .result-card {
        padding: 20px; border-radius: 15px; background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_data():
    try:
        return joblib.load('all_models_components.pkl')
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

model_data = load_model_data()
final_model = model_data['random_forest']

st.markdown("<h1 style='text-align: center; color: #1e3f66;'>🌊 AI Monitoring Kualitas Air</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Input Data Uji Air")
    ammonia = st.number_input("Ammonia (mg/l)", min_value=0.0, value=0.1)
    bod = st.number_input("BOD (mg/l)", min_value=0.0, value=2.0)
    do = st.number_input("DO (mg/l)", min_value=0.0, value=6.5)
    ortho = st.number_input("Orthophosphate (mg/l)", min_value=0.0, value=0.05)
    ph = st.slider("pH", 0.0, 14.0, 7.0)
    temp = st.number_input("Temp (°C)", min_value=0.0, value=27.0)
    nitrogen = st.number_input("Nitrogen (mg/l)", min_value=0.0, value=1.5)
    nitrate = st.number_input("Nitrate (mg/l)", min_value=0.0, value=0.8)

    input_data = pd.DataFrame({
        'Ammonia (mg/l)': [ammonia], 'Biochemical Oxygen Demand (mg/l)': [bod],
        'Dissolved Oxygen (mg/l)': [do], 'Orthophosphate (mg/l)': [ortho],
        'pH (ph units)': [ph], 'Temperature (cel)': [temp],
        'Nitrogen (mg/l)': [nitrogen], 'Nitrate (mg/l)': [nitrate]
    })
    predict_btn = st.button("Analisis Kualitas Air")

# --- AREA UTAMA ---
col_data, col_result = st.columns([1, 1])

with col_data:
    st.subheader("📋 Ringkasan Data Input")
    # PERBAIKAN DI SINI: Menggunakan width="stretch"
    st.dataframe(input_data.T.rename(columns={0: 'Nilai'}), width="stretch")

with col_result:
    st.subheader("🎯 Hasil Klasifikasi AI")
    if predict_btn:
        with st.spinner('Menganalisis...'):
            time.sleep(1)
            prediction = final_model.predict(input_data)
            
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            if prediction[0] == 0:
                st.success("## ✨ STATUS: BAIK")
            elif prediction[0] == 1:
                st.warning("## ⚠️ STATUS: TERCEMAR RINGAN")
            else:
                st.error("## 🚨 STATUS: TERCEMAR BERAT")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👈 Silakan klik tombol 'Analisis' di sidebar.")
