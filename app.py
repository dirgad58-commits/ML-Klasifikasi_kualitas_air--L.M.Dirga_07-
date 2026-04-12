import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Konfigurasi Halaman - Wide Mode untuk perbandingan kolom
st.set_page_config(
    page_title="Dashboard Analisis Klasifikasi Kualitas Air",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS Akademis dan Profesional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .report-title {
        color: #1E3A8A;
        font-family: 'Serif';
        font-weight: bold;
        text-align: center;
        border-bottom: 3px solid #1E3A8A;
        padding-bottom: 20px;
        margin-bottom: 25px;
    }
    .model-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stNumberInput, .stSelectbox {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all_assets():
    path = "models"
    # Metadata
    with open(os.path.join(path, 'feature_info.json'), 'r') as f:
        info = json.load(f)
    
    # Pre-processing
    scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))
    ohe = pickle.load(open(os.path.join(path, 'ohe.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(path, 'label_encoder.pkl'), 'rb'))
    
    # Models
    models = {
        "CatBoost": pickle.load(open(os.path.join(path, 'catboost.pkl'), 'rb')),
        "LightGBM": pickle.load(open(os.path.join(path, 'lightgbm.pkl'), 'rb')),
        "HistGradientBoosting": pickle.load(open(os.path.join(path, 'histgradientboosting.pkl'), 'rb'))
    }
    
    return info, scaler, ohe, le, models

try:
    info, scaler, ohe, le, models = load_all_assets()
except Exception as e:
    st.error(f"Kesalahan Fatal: Gagal memuat komponen model. Detail: {e}")
    st.stop()

# Header
st.markdown("<h1 class='report-title'>Sistem Klasifikasi Kualitas Air Berbasis Pembelajaran Mesin</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analisis Perbandingan Algoritma Gradient Boosting untuk Penentuan Status Mutu Air</p>", unsafe_allow_html=True)

# Container Input
st.markdown("### 1. Parameter Masukan (Data Laboratorium)")
st.info("Nilai default di bawah ini diset untuk pengujian cepat (Normal/Good condition).")

with st.form("input_form"):
    # Penentuan nilai default untuk uji cepat
    default_vals = {
        'ph': 7.20, 'do': 6.50, 'bod': 2.10, 'tc': 50.0, 
        'tn': 1.20, 'tp': 0.05, 'ts': 150.0, 'turb': 4.50, 'temp': 25.0
    }

    col1, col2, col3 = st.columns(3)
    user_inputs = {}

    # Distribusi input numerik
    for i, col_name in enumerate(info['numeric_cols']):
        target_col = [col1, col2, col3][i % 3]
        user_inputs[col_name] = target_col.number_input(
            f"{col_name.upper()}", 
            value=default_vals.get(col_name, 0.0), 
            format="%.2f"
        )

    st.markdown("---")
    # Input Kategorikal
    land_use_options = ohe.categories_[0].tolist()
    user_inputs['macro_land_use'] = st.selectbox(
        "Klasifikasi Penggunaan Lahan (Macro Land Use)", 
        land_use_options,
        index=0
    )

    submit = st.form_submit_button("JALANKAN PROSES KLASIFIKASI")

if submit:
    # --- Pre-processing Data ---
    input_df = pd.DataFrame([user_inputs])
    X_num = input_df[info['numeric_cols']]
    X_cat = input_df[['macro_land_use']]
    
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)
    X_final = np.hstack([X_num_scaled, X_cat_encoded])

    # --- Bagian Hasil Analisis ---
    st.markdown("### 2. Laporan Hasil Klasifikasi Komparatif")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    cols = [res_col1, res_col2, res_col3]

    for idx, (name, model) in enumerate(models.items()):
        # Klasifikasi
        raw_pred = model.predict(X_final)
        
        # Penanganan perbedaan output format antar library
        if isinstance(raw_pred, np.ndarray):
            pred_idx = int(raw_pred.flatten()[0]) if raw_pred.ndim > 1 else int(raw_pred[0])
        else:
            pred_idx = int(raw_pred)
            
        label = le.inverse_transform([pred_idx])[0].upper()
        
        # Visualisasi per kolom
        with cols[idx]:
            st.markdown(f"""
                <div class='model-card'>
                    <small>Algoritma</small>
                    <h4>{name}</h4>
                    <hr>
                    <p>Status Klasifikasi:</p>
                    <h2 style='color: {"#16a34a" if label in ["EXCELENT", "GOOD"] else "#dc2626"};'>{label}</h2>
                </div>
            """, unsafe_allow_html=True)

    # --- Detail Teknis ---
    st.markdown("---")
    with st.expander("Lihat Metadata Vektor Input"):
        st.write("Data yang telah dinormalisasi dan diencode untuk mesin:")
        st.code(str(X_final))

# Footer
st.markdown("<br><br><p style='text-align: center; color: #6b7280; font-size: 12px;'>Laboratorium Komputasi - Teknik Informatika - Universitas Halu Oleo</p>", unsafe_allow_html=True)
