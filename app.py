import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Konfigurasi Halaman Profesional
st.set_page_config(
    page_title="Sistem Klasifikasi Kualitas Air",
    layout="centered", # Menggunakan layout centered agar fokus di tengah
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan akademis
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Times New Roman', serif;
        text-align: center;
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1E3A8A;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model_path = "models"
    with open(os.path.join(model_path, 'feature_info.json'), 'r') as f:
        info = json.load(f)
    
    scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))
    ohe = pickle.load(open(os.path.join(model_path, 'ohe.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(model_path, 'label_encoder.pkl'), 'rb'))
    
    # Load Best Model (CatBoost)
    model_file = info['best_model_name'].lower() + ".pkl"
    model = pickle.load(open(os.path.join(model_path, model_file), 'rb'))
    
    return info, scaler, ohe, le, model

try:
    info, scaler, ohe, le, model = load_assets()
except Exception as e:
    st.error(f"Sistem gagal memuat modul model: {e}")
    st.stop()

# Header Utama
st.title("Sistem Pakar Klasifikasi Kualitas Air")
st.markdown(f"""
    **Informasi Model:** {info['best_model_name']} Classifier  
    **Akurasi Pengujian:** {info['best_accuracy']:.2%}  
    ---
""")

# Bagian Form Input di Tengah Halaman
st.subheader("Parameter Input Laboratorium")
with st.form("classification_form"):
    
    # Mengatur input numerik dalam 3 kolom agar rapi
    col1, col2, col3 = st.columns(3)
    
    user_inputs = {}
    
    # Distribusi input numerik ke kolom
    for i, col_name in enumerate(info['numeric_cols']):
        current_col = [col1, col2, col3][i % 3]
        user_inputs[col_name] = current_col.number_input(
            f"{col_name.upper()}", 
            value=0.0, 
            format="%.4f",
            help=f"Masukkan nilai konsentrasi {col_name}"
        )

    # Input Kategorikal di bawahnya
    st.markdown("---")
    for col_name in info['categorical_cols']:
        options = ohe.categories_[0].tolist()
        user_inputs[col_name] = st.selectbox(
            f"Kategori {col_name.replace('_', ' ').title()}", 
            options
        )

    # Tombol Aksi
    submit_button = st.form_submit_button("Lakukan Klasifikasi")

if submit_button:
    # Pre-processing
    input_df = pd.DataFrame([user_inputs])
    
    X_num = input_df[info['numeric_cols']]
    X_cat = input_df[info['categorical_cols']]
    
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)
    
    X_final = np.hstack([X_num_scaled, X_cat_encoded])
    
    # Proses Klasifikasi
    prediction_idx = model.predict(X_final)
    
    if isinstance(prediction_idx, np.ndarray):
        prediction_idx = int(prediction_idx.flatten()[0])
        
    result_label = le.inverse_transform([prediction_idx])[0]
    
    # Tampilan Hasil Klasifikasi
    st.markdown("### Hasil Analisis Sistem")
    
    # Logika warna berdasarkan tingkat kualitas
    if result_label.lower() in ['excellent', 'good']:
        st.success(f"Klasifikasi Kualitas: **{result_label.upper()}**")
    elif result_label.lower() in ['medium']:
        st.info(f"Klasifikasi Kualitas: **{result_label.upper()}**")
    else:
        st.error(f"Klasifikasi Kualitas: **{result_label.upper()}**")
    
    st.caption("Hasil klasifikasi berdasarkan data masukan laboratorium menggunakan algoritma mesin pembelajaran.")

# Footer Akademis
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Universitas Halu Oleo - Teknik Informatika</p>", unsafe_allow_html=True)
