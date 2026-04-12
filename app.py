import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Water Quality Classifier", layout="wide")

# Fungsi untuk memuat aset dari dalam folder 'models'
@st.cache_resource
def load_assets():
    # Tentukan jalur folder
    model_path = "models"
    
    # Load Metadata JSON
    with open(os.path.join(model_path, 'feature_info.json'), 'r') as f:
        info = json.load(f)
    
    # Load Pre-processing tools
    scaler = pickle.load(open(os.path.join(model_path, 'scaler.pkl'), 'rb'))
    ohe = pickle.load(open(os.path.join(model_path, 'ohe.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(model_path, 'label_encoder.pkl'), 'rb'))
    
    # Memuat model terbaik (berdasarkan data di feature_info.json)
    # Secara dinamis mengambil file catboost.pkl (atau sesuai best_model_name)
    model_file = info['best_model_name'].lower() + ".pkl"
    model = pickle.load(open(os.path.join(model_path, model_file), 'rb'))
    
    return info, scaler, ohe, le, model

# Menjalankan fungsi load
try:
    info, scaler, ohe, le, model = load_assets()
except FileNotFoundError as e:
    st.error(f"Gagal memuat file: {e}. Pastikan folder 'models' sudah ada di GitHub.")
    st.stop()

st.title("💧 Water Quality Classification AI")
st.info(f"Menggunakan Model: **{info['best_model_name']}** | Akurasi: {info['best_accuracy']:.2%}")

# --- Bagian Sidebar untuk Input ---
st.sidebar.header("Input Parameter")

def get_user_input():
    inputs = {}
    
    # Input Numerik berdasarkan data di feature_info.json
    st.sidebar.subheader("Fitur Numerik")
    for col in info['numeric_cols']:
        inputs[col] = st.sidebar.number_input(f"Nilai {col.upper()}", value=0.0, format="%.4f")
        
    # Input Kategorikal (macro_land_use)
    st.sidebar.subheader("Fitur Kategorikal")
    for col in info['categorical_cols']:
        # Ambil kategori asli dari OneHotEncoder
        options = ohe.categories_[0].tolist()
        inputs[col] = st.sidebar.selectbox(f"Pilih {col}", options)
        
    return pd.DataFrame([inputs])

input_df = get_user_input()

# --- Tampilan Utama ---
st.subheader("Data yang Akan Diprediksi")
st.dataframe(input_df)

if st.button("Analisis Kualitas Air", type="primary"):
    # 1. Pre-processing
    X_num = input_df[info['numeric_cols']]
    X_cat = input_df[info['categorical_cols']]
    
    # Transformasi menggunakan scaler dan ohe yang dimuat dari folder models
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)
    
    # Gabungkan kembali fitur
    X_final = np.hstack([X_num_scaled, X_cat_encoded])
    
    # 2. Prediksi
    prediction_idx = model.predict(X_final)
    
    # Pastikan format indeks benar (untuk CatBoost terkadang array 2D)
    if isinstance(prediction_idx, np.ndarray):
        prediction_idx = int(prediction_idx.flatten()[0])
        
    # 3. Decode Label (Mengubah angka kembali ke 'good', 'bad', dll)
    result_label = le.inverse_transform([prediction_idx])[0]
    
    # Tampilkan Hasil dengan gaya visual
    st.markdown("---")
    if result_label.lower() in ['excelent', 'good']:
        st.success(f"### Hasil Prediksi: **{result_label.upper()}**")
        st.balloons()
    else:
        st.warning(f"### Hasil Prediksi: **{result_label.upper()}**")
