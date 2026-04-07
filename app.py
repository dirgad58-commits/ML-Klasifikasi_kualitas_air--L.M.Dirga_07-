import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sklearn

# --- FIX COMPATIBILITY (Mencegah error _fill_dtype) ---
# Ini akan menambahkan atribut yang hilang secara manual jika tidak ditemukan
from sklearn.impute import SimpleImputer
if not hasattr(SimpleImputer, '_fill_dtype'):
    SimpleImputer._fill_dtype = lambda self, X: X.dtype

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Water Quality AI", layout="centered")

# --- LOAD MODEL DENGAN PENANGANAN ERROR ---
@st.cache_resource
def load_all_components():
    try:
        components = joblib.load('all_models_components.pkl')
        return components
    except Exception as e:
        st.error(f"Gagal memuat file pkl: {e}")
        return None

data = load_all_components()

if data:
    st.title("🌊 Klasifikasi Kualitas Air")
    st.write("Aplikasi ini sudah diperbaiki untuk masalah kompatibilitas library.")

    # Ambil komponen
    # Berdasarkan file Anda, model ada dalam dictionary
    model = data['random_forest'] 
    le = data['label_encoder']

    # --- INPUT DATA (Nilai Default untuk kategori GOOD) ---
    st.subheader("📝 Input Parameter Laboratorium")
    col1, col2 = st.columns(2)
    
    with col1:
        ammonia = st.number_input("Ammonia (mg/l)", value=0.01)
        bod = st.number_input("BOD (mg/l)", value=1.0)
        do = st.number_input("Dissolved Oxygen (mg/l)", value=8.5)
        ortho = st.number_input("Orthophosphate (mg/l)", value=0.02)
        
    with col2:
        ph = st.number_input("pH Air", value=7.2)
        temp = st.number_input("Temperature (°C)", value=25.0)
        nitrogen = st.number_input("Nitrogen (mg/l)", value=0.5)
        nitrate = st.number_input("Nitrate (mg/l)", value=0.2)

    # DataFrame Input (Nama kolom harus persis sesuai training)
    input_df = pd.DataFrame({
        'Ammonia (mg/l)': [ammonia],
        'Biochemical Oxygen Demand (mg/l)': [bod],
        'Dissolved Oxygen (mg/l)': [do],
        'Orthophosphate (mg/l)': [ortho],
        'pH (ph units)': [ph],
        'Temperature (cel)': [temp],
        'Nitrogen (mg/l)': [nitrogen],
        'Nitrate (mg/l)': [nitrate]
    })

    st.divider()

    # --- PROSES KLASIFIKASI ---
    if st.button("JALANKAN KLASIFIKASI", type="primary", use_container_width=True):
        try:
            # Prediksi kelas
            pred = model.predict(input_df)
            # Ubah angka kembali ke teks (Good, Excellent, dll)
            label = le.inverse_transform(pred)[0]
            
            st.subheader("🎯 Hasil Analisis:")
            if "Good" in label or "Excellent" in label:
                st.success(f"### KATEGORI: {label}")
                st.balloons()
            else:
                st.warning(f"### KATEGORI: {label}")
                
        except Exception as e:
            st.error(f"Kesalahan saat klasifikasi: {e}")
            st.info("Jika error '_fill_dtype' masih muncul, silakan simpan ulang model (re-save) di lingkungan kerja Anda saat ini.")

else:
    st.error("Model tidak ditemukan. Pastikan file 'all_models_components.pkl' berada di folder yang sama dengan app.py.")
