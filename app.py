import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Kualitas Air", page_icon="💧", layout="centered")

# Custom CSS untuk tampilan lebih rapi
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .stNumberInput { margin-bottom: -10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_classifier():
    try:
        # Memuat file pkl
        data = joblib.load('all_models_components.pkl')
        # Mengambil model Random Forest (Klasifikasi) dari dictionary
        return data['random_forest']
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_classifier()

# --- 3. ANTARMUKA PENGGUNA ---
st.title("🌊 Sistem Klasifikasi Kualitas Air")
st.info("Masukkan parameter hasil laboratorium untuk menentukan kategori kualitas air.")

# Input dibagi menjadi dua kolom agar ringkas
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        ammonia = st.number_input("Ammonia (mg/l)", value=0.10, step=0.01)
        bod = st.number_input("BOD (mg/l)", value=2.00, step=0.01)
        do = st.number_input("Dissolved Oxygen (mg/l)", value=6.50, step=0.01)
        ortho = st.number_input("Orthophosphate (mg/l)", value=0.05, step=0.01)
        
    with col2:
        ph = st.number_input("pH (Derajat Keasaman)", value=7.00, min_value=0.0, max_value=14.0, step=0.1)
        temp = st.number_input("Suhu (°C)", value=28.0, step=0.1)
        nitrogen = st.number_input("Total Nitrogen (mg/l)", value=1.50, step=0.01)
        nitrate = st.number_input("Nitrate (mg/l)", value=0.80, step=0.01)

# Menyiapkan data untuk klasifikasi
# Nama kolom ini disesuaikan persis dengan isi file .pkl Anda
input_features = pd.DataFrame({
    'Ammonia (mg/l)': [ammonia],
    'Biochemical Oxygen Demand (mg/l)': [bod],
    'Dissolved Oxygen (mg/l)': [do],
    'Orthophosphate (mg/l)': [ortho],
    'pH (ph units)': [ph],
    'Temperature (cel)': [temp],
    'Nitrogen (mg/l)': [nitrogen],
    'Nitrate (mg/l)': [nitrate]
})

st.markdown("---")

# --- 4. PROSES KLASIFIKASI ---
if st.button("CEK KLASIFIKASI SEKARANG"):
    if model is not None:
        try:
            # Melakukan klasifikasi
            prediction = model.predict(input_features)
            label = prediction[0]
            
            st.subheader("🎯 Hasil Klasifikasi:")
            
            # Logika penentuan label (Sesuaikan dengan dataset Anda)
            if label == 0:
                st.success(f"### KELAS {label}: KUALITAS BAIK")
                st.balloons()
            elif label == 1:
                st.warning(f"### KELAS {label}: TERCEMAR RINGAN")
            else:
                st.error(f"### KELAS {label}: TERCEMAR BERAT")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
            st.info("Tips: Pastikan versi scikit-learn di GitHub sama dengan saat training.")
    else:
        st.error("Model tidak tersedia.")

# Footer
st.markdown("<br><hr><center>Aplikasi Klasifikasi Kualitas Air v1.0</center>", unsafe_allow_html=True)
