import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Kualitas Air", layout="centered")

# --- FUNGSI LOAD MODEL (Penyebab utama error jika gagal) ---
@st.cache_resource
def load_data():
    try:
        # Membaca file pkl
        model_dict = joblib.load('all_models_components.pkl')
        
        # Mengambil model Random Forest sesuai isi file Anda
        if isinstance(model_dict, dict) and 'random_forest' in model_dict:
            return model_dict['random_forest']
        else:
            # Jika file pkl bukan dict, coba kembalikan langsung
            return model_dict
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
        return None

model = load_data()

# --- TAMPILAN ---
st.title("💧 Klasifikasi Kualitas Air")
st.write("Masukkan parameter lab di bawah ini untuk melihat hasil klasifikasi.")

# Input dibuat simpel dan jelas
col1, col2 = st.columns(2)
with col1:
    ammonia = st.number_input("Ammonia (mg/l)", value=0.1)
    bod = st.number_input("BOD (mg/l)", value=2.0)
    do = st.number_input("Dissolved Oxygen (mg/l)", value=6.5)
    ortho = st.number_input("Orthophosphate (mg/l)", value=0.05)

with col2:
    ph = st.number_input("pH Air", value=7.0)
    temp = st.number_input("Temperature (°C)", value=28.0)
    nitrogen = st.number_input("Nitrogen (mg/l)", value=1.5)
    nitrate = st.number_input("Nitrate (mg/l)", value=0.8)

# Menyiapkan data (Nama kolom SANGAT PENTING harus sama dengan model)
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

# --- TOMBOL KLASIFIKASI ---
if st.button("JALANKAN KLASIFIKASI", type="primary", use_container_width=True):
    if model is not None:
        try:
            # Melakukan prediksi kelas
            hasil = model.predict(input_df)[0]
            
            st.subheader("Hasil Klasifikasi:")
            if hasil == 0:
                st.success(f"### KELAS {hasil}: KUALITAS BAIK ✅")
            elif hasil == 1:
                st.warning(f"### KELAS {hasil}: TERCEMAR RINGAN ⚠️")
            else:
                st.error(f"### KELAS {hasil}: TERCEMAR BERAT 🚨")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
            st.info("Saran: Pastikan nama kolom input sesuai dengan saat training.")
    else:
        st.error("Model tetap tidak ditemukan. Pastikan file 'all_models_components.pkl' sudah di-upload ke GitHub di folder yang sama.")
