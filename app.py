import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- SETTING HALAMAN ---
st.set_page_config(page_title="Water Quality Classifier", page_icon="💧")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        data = joblib.load('all_models_components.pkl')
        # Pastikan mengambil model random_forest dari dictionary
        return data['random_forest']
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        return None

model = load_model()

# --- ANTARMUKA PENGGUNA (UI) ---
st.title("🛡️ Klasifikasi Kualitas Air")
st.markdown("""
Aplikasi ini mengelompokkan sampel air ke dalam kategori tertentu berdasarkan parameter laboratorium.
""")

# Input dibuat sederhana dalam satu blok agar mudah diisi
st.subheader("📊 Masukan Parameter Lab")
col1, col2 = st.columns(2)

with col1:
    ammonia = st.number_input("Ammonia (mg/l)", value=0.10, format="%.2f")
    bod = st.number_input("BOD (mg/l)", value=2.00, format="%.2f")
    do = st.number_input("Dissolved Oxygen (mg/l)", value=6.50, format="%.2f")
    ortho = st.number_input("Orthophosphate (mg/l)", value=0.05, format="%.2f")

with col2:
    ph = st.number_input("pH Air", value=7.00, min_value=0.0, max_value=14.0, format="%.2f")
    temp = st.number_input("Suhu (°C)", value=28.0, format="%.1f")
    nitrogen = st.number_input("Total Nitrogen (mg/l)", value=1.50, format="%.2f")
    nitrate = st.number_input("Nitrate (mg/l)", value=0.80, format="%.2f")

# Membuat Dataframe sesuai urutan fitur model
input_data = pd.DataFrame({
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

# --- LOGIKA KLASIFIKASI ---
if st.button("JALANKAN KLASIFIKASI", type="primary", use_container_width=True):
    if model:
        try:
            # Menggunakan .predict() untuk mendapatkan label kelas
            hasil_kelas = model.predict(input_data)[0]
            
            st.subheader("📍 Hasil Analisis Kategori:")
            
            # Menampilkan hasil berdasarkan kategori (Klasifikasi)
            # Catatan: Sesuaikan keterangan di bawah dengan urutan label asli Anda
            if hasil_kelas == 0:
                st.success(f"### KATEGORI: BAIK (Label {hasil_kelas})")
                st.write("Air memenuhi syarat kualitas dan aman bagi ekosistem.")
            elif hasil_kelas == 1:
                st.warning(f"### KATEGORI: TERCEMAR RINGAN (Label {hasil_kelas})")
                st.write("Ditemukan kontaminasi ringan, diperlukan pemantauan rutin.")
            else:
                st.error(f"### KATEGORI: TERCEMAR BERAT (Label {hasil_kelas})")
                st.write("Kualitas air buruk! Berisiko bagi kesehatan dan lingkungan.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
    else:
        st.error("Model tidak terdeteksi. Periksa file .pkl Anda.")
