import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Tampilan Web
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="centered")

st.title("🌊 Sistem Klasifikasi Kualitas Air")
st.write("Aplikasi ini menggunakan model Ensemble (Stacking) dengan akurasi tinggi untuk memprediksi kategori kualitas air.")
st.markdown("---")

# 1. Load Pipeline (Instrumen + Model)
@st.cache_resource
def load_pipeline():
    # Memuat file pkl yang berisi scaler, imputer, dan model
    return joblib.load('water_pipeline.pkl')

try:
    artifacts = load_pipeline()
    st.sidebar.success("✅ Model & Instrumen Berhasil Dimuat!")
except Exception as e:
    st.sidebar.error("❌ Gagal memuat file 'water_pipeline.pkl'. Pastikan file tersebut ada di folder yang sama.")
    st.stop()

# 2. Form Input Parameter
st.subheader("📝 Masukkan Parameter Air")
st.write("Silakan isi nilai untuk 8 parameter asli di bawah ini:")

col1, col2 = st.columns(2)
num_cols = artifacts['num_cols'] # Mengambil list 8 kolom asli dari pkl
user_inputs = {}

# Membagi form menjadi 2 kolom agar cantik
for i, col_name in enumerate(num_cols):
    with col1 if i % 2 == 0 else col2:
        # Default value diset ke 1.0 untuk mencegah pembagian dengan nol di feature engineering
        user_inputs[col_name] = st.number_input(f"{col_name}", value=1.0, step=0.1)

# 3. Tombol Eksekusi
if st.button("Prediksi Sekarang", type="primary"):
    with st.spinner("Sedang memproses data..."):
        
        # --- TAHAP PREPROCESSING (Meniru persis alur training Anda) ---
        
        # A. Ubah input user menjadi DataFrame
        df_new = pd.DataFrame([user_inputs])
        
        # B. Log Transform pada kolom yang miring
        for col in artifacts['skewed_cols']:
            df_new[col] = np.log1p(df_new[col])
            
        # C. Feature Engineering (Membuat 4 fitur rasio baru)
        df_new['BOD_DO_ratio'] = df_new['Biochemical Oxygen Demand (mg/l)'] / (df_new['Dissolved Oxygen (mg/l)'] + 1e-6)
        df_new['Ammonia_Nitrate_ratio'] = df_new['Ammonia (mg/l)'] / (df_new['Nitrate (mg/l)'] + 1e-6)
        df_new['Nutrient_sum'] = df_new['Ammonia (mg/l)'] + df_new['Nitrate (mg/l)'] + df_new['Orthophosphate (mg/l)']
        df_new['Temp_pH_interaction'] = df_new['Temperature (cel)'] * df_new['pH (ph units)']
        
        # D. Imputasi & Scaling
        X_imputed = artifacts['imputer'].transform(df_new)
        X_scaled = artifacts['scaler'].transform(X_imputed)
        
        # E. Seleksi Fitur (Mutual Information & RFE)
        X_mi = X_scaled[:, artifacts['selected_mi']]
        X_final = artifacts['rfe'].transform(X_mi)
        
        # --- TAHAP PREDIKSI ---
        pred_encoded = artifacts['model'].predict(X_final)
        
        # Decode kembali angka (0,1,2..) menjadi label teks (Good, Excellent..)
        pred_label = artifacts['label_encoder'].inverse_transform(pred_encoded)
        
        # --- TAMPILKAN HASIL ---
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        st.success(f"Status Kualitas Air: **{pred_label[0]}**")
