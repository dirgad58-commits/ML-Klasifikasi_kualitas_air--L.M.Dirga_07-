import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Klasifikasi Kualitas Air", page_icon="💧", layout="wide")

st.title("🌊 Sistem Klasifikasi Kualitas Air (Multi-Algoritma)")
st.write("Aplikasi ini menggunakan teknik **Stacking Ensemble** yang menggabungkan 3 algoritma untuk mengklasifikasikan kategori kualitas air.")
st.markdown("---")

# 1. Load Pipeline (Instrumen + Model)
@st.cache_resource
def load_pipeline():
    return joblib.load('water_pipeline.pkl')

try:
    artifacts = load_pipeline()
    st.sidebar.success("✅ Model & Instrumen Berhasil Dimuat!")
except Exception as e:
    st.sidebar.error("❌ Gagal memuat file 'water_pipeline.pkl'. Pastikan file tersebut ada di folder yang sama.")
    st.stop()

# 2. Form Input Parameter (8 Parameter Asli)
st.subheader("📝 Masukkan Parameter Air")

col1, col2 = st.columns(2)
num_cols = artifacts['num_cols'] # 8 nama kolom asli
user_inputs = {}

for i, col_name in enumerate(num_cols):
    with col1 if i % 2 == 0 else col2:
        user_inputs[col_name] = st.number_input(f"{col_name}", value=1.0, step=0.1)

# 3. Tombol Eksekusi
if st.button("Klasifikasikan Kualitas Air", type="primary"):
    with st.spinner("Sedang memproses data..."):
        
        # --- TAHAP PREPROCESSING ---
        df_new = pd.DataFrame([user_inputs])
        
        # Log Transform
        for col in artifacts['skewed_cols']:
            df_new[col] = np.log1p(df_new[col])
            
        # Feature Engineering (Rasio)
        df_new['BOD_DO_ratio'] = df_new['Biochemical Oxygen Demand (mg/l)'] / (df_new['Dissolved Oxygen (mg/l)'] + 1e-6)
        df_new['Ammonia_Nitrate_ratio'] = df_new['Ammonia (mg/l)'] / (df_new['Nitrate (mg/l)'] + 1e-6)
        df_new['Nutrient_sum'] = df_new['Ammonia (mg/l)'] + df_new['Nitrate (mg/l)'] + df_new['Orthophosphate (mg/l)']
        df_new['Temp_pH_interaction'] = df_new['Temperature (cel)'] * df_new['pH (ph units)']
        
        # Imputasi & Scaling
        X_imputed = artifacts['imputer'].transform(df_new)
        X_scaled = artifacts['scaler'].transform(X_imputed)
        
        # Seleksi Fitur
        X_mi = X_scaled[:, artifacts['selected_mi']]
        X_final = artifacts['rfe'].transform(X_mi)
        
        # --- TAHAP KLASIFIKASI ---
        # 1. Hasil Model Utama (Stacking)
        pred_stack = artifacts['model'].predict(X_final)
        label_stack = artifacts['label_encoder'].inverse_transform(pred_stack)[0]
        
        # 2. Hasil Ekstraksi 3 Algoritma di dalam Stacking
        # Kita ambil estimator yang sudah dilatih di dalam model stacking
        rf_model = artifacts['model'].estimators_[0]
        xgb_model = artifacts['model'].estimators_[1]
        gb_model = artifacts['model'].estimators_[2]
        
        # Prediksi masing-masing algoritma
        label_rf = artifacts['label_encoder'].inverse_transform(rf_model.predict(X_final))[0]
        label_xgb = artifacts['label_encoder'].inverse_transform(xgb_model.predict(X_final))[0]
        label_gb = artifacts['label_encoder'].inverse_transform(gb_model.predict(X_final))[0]
        
        # --- TAMPILKAN HASIL ---
        st.markdown("---")
        st.subheader("📊 Hasil Klasifikasi")
        
        # Menampilkan Hasil Utama
        st.info(f"💡 **Rekomendasi Akhir (Stacking Ensemble):** Kualitas Air masuk dalam kategori **{label_stack}**")
        
        st.markdown("### 🔍 Perbandingan Keputusan Algoritma:")
        # Membuat 3 kolom untuk menampilkan hasil tiap algoritma
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(label="Klasifikasi Random Forest", value=label_rf)
        with res_col2:
            st.metric(label="Klasifikasi XGBoost", value=label_xgb)
        with res_col3:
            st.metric(label="Klasifikasi Gradient Boosting", value=label_gb)
