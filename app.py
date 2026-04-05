import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Tampilan Web (Lebar Penuh)
st.set_page_config(
    page_title="Sistem Pakar Kualitas Air", 
    page_icon="🌊", 
    layout="wide"
)

# --- CSS Kustom untuk Mempercantik Tampilan ---
st.markdown("""
<style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #6B7280; text-align: center; margin-bottom: 25px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 Sistem Klasifikasi Kualitas Air</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Integrasi Advanced Machine Learning (Stacking Ensemble: RF, XGB, GB)</div>', unsafe_allow_html=True)

# 2. Fungsi Load Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('water_pipeline.pkl')

try:
    artifacts = load_pipeline()
    st.sidebar.success("✅ Model & Instrumen Siap!")
except Exception as e:
    st.sidebar.error("❌ Gagal memuat file 'water_pipeline.pkl'")
    st.stop()

# --- MENU TAB ---
# Memecah web menjadi 3 bagian agar tidak terlalu panjang ke bawah
tab1, tab2, tab3 = st.tabs(["📊 Klasifikasi Data Baru", "📈 Eksplorasi Data", "🧠 Cara Kerja Sistem"])

with tab1:
    st.header("📝 Masukkan Parameter Air")
    st.write("Isi nilai parameter di bawah ini untuk melihat keputusan klasifikasi dari sistem.")
    
    # Form Input dibagi 4 kolom agar rapi dan tidak memakan tempat
    col1, col2, col3, col4 = st.columns(4)
    num_cols = artifacts['num_cols']
    user_inputs = {}
    
    for i, col_name in enumerate(num_cols):
        if i % 4 == 0: current_col = col1
        elif i % 4 == 1: current_col = col2
        elif i % 4 == 2: current_col = col3
        else: current_col = col4
            
        with current_col:
            user_inputs[col_name] = st.number_input(f"{col_name}", value=1.0, step=0.1)

    # Tombol Eksekusi
    if st.button("🚀 Jalankan Klasifikasi", type="primary", use_container_width=True):
        with st.spinner("Sistem sedang menganalisis..."):
            
            # --- PREPROCESSING ---
            df_new = pd.DataFrame([user_inputs])
            for col in artifacts['skewed_cols']:
                df_new[col] = np.log1p(df_new[col])
                
            df_new['BOD_DO_ratio'] = df_new['Biochemical Oxygen Demand (mg/l)'] / (df_new['Dissolved Oxygen (mg/l)'] + 1e-6)
            df_new['Ammonia_Nitrate_ratio'] = df_new['Ammonia (mg/l)'] / (df_new['Nitrate (mg/l)'] + 1e-6)
            df_new['Nutrient_sum'] = df_new['Ammonia (mg/l)'] + df_new['Nitrate (mg/l)'] + df_new['Orthophosphate (mg/l)']
            df_new['Temp_pH_interaction'] = df_new['Temperature (cel)'] * df_new['pH (ph units)']
            
            X_imputed = artifacts['imputer'].transform(df_new)
            X_scaled = artifacts['scaler'].transform(X_imputed)
            X_mi = X_scaled[:, artifacts['selected_mi']]
            X_final = artifacts['rfe'].transform(X_mi)
            
            # --- TAHAP KLASIFIKASI ---
            # Model Utama
            pred_stack = artifacts['model'].predict(X_final)
            label_stack = artifacts['label_encoder'].inverse_transform(pred_stack)[0]
            
            # 3 Algoritma Dasar
            rf_model = artifacts['model'].estimators_[0]
            xgb_model = artifacts['model'].estimators_[1]
            gb_model = artifacts['model'].estimators_[2]
            
            label_rf = artifacts['label_encoder'].inverse_transform(rf_model.predict(X_final))[0]
            label_xgb = artifacts['label_encoder'].inverse_transform(xgb_model.predict(X_final))[0]
            label_gb = artifacts['label_encoder'].inverse_transform(gb_model.predict(X_final))[0]
            
            # --- TAMPILKAN HASIL ---
            st.markdown("---")
            
            # Penentuan Warna Indikator Berdasarkan Hasil
            colors = {"Excellent": "#10B981", "Good": "#3B82F6", "Fair": "#F59E0B", "Marginal": "#EF4444", "Poor": "#7F1D1D"}
            bg_color = colors.get(label_stack, "#6B7280")
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; padding:20px; border-radius:10px; text-align:center; color:white;">
                <h2 style="margin:0;">HASIL REKOMENDASI (STACKING): {label_stack.upper()}</h2>
                <p style="margin:5px 0 0 0;">Meta-learner menggabungkan keputusan terbaik dari 3 algoritma.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.subheader("🔍 Keputusan Masing-Masing Algoritma Dasar")
            st.write("Sebelum digabung oleh Stacking, ini adalah hasil tebakan dari setiap algoritma:")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="Random Forest", value=label_rf)
            with res_col2:
                st.metric(label="XGBoost", value=label_xgb)
            with res_col3:
                st.metric(label="Gradient Boosting", value=label_gb)

with tab2:
    st.header("📈 Data Historis Kualitas Air")
    st.write("Unggah dataset `water_quality.csv` Anda di folder yang sama untuk menampilkan grafik ringkasan di sini.")
    
    try:
        df_csv = pd.read_csv("water_quality.csv", sep=";")
        
        # Visualisasi sederhana distribusi kelas
        st.subheader("Distribusi Kelas Kualitas Air (Dataset Asli)")
        st.bar_chart(df_csv['CCME_WQI'].value_counts())
        
        # Tampilkan tabel data
        st.subheader("Preview Dataset")
        st.dataframe(df_csv.head(10), use_container_width=True)
    except:
        st.info("💡 Grafik akan muncul di sini jika file `water_quality.csv` diunggah ke repositori GitHub Anda.")

with tab3:
    st.header("🧠 Bagaimana Sistem Ini Bekerja?")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### 1. Data Preprocessing & Rekayasa Fitur
        Data yang Anda masukkan tidak langsung dihitung oleh AI. Data tersebut melewati:
        * **Log Transformation:** Untuk mengatasi data yang tidak simetris (miring).
        * **Feature Engineering:** Membuat 4 rumus rasio baru untuk memperkuat deteksi.
        * **KNN Imputer & Scaler:** Menjaga rentang nilai agar adil bagi semua algoritma.
        """)
        
    with col_info2:
        st.markdown("""
        ### 2. Metode Stacking Ensemble
        Sistem ini tidak bergantung pada satu algoritma saja, melainkan menggabungkan:
        * **Random Forest** (Bagus untuk kestabilan data).
        * **XGBoost** (Sangat cepat dan akurat).
        * **Gradient Boosting** (Kuat dalam meminimalkan error).
        
        Hasil dari ketiga algoritma di atas dirangkum kembali menggunakan **Logistic Regression** sebagai pengambil keputusan akhir.
        """)
