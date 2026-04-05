import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Tampilan Web Premium
st.set_page_config(
    page_title="Water Quality Intelligence", 
    page_icon="🌊", 
    layout="wide"
)

# --- CSS Kustom untuk Membuat Tampilan Modern ---
st.markdown("""
<style>
    .main-title { font-size: 42px; font-weight: 800; color: #0F172A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #475569; text-align: center; margin-bottom: 30px; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 Dashboard Klasifikasi Kualitas Air</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Cerdas Pendukung Keputusan Berbasis Stacking Ensemble (RF, XGB, GB)</div>', unsafe_allow_html=True)

# 2. Fungsi Load Pipeline (Model + Instrumen)
@st.cache_resource
def load_pipeline():
    return joblib.load('water_pipeline.pkl')

try:
    artifacts = load_pipeline()
    st.sidebar.success("🟢 AI System & Instrumen Online!")
except Exception as e:
    st.sidebar.error("🔴 File 'water_pipeline.pkl' tidak ditemukan!")
    st.stop()

# --- MENU TAB ---
tab1, tab2, tab3 = st.tabs(["🎯 Klasifikasi & Analisis", "📊 Statistik Dataset", "🔬 Arsitektur AI"])

with tab1:
    st.subheader("📥 Input Parameter Sampel Air")
    st.write("Silakan geser atau masukkan nilai parameter fisik dan kimia air di bawah ini:")
    
    # Memakai Slider agar web lebih interaktif dibanding sekadar mengetik angka
    col1, col2 = st.columns(2)
    
    with col1:
        ammonia = st.slider("Ammonia (mg/l)", 0.0, 50.0, 0.5, 0.01)
        bod = st.slider("Biochemical Oxygen Demand (mg/l)", 0.0, 30.0, 2.0, 0.1)
        do = st.slider("Dissolved Oxygen (mg/l)", 0.0, 20.0, 7.5, 0.1)
        ortho = st.slider("Orthophosphate (mg/l)", 0.0, 10.0, 0.1, 0.01)
        
    with col2:
        ph = st.slider("pH Level", 4.0, 10.0, 7.0, 0.1)
        temp = st.slider("Temperature (°C)", -5.0, 45.0, 25.0, 0.5)
        nitrogen = st.slider("Nitrogen (mg/l)", 0.0, 20.0, 1.5, 0.1)
        nitrate = st.slider("Nitrate (mg/l)", 0.0, 50.0, 1.0, 0.1)

    # Menampung input dalam dict untuk diproses
    user_inputs = {
        'Ammonia (mg/l)': ammonia, 'Biochemical Oxygen Demand (mg/l)': bod,
        'Dissolved Oxygen (mg/l)': do, 'Orthophosphate (mg/l)': ortho,
        'pH (ph units)': ph, 'Temperature (cel)': temp,
        'Nitrogen (mg/l)': nitrogen, 'Nitrate (mg/l)': nitrate
    }

    st.write("")
    if st.button("🚀 Jalankan Analisis Cerdas", type="primary", use_container_width=True):
        with st.spinner("Model Stacking sedang berdiskusi menentukan hasil..."):
            
            # --- PROSES PIPELINE ---
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
            
            # --- PREDIKSI MULTI-ALGORITMA ---
            pred_stack = artifacts['model'].predict(X_final)
            label_stack = artifacts['label_encoder'].inverse_transform(pred_stack)[0]
            
            rf_model = artifacts['model'].estimators_[0]
            xgb_model = artifacts['model'].estimators_[1]
            gb_model = artifacts['model'].estimators_[2]
            
            label_rf = artifacts['label_encoder'].inverse_transform(rf_model.predict(X_final))[0]
            label_xgb = artifacts['label_encoder'].inverse_transform(xgb_model.predict(X_final))[0]
            label_gb = artifacts['label_encoder'].inverse_transform(gb_model.predict(X_final))[0]
            
            # --- TAMPILAN DASHBOARD HASIL ---
            st.markdown("---")
            st.subheader("📊 Hasil Keputusan Sistem")
            
            # Pengkondisian Warna & Rekomendasi
            theme = {
                "Excellent": ("#059669", "Sangat Baik", "Air sangat bersih. Aman digunakan untuk berbagai keperluan tanpa pengolahan khusus."),
                "Good": ("#2563EB", "Baik", "Air dalam kondisi baik. Sedikit treatment ringan mungkin dibutuhkan untuk konsumsi."),
                "Fair": ("#D97706", "Cukup / Sedang", "Terdapat indikasi penurunan kualitas air. Perlu pemantauan berkala."),
                "Marginal": ("#DC2626", "Kurang Baik", "Air tercemar ringan-sedang. Butuh penanganan sebelum digunakan untuk aktivitas."),
                "Poor": ("#7F1D1D", "Buruk", "Kondisi air kritis! Sangat berbahaya jika langsung digunakan. Butuh sterilisasi penuh.")
            }
            color, clean_title, action_plan = theme.get(label_stack, ("#475569", "Tidak Diketahui", "N/A"))
            
            # Kartu Output Utama
            st.markdown(f"""
            <div style="background-color:{color}; padding:25px; border-radius:12px; text-align:center; color:white; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
                <p style="margin:0; font-size:14px; text-transform:uppercase; letter-spacing:1px; opacity:0.8;">Status Akhir Sampel</p>
                <h1 style="margin:5px 0 10px 0; font-size:40px; font-weight:800;">{label_stack.upper()} ({clean_title})</h1>
                <hr style="border-color: rgba(255,255,255,0.2); margin:15px auto; width:50%;">
                <p style="margin:0; font-size:16px;"><b>Rekomendasi Tindakan:</b> {action_plan}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.write("")
            
            # Pembanding Algoritma (Menarik untuk Sidang)
            st.markdown("##### 🔍 Transparansi Voting Algoritma Dasar")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="Prediksi Random Forest", value=label_rf)
            with res_col2:
                st.metric(label="Prediksi XGBoost", value=label_xgb)
            with res_col3:
                st.metric(label="Prediksi Gradient Boosting", value=label_gb)
            
            # Fitur Download Laporan
            st.write("")
            report_text = f"LAPORAN ANALISIS KUALITAS AIR\n\nStatus Akhir: {label_stack}\nSaran: {action_plan}\n\nHasil Voting:\n- RF: {label_rf}\n- XGB: {label_xgb}\n- GB: {label_gb}"
            st.download_button(label="📥 Unduh Laporan (.txt)", data=report_text, file_name="laporan_kualitas_air.txt", mime="text/plain", use_container_width=True)

with tab2:
    st.subheader("📈 Analisis Sebaran Data Latih")
    st.write("Visualisasi ini membaca file `water_quality.csv` yang Anda lampirkan di repositori.")
    
    try:
        df_csv = pd.read_csv("water_quality.csv", sep=";")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("")
            st.write("**Total Data:**", len(df_csv))
            st.write("**Metrik Target:** `CCME_WQI`")
            st.dataframe(df_csv['CCME_WQI'].value_counts(), use_container_width=True)
            
        with c2:
            st.bar_chart(df_csv['CCME_WQI'].value_counts())
            
    except Exception as e:
        st.warning("Unggah file `water_quality.csv` di GitHub Anda agar visualisasi ini menyala secara dinamis.")

with tab3:
    st.subheader("🧠 Mengapa Menggunakan Stacking Ensemble?")
    st.write("Halaman ini menjelaskan kecanggihan sistem di balik layar untuk meyakinkan audiens atau penguji Anda.")
    
    col_flow1, col_flow2, col_flow3 = st.columns(3)
    
    with col_flow1:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🛠️ Step 1: Pre-Processing</h4>
            <p style="font-size:14px; color:#64748B;">
            Input pengguna tidak langsung diproses. AI melakukan <i>Log-transformation</i> untuk meredakan fitur yang miring, diikuti standarisasi nilai agar rentang angka adil.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_flow2:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🎭 Step 2: Klasifikasi Paralel</h4>
            <p style="font-size:14px; color:#64748B;">
            Data diumpankan ke 3 algoritma hebat secara mandiri: Random Forest, XGBoost, dan Gradient Boosting. Masing-masing mengeluarkan label pilihannya.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_flow3:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🏆 Step 3: Meta-Learner</h4>
            <p style="font-size:14px; color:#64748B;">
            Logistic Regression bertindak sebagai juri (Meta-Learner). Ia tidak melakukan voting mayoritas biasa, tetapi mempelajari bobot keakuratan 3 model di atas untuk memberikan keputusan mutlak.
            </p>
        </div>
        """, unsafe_allow_html=True)
