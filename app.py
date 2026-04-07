import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

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
    
    /* Mengurangi jarak antar input agar lebih padat dan rapi */
    div[data-testid="stVerticalBlock"] > div { margin-bottom: -5px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 Dashboard Klasifikasi Kualitas Air</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Cerdas Pendukung Keputusan Berbasis Machine Learning (RF, XGB, GB)</div>', unsafe_allow_html=True)

# 2. Fungsi Load Components (Model + Pipeline)
@st.cache_resource
def load_models():
    # Menyesuaikan dengan nama file model yang baru
    return joblib.load('all_models_components.pkl')

try:
    components = load_models()
    rf_pipe = components['random_forest']
    xgb_pipe = components['xgboost']
    gb_pipe = components['gradient_boosting']
    le = components['label_encoder']
    st.sidebar.success("🟢 AI System & Pipeline Online!")
except Exception as e:
    st.sidebar.error("🔴 File 'all_models_components.pkl' tidak ditemukan!")
    st.stop()

# --- MENU TAB ---
tab1, tab2, tab3 = st.tabs(["🎯 Klasifikasi & Analisis", "📊 Statistik Dataset", "🔬 Arsitektur AI"])

with tab1:
    st.subheader("📥 Input Parameter Sampel Air")
    st.write("Silakan ketikkan nilai parameter fisik dan kimia air di bawah ini:")
    st.write("")
    
    # Membagi input angka menjadi 2 kolom agar seimbang
    col_kiri, col_kanan = st.columns(2, gap="large")
    
    with col_kiri:
        st.markdown("##### 🧪 Senyawa Kimia")
        ammonia = st.number_input("Ammonia (mg/l)", min_value=0.0, max_value=50.0, value=0.50, step=0.01)
        bod = st.number_input("Biochemical Oxygen Demand / BOD (mg/l)", min_value=0.0, max_value=30.0, value=2.0, step=0.1)
        do = st.number_input("Dissolved Oxygen / DO (mg/l)", min_value=0.0, max_value=20.0, value=7.5, step=0.1)
        ortho = st.number_input("Orthophosphate (mg/l)", min_value=0.0, max_value=10.0, value=0.10, step=0.01)

    with col_kanan:
        st.markdown("##### 🌡️ Parameter Fisik & Nutrien")
        ph = st.number_input("pH Level", min_value=4.0, max_value=14.0, value=7.0, step=0.1)
        temp = st.number_input("Temperature (°C)", min_value=-5.0, max_value=45.0, value=25.0, step=0.5)
        nitrogen = st.number_input("Total Nitrogen (mg/l)", min_value=0.0, max_value=50.0, value=1.5, step=0.1)
        nitrate = st.number_input("Nitrate (mg/l)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)

    # Menampung input dalam dict. PASTIKAN nama kolom sama persis dengan dataset
    user_inputs = {
        'Ammonia (mg/l)': ammonia, 'Biochemical Oxygen Demand (mg/l)': bod,
        'Dissolved Oxygen (mg/l)': do, 'Orthophosphate (mg/l)': ortho,
        'pH (ph units)': ph, 'Temperature (cel)': temp,
        'Nitrogen (mg/l)': nitrogen, 'Nitrate (mg/l)': nitrate
    }

    st.write("")
    if st.button("🚀 Klasifikasi", type="primary", use_container_width=True):
        with st.spinner("Model sedang menganalisis data..."):
            
            # Konversi input ke DataFrame
            df_new = pd.DataFrame([user_inputs])
            
            # --- PREDIKSI MULTI-ALGORITMA ---
            # Model Pipeline di .pkl sudah otomatis menangani missing values & feature selection
            pred_rf = rf_pipe.predict(df_new)
            pred_xgb = xgb_pipe.predict(df_new)
            pred_gb = gb_pipe.predict(df_new)
            
            # Decode label dari angka kembali ke teks (Excellent, Good, dsb)
            label_rf = le.inverse_transform(pred_rf)[0]
            label_xgb = le.inverse_transform(pred_xgb)[0]
            label_gb = le.inverse_transform(pred_gb)[0]
            
            # --- KONSENSUS (MAJORITY VOTING) ---
            # Mengambil keputusan akhir berdasarkan suara terbanyak dari 3 model
            votes = [label_rf, label_xgb, label_gb]
            vote_counts = Counter(votes)
            label_final = vote_counts.most_common(1)[0][0]
            
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
            color, clean_title, action_plan = theme.get(label_final, ("#475569", "Tidak Diketahui", "N/A"))
            
            # Kartu Output Utama
            st.markdown(f"""
            <div style="background-color:{color}; padding:25px; border-radius:12px; text-align:center; color:white; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
                <p style="margin:0; font-size:14px; text-transform:uppercase; letter-spacing:1px; opacity:0.8;">Status Akhir Konsensus</p>
                <h1 style="margin:5px 0 10px 0; font-size:40px; font-weight:800;">{label_final.upper()} ({clean_title})</h1>
                <hr style="border-color: rgba(255,255,255,0.2); margin:15px auto; width:50%;">
                <p style="margin:0; font-size:16px;"><b>Rekomendasi Tindakan:</b> {action_plan}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.write("")
            
            # Pembanding Algoritma
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
            report_text = f"LAPORAN ANALISIS KUALITAS AIR\n\nStatus Akhir: {label_final}\nSaran: {action_plan}\n\nHasil Voting:\n- RF: {label_rf}\n- XGB: {label_xgb}\n- GB: {label_gb}"
            st.download_button(label="📥 Unduh Laporan (.txt)", data=report_text, file_name="laporan_kualitas_air.txt", mime="text/plain", use_container_width=True)

with tab2:
    st.subheader("📈 Analisis Sebaran Data Latih")
    # Anda bisa mengganti nama file csv di bawah ini dengan Canada_dataset.csv jika itu yang digunakan
    dataset_name = "Canada_dataset.csv" 
    st.write(f"Visualisasi ini membaca file `{dataset_name}` yang Anda lampirkan di repositori.")
    
    try:
        df_csv = pd.read_csv(dataset_name, sep=";")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("")
            st.write("**Total Data:**", len(df_csv))
            st.write("**Metrik Target:** `CCME_WQI`")
            st.dataframe(df_csv['CCME_WQI'].value_counts(), use_container_width=True)
            
        with c2:
            st.bar_chart(df_csv['CCME_WQI'].value_counts())
            
    except Exception as e:
        st.warning(f"Unggah file `{dataset_name}` di GitHub Anda agar visualisasi ini menyala secara dinamis.")

with tab3:
    st.subheader("🧠 Mengapa Menggunakan Pendekatan Ini?")
    st.write("Sistem ini terintegrasi langsung dengan Pipeline Scikit-Learn yang mengotomatisasi pemrosesan data mentah.")
    
    col_flow1, col_flow2, col_flow3 = st.columns(3)
    
    with col_flow1:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🛠️ Step 1: Automated Pre-Processing</h4>
            <p style="font-size:14px; color:#64748B;">
            Input dimasukkan ke dalam Pipeline. Imputasi data kosong diselesaikan otomatis, dan fitur yang paling berdampak disaring menggunakan <i>Mutual Information</i>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_flow2:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🎭 Step 2: Paralel Klasifikasi</h4>
            <p style="font-size:14px; color:#64748B;">
            Data diumpankan ke 3 algoritma hebat secara mandiri: Random Forest, XGBoost, dan Gradient Boosting untuk menghindari bias satu model.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_flow3:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🏆 Step 3: Majority Voting</h4>
            <p style="font-size:14px; color:#64748B;">
            Hasil akhir ditentukan dari konsensus (suara terbanyak) ketiga algoritma tersebut untuk memberikan keputusan mutlak yang presisi dan stabil.
            </p>
        </div>
        """, unsafe_allow_html=True)
