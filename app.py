import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

# 1. Konfigurasi Tampilan Web
st.set_page_config(
    page_title="Sistem Cerdas Kualitas Air", 
    page_icon="🌊", 
    layout="wide"
)

# --- CSS Kustom untuk Tampilan Profesional ---
st.markdown("""
<style>
    .main-title { font-size: 40px; font-weight: 800; color: #0F172A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #475569; text-align: center; margin-bottom: 30px; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    div[data-testid="stVerticalBlock"] > div { margin-bottom: -5px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 Dashboard Klasifikasi Kualitas Air</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Pendukung Keputusan Berbasis Ensemble Learning (RF, XGB, GB)</div>', unsafe_allow_html=True)

# 2. Fungsi Load Components (Model + Pipeline)
@st.cache_resource
def load_models():
    # Menyesuaikan path berdasarkan log output Anda (terkadang tersimpan di dalam folder saved_models)
    # Jika file pkl dipindah ke luar, ubah path di bawah ini menjadi 'all_models_components.pkl'
    try:
        return joblib.load('all_models_components.pkl')
   

try:
    components = load_models()
    rf_pipe = components['random_forest']
    xgb_pipe = components['xgboost']
    gb_pipe = components['gradient_boosting']
    le = components['label_encoder']
    
    st.sidebar.success("🟢 AI System & Pipeline Online!")
    st.sidebar.info(f"🏆 Model Terbaik Saat Latih:\n**{components['best_model_name']}**\nAkurasi: **{components['best_accuracy']*100:.2f}%**")
except Exception as e:
    st.sidebar.error("🔴 File 'all_models_components.pkl' tidak ditemukan! Pastikan file berada di folder yang sama.")
    st.stop()

# --- MENU TAB ---
tab1, tab2, tab3 = st.tabs(["🎯 Portal Analisis", "📊 Statistik Dataset", "🔬 Dokumentasi AI"])

with tab1:
    st.subheader("📥 Input Parameter Laboratorium")
    st.write("Masukkan nilai dari ke-8 parameter kualitas air di bawah ini:")
    st.write("")
    
    col_kiri, col_kanan = st.columns(2, gap="large")
    
    # Berdasarkan fitur dari "Fitur terpilih oleh mutual information"
    with col_kiri:
        st.markdown("##### 🧪 Senyawa Kimia")
        ammonia = st.number_input("Ammonia (mg/l)", min_value=0.0, max_value=100.0, value=0.05, step=0.01)
        bod = st.number_input("Biochemical Oxygen Demand / BOD (mg/l)", min_value=0.0, max_value=50.0, value=1.3, step=0.1)
        do = st.number_input("Dissolved Oxygen / DO (mg/l)", min_value=0.0, max_value=50.0, value=8.15, step=0.1)
        ortho = st.number_input("Orthophosphate (mg/l)", min_value=0.0, max_value=10.0, value=0.01, step=0.01)

    with col_kanan:
        st.markdown("##### 🌡️ Parameter Fisik & Nutrien")
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=9.8, step=0.5)
        nitrogen = st.number_input("Total Nitrogen (mg/l)", min_value=0.0, max_value=1000.0, value=0.34, step=0.1)
        nitrate = st.number_input("Nitrate (mg/l)", min_value=0.0, max_value=1000.0, value=1.17, step=0.1)

    user_inputs = {
        'Ammonia (mg/l)': ammonia, 
        'Biochemical Oxygen Demand (mg/l)': bod,
        'Dissolved Oxygen (mg/l)': do, 
        'Orthophosphate (mg/l)': ortho,
        'pH (ph units)': ph, 
        'Temperature (cel)': temp,
        'Nitrogen (mg/l)': nitrogen, 
        'Nitrate (mg/l)': nitrate
    }

    st.write("")
    if st.button("🚀 Eksekusi Klasifikasi Cerdas", type="primary", use_container_width=True):
        with st.spinner("Model sedang memproses melalui pipeline..."):
            
            # Konversi input
            df_new = pd.DataFrame([user_inputs])
            
            # --- PREDIKSI MODEL ---
            pred_rf = rf_pipe.predict(df_new)
            pred_xgb = xgb_pipe.predict(df_new)
            pred_gb = gb_pipe.predict(df_new)
            
            # --- DECODE LABEL (Excellent, Good, dsb) ---
            label_rf = le.inverse_transform(pred_rf)[0]
            label_xgb = le.inverse_transform(pred_xgb)[0]
            label_gb = le.inverse_transform(pred_gb)[0]
            
            # --- MAJORITY VOTING ---
            votes = [label_rf, label_xgb, label_gb]
            vote_counts = Counter(votes)
            label_final = vote_counts.most_common(1)[0][0]
            
            # --- TAMPILAN DASHBOARD HASIL ---
            st.markdown("---")
            st.subheader("📊 Keputusan Akhir Sistem")
            
            theme = {
                "Excellent": ("#059669", "Sangat Baik", "Air sangat bersih. Aman digunakan untuk berbagai keperluan tanpa pengolahan khusus."),
                "Good": ("#2563EB", "Baik", "Air dalam kondisi baik. Sedikit treatment ringan mungkin dibutuhkan untuk konsumsi."),
                "Fair": ("#D97706", "Cukup / Sedang", "Terdapat indikasi penurunan kualitas air. Perlu pemantauan berkala."),
                "Marginal": ("#DC2626", "Kurang Baik", "Air tercemar ringan-sedang. Butuh penanganan sebelum digunakan untuk aktivitas."),
                "Poor": ("#7F1D1D", "Buruk", "Kondisi air kritis! Sangat berbahaya jika langsung digunakan. Butuh sterilisasi penuh.")
            }
            color, clean_title, action_plan = theme.get(label_final, ("#475569", "Tidak Diketahui", "N/A"))
            
            st.markdown(f"""
            <div style="background-color:{color}; padding:25px; border-radius:12px; text-align:center; color:white; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
                <p style="margin:0; font-size:14px; text-transform:uppercase; letter-spacing:1px; opacity:0.8;">Status Klasifikasi</p>
                <h1 style="margin:5px 0 10px 0; font-size:42px; font-weight:800;">{label_final.upper()}</h1>
                <p style="margin:0; font-size:18px; font-weight:600;">{clean_title}</p>
                <hr style="border-color: rgba(255,255,255,0.2); margin:15px auto; width:50%;">
                <p style="margin:0; font-size:16px;"><b>Rekomendasi Manajerial:</b> {action_plan}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("##### 🔍 Transparansi Suara Algoritma Dasar")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label="Random Forest (Akurasi: 93.8%)", value=label_rf)
            with res_col2:
                st.metric(label="XGBoost (Akurasi: 94.2%)", value=label_xgb)
            with res_col3:
                st.metric(label="Gradient Boosting (Akurasi: 94.9%)", value=label_gb)
                
            report_text = f"LAPORAN ANALISIS KUALITAS AIR\n\nStatus Akhir: {label_final}\nSaran: {action_plan}\n\nHasil Voting:\n- Random Forest: {label_rf}\n- XGBoost: {label_xgb}\n- Gradient Boosting: {label_gb}\n\nParameter Uji:\n{user_inputs}"
            st.download_button(label="📥 Unduh Dokumen Laporan (.txt)", data=report_text, file_name="laporan_kualitas_air.txt", mime="text/plain", use_container_width=True)

with tab2:
    st.subheader("📈 Analisis Sebaran Data Latih")
    st.write("Data berikut bersumber dari file `Canada_dataset.csv` (Total: 3949 Baris).")
    
    try:
        df_csv = pd.read_csv("Canada_dataset.csv", sep=";")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("")
            st.metric(label="Total Observasi", value=f"{len(df_csv)} Sampel")
            st.write("**Metrik Target:** `CCME_WQI`")
            st.dataframe(df_csv['CCME_WQI'].value_counts(), use_container_width=True)
            
        with c2:
            st.bar_chart(df_csv['CCME_WQI'].value_counts(), color="#2563EB")
            
    except Exception as e:
        st.warning("Unggah file `Canada_dataset.csv` di GitHub Anda agar visualisasi ini berjalan dinamis.")

with tab3:
    st.subheader("🧠 Metodologi & Kerangka Kerja")
    st.write("Berdasarkan eksperimen data, sistem menggunakan pendekatan berikut:")
    
    col_flow1, col_flow2 = st.columns(2)
    
    with col_flow1:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🛠️ 1. Pipeline Imputasi & Seleksi</h4>
            <p style="font-size:14px; color:#64748B;">
            Input dari form ini dikirim menuju Pipeline. Data kosong/hilang diisi otomatis menggunakan <code>SimpleImputer(median)</code>. Setelah itu, sistem akan memilih fitur terbaik berdasarkan skor <i>Mutual Information</i>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_flow2:
        st.markdown("""
        <div style="background-color:#F8FAFC; padding:20px; border-radius:10px; border:1px solid #E2E8F0;">
            <h4>🏆 2. Hard Voting Ensemble</h4>
            <p style="font-size:14px; color:#64748B;">
            Daripada mempercayai satu model saja, prediksi dilakukan oleh 3 algoritma kuat secara bersamaan. Keputusan akhir mutlak diambil berdasarkan sistem musyawarah (voting suara terbanyak).
            </p>
        </div>
        """, unsafe_allow_html=True)
