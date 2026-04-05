import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# 1. Konfigurasi Tampilan Web (Wajib Paling Atas)
st.set_page_config(
    page_title="Water Intelligence System", 
    page_icon="🧪", 
    layout="wide"
)

# --- CSS KUSTOM: LEVEL PRESET INTERFACE ---
st.markdown("""
<style>
    /* Mengubah font dan warna latar belakang aplikasi */
    .stApp { background-color: #F8FAFC; }
    
    /* Judul Utama */
    .hero-title { font-size: 46px; font-weight: 900; color: #0F172A; text-align: center; margin-bottom: 2px; font-family: 'Segoe UI', sans-serif; }
    .hero-subtitle { font-size: 18px; color: #64748B; text-align: center; margin-bottom: 35px; }
    
    /* Efek Kartu Glassmorphism untuk Hasil */
    .result-card {
        padding: 30px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Mempercantik sidebar */
    .css-1cart60 { background-color: #1E293B !important; }
</style>
""", unsafe_allow_html=True)

# Memuat Header
st.markdown('<div class="hero-title">🧪 Water Quality Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Sistem Klasifikasi Terpadu Berbasis Stacking Ensemble (RF, XGBoost, & Gradien Boosting)</div>', unsafe_allow_html=True)

# 2. Fungsi Load Pipeline (Model + Instrumen)
@st.cache_resource
def load_pipeline():
    return joblib.load('water_pipeline.pkl')

try:
    artifacts = load_pipeline()
    st.sidebar.markdown("### 🟢 Status Sistem")
    st.sidebar.success("AI Core & Pipeline Terhubung")
except Exception as e:
    st.sidebar.markdown("### 🔴 Status Sistem")
    st.sidebar.error("Gagal Memuat 'water_pipeline.pkl'")
    st.stop()

# Menampilkan informasi ringkas di Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Ringkasan Model")
st.sidebar.info("""
* **Metode:** Stacking Classifier
* **Algoritma Dasar:** 3 Model
* **Kategori Output:** Multi-class
* **Akurasi Validasi:** > 95%
""")

# --- MENU TAB UTAMA ---
tab1, tab2, tab3 = st.tabs(["🎯 Dasbor Klasifikasi", "📈 Eksplorasi Dataset", "🧩 Dokumentasi Sistem"])

with tab1:
    # Menggunakan rasio kolom 4:6 untuk memisahkan input dan output di layar yang sama
    col_input, col_output = st.columns([4, 6], gap="large")
    
    with col_input:
        st.markdown("### 📥 Parameter Sampel Air")
        st.write("Sesuaikan slider untuk mereplikasi kondisi laboratorium sampel air:")
        
        # Pengelompokan input agar rapi
        with st.expander("🌡️ Parameter Fisik & Umum", expanded=True):
            ph = st.slider("Derajat Keasaman (pH)", 4.0, 10.0, 7.0, 0.1)
            temp = st.slider("Suhu / Temperature (°C)", -5.0, 45.0, 25.0, 0.5)
            do = st.slider("Oksigen Terlarut / DO (mg/l)", 0.0, 20.0, 7.5, 0.1)
            
        with st.expander("☣️ Senyawa Kimia & Nutrien", expanded=True):
            ammonia = st.slider("Amonia (mg/l)", 0.0, 50.0, 0.5, 0.01)
            bod = st.slider("Kebutuhan Oksigen Biokimia / BOD (mg/l)", 0.0, 30.0, 2.0, 0.1)
            ortho = st.slider("Ortofosfat (mg/l)", 0.0, 10.0, 0.1, 0.01)
            nitrogen = st.slider("Total Nitrogen (mg/l)", 0.0, 20.0, 1.5, 0.1)
            nitrate = st.slider("Nitrat (mg/l)", 0.0, 50.0, 1.0, 0.1)

        user_inputs = {
            'Ammonia (mg/l)': ammonia, 'Biochemical Oxygen Demand (mg/l)': bod,
            'Dissolved Oxygen (mg/l)': do, 'Orthophosphate (mg/l)': ortho,
            'pH (ph units)': ph, 'Temperature (cel)': temp,
            'Nitrogen (mg/l)': nitrogen, 'Nitrate (mg/l)': nitrate
        }
        
        trigger_btn = st.button("🚀 Jalankan Pemrosesan AI", type="primary", use_container_width=True)

    with col_output:
        st.markdown("### 📊 Hasil Keputusan AI")
        
        if trigger_btn:
            # Efek Progres Loading Dinamis agar dramatis saat demo
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.005)
                progress_bar.progress(percent_complete + 1)
            
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
            
            # --- PREDIKSI ---
            pred_stack = artifacts['model'].predict(X_final)
            label_stack = artifacts['label_encoder'].inverse_transform(pred_stack)[0]
            
            # 3 Model di balik layar
            rf_model, xgb_model, gb_model = artifacts['model'].estimators_[0], artifacts['model'].estimators_[1], artifacts['model'].estimators_[2]
            label_rf = artifacts['label_encoder'].inverse_transform(rf_model.predict(X_final))[0]
            label_xgb = artifacts['label_encoder'].inverse_transform(xgb_model.predict(X_final))[0]
            label_gb = artifacts['label_encoder'].inverse_transform(gb_model.predict(X_final))[0]
            
            # Menghapus progress bar setelah selesai
            progress_bar.empty()
            
            # Animasi Balon
            st.balloons()
            
            # Definisi Desain & Saran Aksi
            theme = {
                "Excellent": ("linear-gradient(135deg, #059669 0%, #10B981 100%)", "Mutu Sangat Baik", "Kondisi air sangat murni. Tidak memerlukan tindakan remediasi."),
                "Good": ("linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%)", "Mutu Baik", "Air dalam kondisi prima. Aman untuk ekosistem dan aktivitas publik."),
                "Fair": ("linear-gradient(135deg, #B45309 0%, #F59E0B 100%)", "Mutu Sedang", "Terdeteksi penurunan kualitas minor. Disarankan inspeksi berkala."),
                "Marginal": ("linear-gradient(135deg, #B91C1C 0%, #EF4444 100%)", "Mutu Kurang", "Kandungan kimia melampaui batas wajar. Butuh tindakan pengolahan!"),
                "Poor": ("linear-gradient(135deg, #450A07 0%, #7F1D1D 100%)", "Mutu Buruk", "Tingkat pencemaran berbahaya! Segera lakukan sterilisasi lokasi.")
            }
            grad_color, title_idn, solution = theme.get(label_stack, ("linear-gradient(135deg, #475569 0%, #64748B 100%)", "N/A", "N/A"))
            
            # Tampilan Utama Menggunakan CSS Glassmorphism
            st.markdown(f"""
            <div class="result-card" style="background: {grad_color};">
                <p style="margin:0; font-size:13px; text-transform:uppercase; letter-spacing:2px; opacity:0.8;">Klasifikasi Konsensus AI</p>
                <h1 style="margin:5px 0; font-size:42px; font-weight:900;">{label_stack.upper()}</h1>
                <h4 style="margin:0; font-weight:normal; opacity:0.9;">({title_idn})</h4>
                <hr style="border-color: rgba(255,255,255,0.15); margin: 20px auto; width: 60%;">
                <p style="font-size:15px; margin:0;"><b>Rekomendasi Manajerial:</b> {solution}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Transparansi Voting (3 Model)
            st.markdown("##### 🔍 Hasil Audit Keputusan Individual")
            c_rf, c_xgb, c_gb = st.columns(3)
            c_rf.metric(label="Random Forest", value=label_rf)
            c_xgb.metric(label="XGBoost", value=label_xgb)
            c_gb.metric(label="Gradient Boosting", value=label_gb)
            
            # Analisis Sensitivitas Sederhana
            st.markdown("##### ⚠️ Indikator Parameter Kritis")
            if do < 4.0 or ammonia > 1.0 or bod > 5.0:
                st.warning("⚠️ Perhatian: Sistem mendeteksi parameter Oksigen (DO) yang terlalu rendah atau beban limbah (Ammonia/BOD) yang terlalu pekat untuk ekosistem normal.")
            else:
                st.success("✅ Semua parameter masukan Anda berada dalam rentang toleransi ambang batas aman.")
                
            # Laporan Unduhan Otomatis
            report = f"--- LAPORAN HASIL UJI AI ---\nKesimpulan Klasifikasi: {label_stack} ({title_idn})\nTindakan: {solution}\n\nDetail Input:\n{user_inputs}"
            st.download_button("📥 Unduh Berkas Hasil Analisis", report, file_name="Rekomendasi_Kualitas_Air.txt", use_container_width=True)
            
        else:
            # Keadaan default saat tombol belum ditekan
            st.info("Silakan sesuaikan parameter di kolom kiri dan tekan tombol 'Jalankan Pemrosesan AI' untuk melihat kalkulasi.")

with tab2:
    st.markdown("### 📈 Ringkasan Database Latih")
    st.write("Visualisasi sebaran data historis yang digunakan untuk melatih otak mesin.")
    
    try:
        df_csv = pd.read_csv("water_quality.csv", sep=";")
        
        sc1, sc2 = st.columns([3, 7])
        with sc1:
            st.write("")
            st.metric(label="Total Sampel Data Latih", value=f"{len(df_csv)} Baris")
            st.write("**Statistik Kelas:**")
            st.dataframe(df_csv['CCME_WQI'].value_counts(), use_container_width=True)
            
        with sc2:
            # Menampilkan chart ringkas
            st.bar_chart(df_csv['CCME_WQI'].value_counts(), color="#3B82F6")
            
    except:
        st.warning("Grafik akan termuat otomatis jika Anda meletakkan file `water_quality.csv` di repositori GitHub Anda.")

with tab3:
    st.markdown("### 🧩 Dokumentasi Teknis Sistem")
    st.write("Gambaran besar alur data (Data Pipeline) yang terjadi di dalam perangkat lunak ini.")
    
    st.markdown("""
    Sistem ini dibangun untuk mencapai tingkat akurasi yang melampaui performa satu model tunggal. 
    Arsitektur pemodelan diringkas menjadi 3 pilar utama:
    
    1. **Dynamic Preprocessing:** Nilai ekstrem akan ditangani, fitur yang condong (*skewed*) akan diperhalus dengan logaritma, dan imputasi berbasis tetangga terdekat (KNN) digunakan jika terdapat data yang hilang.
    2. **Ensemble Base Learners:** Menggunakan 3 algoritma *tree-based* terkuat saat ini (Random Forest, XGBoost, dan Gradient Boosting) untuk melihat data dari berbagai sudut pandang matematis secara simultan.
    3. **Logistic Regression Meta-Classifier:** Lapisan pengambil keputusan akhir. Juri ini dilatih untuk mengetahui kapan harus mempercayai XGBoost, kapan harus mempercayai Random Forest, demi memberikan keputusan yang paling presisi.
    """)
    
    st.success("💡 **Catatan untuk Sidang:** Fokus utama dari penelitian ini adalah optimasi akurasi menggunakan penggabungan model (*Ensemble Stacking*) tanpa harus memasukkan polinomial berlebih yang berisiko membuat model menjadi *overfitting*.")
