import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Tampilan Web Portal Akademis
st.set_page_config(
    page_title="Sistem Informasi Kualitas Air", 
    page_icon="🔬", 
    layout="wide"
)

# --- CSS KUSTOM: GAYA PORTAL AKADEMIS ---
st.markdown("""
<style>
    /* Latar belakang aplikasi yang bersih */
    .stApp { background-color: #F8FAFC; }
    
    /* Tipografi Judul Gaya Jurnal/Portal Riset */
    .academic-title { font-size: 34px; font-weight: 700; color: #1E293B; text-align: center; margin-bottom: 2px; font-family: 'Georgia', serif; }
    .academic-subtitle { font-size: 16px; color: #64748B; text-align: center; margin-bottom: 30px; }
    
    /* Kartu Hasil dengan Gaya Minimalis Modern */
    .academic-card {
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    /* Merapikan Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; color: #475569; }
</style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown('<div class="academic-title">Sistem Pendukung Keputusan: Klasifikasi Kualitas Air</div>', unsafe_allow_html=True)
st.markdown('<div class="academic-subtitle">Implementasi Metode Stacking Ensemble (Random Forest, XGBoost, Gradient Boosting)</div>', unsafe_allow_html=True)

# 2. Fungsi Load Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('water_pipeline.pkl')

try:
    artifacts = load_pipeline()
    st.sidebar.markdown("### 🔍 Status Validasi")
    st.sidebar.success("Model Terverifikasi (Online)")
except Exception as e:
    st.sidebar.markdown("### 🔍 Status Validasi")
    st.sidebar.error("Gagal Memuat Pipeline")
    st.stop()

# Sidebar Deskripsi
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Informasi Penelitian")
st.sidebar.info("""
Aplikasi ini merupakan visualisasi hasil penelitian klasifikasi kualitas air dengan pendekatan **Ensemble Stacking**.
* **Variabel Input:** 8 Parameter
* **Metode Preprocessing:** Log-Transform, KNN Imputer, Robust Scaling, RFE.
""")

# --- MENU MENU UTAMA ---
tab1, tab2, tab3 = st.tabs(["🎯 Portal Klasifikasi", "📊 Eksplorasi Dataset", "🔬 Dokumen Arsitektur"])

with tab1:
    # Membagi layar menjadi 2 kolom (Kiri Input, Kanan Output)
    col_input, col_output = st.columns([4, 6], gap="large")
    
    with col_input:
        st.markdown("### 📥 Pengukuran Sampel")
        st.write("Silakan masukkan nilai parameter hasil uji laboratorium:")
        
        # Pengelompokan Parameter
        with st.expander("🌡️ Parameter Fisik & Umum", expanded=True):
            ph = st.slider("Derajat Keasaman (pH)", 4.0, 10.0, 7.0, 0.1)
            temp = st.slider("Suhu Air (°C)", -5.0, 45.0, 25.0, 0.5)
            do = st.slider("Oksigen Terlarut / DO (mg/l)", 0.0, 20.0, 7.5, 0.1)
            
        with st.expander("☣️ Parameter Kimia & Nutrien", expanded=True):
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

        st.write("")
        btn_analisis = st.button("Jalankan Klasifikasi Sistem", type="primary", use_container_width=True)

    with col_output:
        st.markdown("### 📊 Hasil Komputasi Model")
        
        if btn_analisis:
            with st.spinner("Sistem sedang memproses data..."):
                
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
                
                rf_model = artifacts['model'].estimators_[0]
                xgb_model = artifacts['model'].estimators_[1]
                gb_model = artifacts['model'].estimators_[2]
                
                label_rf = artifacts['label_encoder'].inverse_transform(rf_model.predict(X_final))[0]
                label_xgb = artifacts['label_encoder'].inverse_transform(xgb_model.predict(X_final))[0]
                label_gb = artifacts['label_encoder'].inverse_transform(gb_model.predict(X_final))[0]
                
                # Pengkondisian Warna Jurnal (Kalem/Pastel)
                theme = {
                    "Excellent": ("#0F5132", "Sangat Baik", "Sampel air memenuhi seluruh standar baku mutu dengan kategori sempurna."),
                    "Good": ("#084298", "Baik", "Sampel air dalam batas aman untuk ekosistem dan penggunaan umum."),
                    "Fair": ("#664D03", "Sedang / Cukup", "Ditemukan penurunan kualitas minor. Disarankan pemantauan berkala."),
                    "Marginal": ("#842029", "Kurang Baik", "Terdapat parameter yang melampaui ambang batas. Butuh pengolahan lanjutan."),
                    "Poor": ("#41060A", "Buruk", "Tingkat pencemaran tinggi! Tidak aman digunakan tanpa sterilisasi total.")
                }
                color, clean_title, action_plan = theme.get(label_stack, ("#475569", "N/A", "N/A"))
                
                # Kartu Hasil dengan Warna Jurnal
                st.markdown(f"""
                <div class="academic-card" style="background-color: {color};">
                    <p style="margin:0; font-size:11px; text-transform:uppercase; letter-spacing:2px; opacity:0.8;">Hasil Konsensus Ensemble</p>
                    <h1 style="margin:5px 0; font-size:38px; font-weight:800;">{label_stack.upper()}</h1>
                    <p style="margin:0; font-size:15px; opacity:0.9;">({clean_title})</p>
                    <hr style="border-color: rgba(255,255,255,0.15); margin:15px auto; width:50%;">
                    <p style="margin:0; font-size:14px;"><b>Catatan Ilmiah:</b> {action_plan}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Transparansi Voting Model
                st.markdown("##### 🔍 Distribusi Voting Algoritma Dasar")
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric(label="Random Forest", value=label_rf)
                with res_col2:
                    st.metric(label="XGBoost", value=label_xgb)
                with res_col3:
                    st.metric(label="Gradient Boosting", value=label_gb)
                
                # Analisis Risiko
                st.markdown("---")
                st.markdown("##### ⚠️ Deteksi Parameter Anomali")
                risks = []
                if do < 4.0: risks.append("Tingkat **Dissolved Oxygen (DO)** terlalu rendah (< 4.0 mg/l). Berisiko pada biota air.")
                if ammonia > 1.0: risks.append("Kadar **Ammonia** melampaui batas wajar (> 1.0 mg/l). Indikasi kontaminasi limbah.")
                if bod > 6.0: risks.append("Nilai **BOD** tinggi (> 6.0 mg/l). Menunjukkan tingginya beban pencemaran organik.")
                
                if risks:
                    for risk in risks:
                        st.warning(risk)
                else:
                    st.success("Seluruh parameter kimia kunci berada dalam ambang batas toleransi lingkungan.")
                
                # Fitur Download Laporan
                st.write("")
                report_text = f"LAPORAN HASIL UJI KLASIFIKASI AI\n\nStatus Akhir: {label_stack} ({clean_title})\n\nHasil Voting:\n- Random Forest: {label_rf}\n- XGBoost: {label_xgb}\n- Gradient Boosting: {label_gb}\n\nCatatan: {action_plan}"
                st.download_button(label="📥 Unduh Dokumen Hasil (.txt)", data=report_text, file_name="hasil_klasifikasi_air.txt", use_container_width=True)
        else:
            st.info("Atur parameter di kolom sebelah kiri dan tekan tombol 'Jalankan Klasifikasi Sistem' untuk melihat hasil kalkulasi.")

with tab2:
    st.subheader("📈 Analisis Sebaran Data Latih")
    st.write("Statistik deskriptif file `water_quality.csv` yang Anda simpan di repositori.")
    
    try:
        df_csv = pd.read_csv("water_quality.csv", sep=";")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("")
            st.metric(label="Total Observasi", value=f"{len(df_csv)} Sampel")
            st.write("**Frekuensi Kelas Target:**")
            st.dataframe(df_csv['CCME_WQI'].value_counts(), use_container_width=True)
            
        with c2:
            st.bar_chart(df_csv['CCME_WQI'].value_counts(), color="#0d6efd")
            
    except Exception as e:
        st.warning("Grafik otomatis akan muncul jika Anda mengunggah file `water_quality.csv` di repositori GitHub.")

with tab3:
    st.subheader("🔬 Metodologi & Kerangka Kerja Sistem")
    st.write("Penjelasan terstruktur mengenai pengolahan data dan perakitan algoritma.")
    
    st.markdown("""
    Sistem Pendukung Keputusan (SPK) ini menggunakan metode pemodelan tingkat lanjut untuk menjamin keakuratan klasifikasi:
    
    1. **Pre-Processing Tahap Awal:** Melibatkan transformasi logaritmik untuk menormalkan fitur yang miring (*skewed*), serta menggunakan KNN Imputer untuk mengisi data yang hilang secara cerdas.
    2. **Penerapan Algoritma Dasar (Base Learners):** Menggunakan 3 algoritma kuat berbasis pohon keputusan secara independen, yaitu *Random Forest*, *XGBoost*, dan *Gradient Boosting*.
    3. **Meta-Learner (Stacking):** Model Logistic Regression diletakkan di lapisan teratas untuk mengevaluasi prediksi dari ketiga model dasar tersebut. Pendekatan ini menghasilkan satu kesimpulan akhir yang solid dan meminimalkan bias.
    """)
    
    st.success("💡 **Tip Sidang:** Arsitektur ini membuktikan bahwa penggabungan beberapa algoritma (Ensemble) menghasilkan batas keputusan yang jauh lebih baik daripada hanya mengandalkan satu algoritma saja.")
