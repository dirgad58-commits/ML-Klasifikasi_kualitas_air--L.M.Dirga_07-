import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="WaterQuality AI | Sistem Klasifikasi",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS UNTUK TAMPILAN LEBIH MENARIK ---
st.markdown("""
<style>
    /* Mengubah font global */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Mengubah warna background sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f8ff;
    }
    
    /* Styling tombol Prediksi */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        width: 100%;
        height: 3em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border-color: #0056b3;
        color: white;
        transform: scale(1.02);
    }
    
    /* Styling kotak hasil */
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Centering spinner */
    .stSpinner {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- FUNGSI LOAD MODEL (DI-CACHE) ---
@st.cache_resource
def load_model_data():
    """Memuat dictionary model dari file pkl."""
    try:
        data = joblib.load('all_models_components.pkl')
        return data
    except FileNotFoundError:
        st.error("⚠️ File 'all_models_components.pkl' tidak ditemukan. Pastikan file berada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat model: {e}")
        st.stop()

# Load data model
model_data = load_model_data()

# Memilih model (Misalnya Random Forest, sesuaikan kunci jika perlu)
# Berdasarkan file Anda, kuncinya adalah 'random_forest'
if 'random_forest' in model_data:
    final_model = model_data['random_forest']
else:
    st.error("⚠️ Model 'random_forest' tidak ditemukan dalam file pkl.")
    st.stop()


# --- HEADER & BANNER ---
st.markdown("<h1 style='text-align: center; color: #1e3f66;'>🌊 AI Monitoring Kualitas Air</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #555;'>Masukkan parameter kimia air di sebelah kiri untuk mengklasifikasikan status kelayakan air secara instant.</p>", unsafe_allow_html=True)
st.markdown("---")


# --- SIDEBAR: INPUT PARAMETER ---
with st.sidebar:
    st.image("https://img.icons8.com/external/flaticons-lineal-color-flat-icons/64/external-water-pollution-ecology-flaticons-lineal-color-flat-icons.png", width=60)
    st.header("Input Data Uji Air")
    st.write("Atur nilai sesuai hasil lab:")

    # Container untuk input agar rapi
    with st.container():
        ammonia = st.number_input("Ammonia (mg/l)", min_value=0.00, value=0.10, step=0.01, help="Kadar Amonia")
        bod = st.number_input("BOD (mg/l)", min_value=0.00, value=2.00, step=0.01, help="Biochemical Oxygen Demand")
        do = st.number_input("Dissolved Oxygen (mg/l)", min_value=0.00, value=6.50, step=0.10, help="Oksigen Terlarut")
        ortho = st.number_input("Orthophosphate (mg/l)", min_value=0.00, value=0.05, step=0.01)
        ph = st.slider("pH (Keasaman)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        temp = st.number_input("Temperature (°C)", min_value=0.0, value=27.0, step=0.1)
        nitrogen = st.number_input("Total Nitrogen (mg/l)", min_value=0.00, value=1.50, step=0.01)
        nitrate = st.number_input("Nitrate (mg/l)", min_value=0.00, value=0.80, step=0.01)

    # Membuat DataFrame dari input
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

    # Tombol Prediksi di Sidebar
    predict_btn = st.button("Analisis Kualitas Air")


# --- AREA UTAMA: TAMPILAN DATA & HASIL ---
col_data, col_result = st.columns([1, 1])

with col_data:
    st.subheader("📋 Ringkasan Data Input")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    
    # Menampilkan data input dalam tabel yang rapi
    st.dataframe(input_data.T.rename(columns={0: 'Nilai'}), use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


with col_result:
    st.subheader("🎯 Hasil Klasifikasi AI")
    
    if predict_btn:
        with st.spinner('Sedang menganalisis data...'):
            time.sleep(1.5) # Efek loading agar terlihat dramatis
            
            try:
                # Proses Prediksi
                prediction = final_model.predict(input_data)
                
                # Mengambil Probabilitas jika model mendukung
                try:
                    prediction_proba = final_model.predict_proba(input_data)
                    confidence = np.max(prediction_proba) * 100
                except:
                    confidence = None # Model tidak mendukung predict_proba

                # TAMPILAN HASIL BERDASARKAN LABEL KLASIFIKASI
                # ASUMSI: 0 = Baik, 1 = Tercemar Ringan, 2 = Tercemar Berat
                # SESUAIKAN LOGIKA IF/ELIF INI DENGAN LABEL ASLI DATA ANDA
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                if prediction[0] == 0:
                    st.balloons()
                    st.success("## ✨ STATUS: BAIK")
                    st.markdown("""
                        **Deskripsi:** Air memenuhi standar baku mutu. Aman untuk ekosistem air dan penggunaan tertentu (sesuai regulasi).
                        """)
                    icon = "✅"
                
                elif prediction[0] == 1:
                    st.warning("## ⚠️ STATUS: TERCEMAR RINGAN")
                    st.markdown("""
                        **Deskripsi:** Indikasi pencemaran ringan terdeteksi. Diperlukan monitoring lebih lanjut dan pembatasan aktivitas pembuangan limbah.
                        """)
                    icon = "🚧"
                
                else:
                    st.error("## 🚨 STATUS: TERCEMAR BERAT")
                    st.markdown("""
                        **Deskripsi:** Air terdeteksi berbahaya bagi ekosistem. Tindakan remediasi mendesak diperlukan. Tidak layak digunakan.
                        """)
                    icon = "🛑"
                
                # Menampilkan akurasi prediksi
                if confidence:
                    st.metric(label="Tingkat Keyakinan Model", value=f"{confidence:.2f}%", help="Seberapa yakin AI dengan hasil klasifikasi ini.")
                
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
                
    else:
        # Tampilan sebelum tombol ditekan
        st.info("👈 Silakan atur parameter air di sidebar dan klik tombol 'Analisis' untuk melihat hasil.")
        st.markdown("""
        <div style='text-align: center; margin-top: 50px; opacity: 0.5;'>
            <img src='https://img.icons8.com/fluency/96/water-analysis.png'/>
            <p>Menunggu Input...</p>
        </div>
        """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>Built with ❤️ by [Nama Anda] | Dataset: Monitoring Kualitas Air</p>
        <p style='font-size: 0.8em;'>Model: Random Forest Classifier (Compressed)</p>
    </div>
    """,
    unsafe_allow_html=True
)
