import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Water Quality Classifier AI",
    page_icon="🌊",
    layout="wide"
)

# --- 2. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_all_components():
    try:
        # Memuat file pkl yang berisi dictionary semua model
        components = joblib.load('all_models_components.pkl')
        return components
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

data = load_all_components()

if data:
    # --- 3. SIDEBAR: PENGATURAN & INFORMASI ---
    st.sidebar.image("https://img.icons8.com/fluency/96/water-drawbridge.png", width=100)
    st.sidebar.title("Navigasi")
    
    # Pilih Model yang ingin digunakan
    st.sidebar.subheader("Pilih Model AI")
    model_choice = st.sidebar.selectbox(
        "Gunakan algoritma:",
        ["Random Forest", "XGBoost", "Gradient Boosting"]
    )
    
    # Mapping pilihan ke key di dictionary pkl
    model_map = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "Gradient Boosting": "gradient_boosting"
    }
    
    selected_pipeline = data[model_map[model_choice]]
    le = data['label_encoder']
    
    st.sidebar.divider()
    st.sidebar.write(f"**Model Terbaik Saat Training:** {data['best_model_name']}")
    st.sidebar.write(f"**Akurasi Terbaik:** {data['best_accuracy']:.2f}")

    # --- 4. TAMPILAN UTAMA ---
    st.title("🌊 Klasifikasi Kualitas Air (Multi-Model)")
    st.markdown("""
    Aplikasi ini mengklasifikasikan kualitas air berdasarkan parameter kimia dan fisik menggunakan teknik 
    *Feature Selection* dan *Imputation* otomatis.
    """)

    st.subheader("📝 Masukkan Parameter Air")
    st.info("Silakan isi nilai parameter di bawah ini sesuai hasil uji laboratorium.")

    # Membuat form input berdasarkan fitur yang ada di model
    # Kita gunakan fitur asli (all_feature_names) agar user bisa menginput semuanya, 
    # Selector di dalam pipeline akan memilih fitur yang dibutuhkan secara otomatis.
    all_features = data['all_feature_names']
    
    # Membagi input menjadi 3 kolom agar rapi
    cols = st.columns(3)
    user_inputs = {}

    for i, feature in enumerate(all_features):
        with cols[i % 3]:
            # Memberikan nilai default 0.0 atau median jika memungkinkan
            user_inputs[feature] = st.number_input(
                f"{feature}", 
                value=0.0, 
                format="%.4f",
                help=f"Masukkan nilai untuk {feature}"
            )

    # Membuat DataFrame dari input user
    input_df = pd.DataFrame([user_inputs])

    st.divider()

    # --- 5. PROSES KLASIFIKASI ---
    if st.button("🚀 JALANKAN KLASIFIKASI", type="primary", use_container_width=True):
        with st.spinner(f"Menganalisis menggunakan {model_choice}..."):
            try:
                # 1. Prediksi menggunakan pipeline (sudah termasuk imputer & selector)
                prediction_encoded = selected_pipeline.predict(input_df)
                
                # 2. Decode label angka ke teks asli (misal: Excellent, Good, Poor)
                prediction_label = le.inverse_transform(prediction_encoded)[0]
                
                # 3. Ambil probabilitas jika tersedia
                try:
                    probs = selected_pipeline.predict_proba(input_df)
                    confidence = np.max(probs) * 100
                except:
                    confidence = None

                # --- 6. TAMPILAN HASIL ---
                st.success("### Hasil Analisis Berhasil!")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Kategori Kualitas Air", prediction_label)
                
                with res_col2:
                    if confidence:
                        st.metric("Tingkat Keyakinan (Confidence)", f"{confidence:.2f}%")
                
                # Memberikan penjelasan visual berdasarkan hasil
                if "Excellent" in prediction_label or "Good" in prediction_label:
                    st.balloons()
                    st.info("💡 **Rekomendasi:** Air dalam kondisi layak dan memenuhi standar kualitas.")
                else:
                    st.warning("⚠️ **Peringatan:** Air terdeteksi memiliki kualitas rendah. Diperlukan penanganan lebih lanjut.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
                st.info("Pastikan semua nilai input sudah benar.")

else:
    st.error("File 'all_models_components.pkl' tidak ditemukan di direktori utama.")
    st.info("Harap pastikan Anda sudah mengunggah file model tersebut ke GitHub.")

# --- FOOTER ---
st.divider()
st.caption("Aplikasi Klasifikasi Kualitas Air - Dibuat dengan Streamlit & Scikit-Learn")
