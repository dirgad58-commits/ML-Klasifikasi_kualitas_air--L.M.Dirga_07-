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
        # Memuat file pkl
        components = joblib.load('all_models_components.pkl')
        return components
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

data = load_all_components()

if data:
    # --- 3. SIDEBAR ---
    st.sidebar.title("Konfigurasi")
    model_choice = st.sidebar.selectbox(
        "Pilih Algoritma:",
        ["Random Forest", "XGBoost", "Gradient Boosting"]
    )
    
    model_map = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "Gradient Boosting": "gradient_boosting"
    }
    
    selected_pipeline = data[model_map[model_choice]]
    le = data['label_encoder']

    # --- 4. TAMPILAN UTAMA ---
    st.title("🌊 Klasifikasi Kualitas Air")
    st.markdown("Nilai input di bawah ini telah diatur secara default untuk menghasilkan kategori **Good/Excellent**.")

    st.subheader("📝 Masukkan Parameter Air")

    # --- 5. INPUT DENGAN NILAI DEFAULT 'GOOD' ---
    # Nilai-nilai ini diatur agar model cenderung memprediksi hasil yang baik
    col1, col2, col3 = st.columns(3)

    with col1:
        ammonia = st.number_input("Ammonia (mg/l)", value=0.01, format="%.4f", help="Nilai rendah = Baik")
        bod = st.number_input("Biochemical Oxygen Demand (mg/l)", value=1.0, format="%.2f")
        do = st.number_input("Dissolved Oxygen (mg/l)", value=8.5, format="%.2f", help="Nilai tinggi = Baik")

    with col2:
        ortho = st.number_input("Orthophosphate (mg/l)", value=0.02, format="%.4f")
        ph = st.number_input("pH (ph units)", value=7.2, min_value=0.0, max_value=14.0, format="%.2f")
        temp = st.number_input("Temperature (cel)", value=24.0, format="%.1f")

    with col3:
        nitrogen = st.number_input("Nitrogen (mg/l)", value=0.5, format="%.2f")
        nitrate = st.number_input("Nitrate (mg/l)", value=0.2, format="%.2f")

    # Membuat DataFrame untuk prediksi
    # Pastikan nama kolom sesuai dengan 'all_feature_names' di file pkl Anda
    input_df = pd.DataFrame({
        'Ammonia (mg/l)': [ammonia],
        'Biochemical Oxygen Demand (mg/l)': [bod],
        'Dissolved Oxygen (mg/l)': [do],
        'Orthophosphate (mg/l)': [ortho],
        'pH (ph units)': [ph],
        'Temperature (cel)': [temp],
        'Nitrogen (mg/l)': [nitrogen],
        'Nitrate (mg/l)': [nitrate]
    })

    st.divider()

    # --- 6. TOMBOL KLASIFIKASI ---
    if st.button("🚀 JALANKAN KLASIFIKASI", type="primary", use_container_width=True):
        try:
            # Prediksi
            prediction_encoded = selected_pipeline.predict(input_df)
            prediction_label = le.inverse_transform(prediction_encoded)[0]
            
            # Probabilitas
            probs = selected_pipeline.predict_proba(input_df)
            confidence = np.max(probs) * 100

            # Tampilan Hasil
            st.subheader("🎯 Hasil Klasifikasi:")
            
            # Layout hasil
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if "Excellent" in prediction_label or "Good" in prediction_label:
                    st.success(f"### {prediction_label}")
                    st.balloons()
                else:
                    st.warning(f"### {prediction_label}")
            
            with res_col2:
                st.metric("Tingkat Keyakinan AI", f"{confidence:.2f}%")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

else:
    st.error("Model tidak ditemukan. Pastikan 'all_models_components.pkl' ada di folder yang sama.")
