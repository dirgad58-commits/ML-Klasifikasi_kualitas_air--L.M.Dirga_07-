import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Canada Water Quality Classifier",
    page_icon="🇨🇦",
    layout="wide"
)

# --- 2. FUNGSI LOAD & FIX MODEL ---
@st.cache_resource
def load_and_prepare_model():
    try:
        # Load file gabungan dari folder saved_models (sesuai script Anda)
        # Jika file berada di root, ganti menjadi 'all_models_components.pkl'
        data = joblib.load('all_models_components.pkl')
        
        # Simpan komponen ke dalam variabel
        models_dict = {
            'Gradient Boosting (Best)': data['gradient_boosting'],
            'XGBoost': data['xgboost'],
            'Random Forest': data['random_forest']
        }
        
        # Perbaikan Otomatis untuk error _fill_dtype
        # Kami mengganti imputer lama dengan yang baru di setiap pipeline
        for name in models_dict:
            new_imputer = SimpleImputer(strategy='median')
            models_dict[name].steps[0] = ('imputer', new_imputer)
            
        return models_dict, data['label_encoder'], data['all_feature_names']
    except Exception as e:
        st.error(f"Gagal memuat komponen model: {e}")
        return None, None, None

models, le, feature_names = load_and_prepare_model()

if models:
    # --- 3. SIDEBAR ---
    st.sidebar.header("Konfigurasi Model")
    selected_model_name = st.sidebar.selectbox("Pilih Algoritma", list(models.keys()))
    model_pipeline = models[selected_model_name]
    
    st.sidebar.divider()
    st.sidebar.success(f"Akurasi Model: {selected_model_name}")
    st.sidebar.write("- Excellent: Kualitas Sempurna")
    st.sidebar.write("- Good: Kualitas Layak")
    st.sidebar.write("- Fair: Cukup")
    st.sidebar.write("- Marginal: Terancam")
    st.sidebar.write("- Poor: Buruk/Tercemar")

    # --- 4. TAMPILAN UTAMA ---
    st.title("🇨🇦 Canada Water Quality Index (CCME WQI)")
    st.markdown("Klasifikasi multi-level kualitas air berdasarkan parameter kimia dan fisik.")

    # --- 5. INPUT DATA (NILAI DEFAULT: GOOD) ---
    st.subheader("📝 Masukkan Nilai Parameter")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Nilai default disesuaikan agar menghasilkan kategori 'Good'
            ammonia = st.number_input("Ammonia (mg/l)", value=0.03, format="%.4f")
            bod = st.number_input("BOD (mg/l)", value=1.30, format="%.2f")
            do = st.number_input("Dissolved Oxygen (mg/l)", value=8.10, format="%.2f")
            
        with col2:
            ortho = st.number_input("Orthophosphate (mg/l)", value=0.01, format="%.4f")
            ph = st.number_input("pH (ph units)", value=8.0, format="%.1f")
            temp = st.number_input("Temperature (cel)", value=10.0, format="%.2f")
            
        with col3:
            nitrogen = st.number_input("Nitrogen (mg/l)", value=0.34, format="%.4f")
            nitrate = st.number_input("Nitrate (mg/l)", value=1.17, format="%.4f")

    # Mapping input ke DataFrame
    # Pastikan urutan dan nama kolom sama persis dengan 'all_feature_names'
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

    st.divider()

    # --- 6. EKSEKUSI KLASIFIKASI ---
    if st.button("PROSES KLASIFIKASI", type="primary", use_container_width=True):
        try:
            # Fit imputer baru dengan data input saat ini untuk menghindari error _fill_dtype
            model_pipeline.named_steps['imputer'].fit(input_data)
            
            # Prediksi
            prediction = model_pipeline.predict(input_df=input_data if 'input_df' not in locals() else input_data)
            # Hasil prediksi (misal: [2]) diubah menjadi teks (misal: 'Good')
            result_text = le.inverse_transform(prediction)[0]
            
            # Tampilan Hasil
            st.subheader("🎯 Hasil Klasifikasi:")
            
            if result_text in ['Excellent', 'Good']:
                st.success(f"## Kualitas Air: {result_text}")
                st.balloons()
            elif result_text == 'Fair':
                st.info(f"## Kualitas Air: {result_text}")
            else:
                st.error(f"## Kualitas Air: {result_text}")

            # Tampilkan Probabilitas jika model mendukung
            try:
                prob = model_pipeline.predict_proba(input_data)
                confidence = np.max(prob) * 100
                st.write(f"Tingkat keyakinan model: **{confidence:.2f}%**")
            except:
                pass

        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
            st.info("Pastikan semua kolom input terisi dengan angka yang valid.")

else:
    st.warning("⚠️ Menunggu file 'all_models_components.pkl' diunggah...")
