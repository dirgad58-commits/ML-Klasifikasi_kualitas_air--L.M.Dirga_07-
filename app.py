import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Canada Water Quality AI", page_icon="🇨🇦")

@st.cache_resource
def load_and_fix_model():
    try:
        # Load file pkl
        data = joblib.load('all_models_components.pkl')
        
        # Daftar model yang tersedia di file Anda
        models_dict = {
            'Gradient Boosting (Terbaik)': data['gradient_boosting'],
            'XGBoost': data['xgboost'],
            'Random Forest': data['random_forest']
        }
        
        # FIX: Ganti imputer yang rusak dengan yang baru untuk setiap model
        for name in models_dict:
            new_imputer = SimpleImputer(strategy='median')
            models_dict[name].steps[0] = ('imputer', new_imputer)
            
        return models_dict, data['label_encoder']
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

models, le = load_and_fix_model()

if models and le:
    st.title("🌊 Klasifikasi Kualitas Air Canada")
    
    # Pilih Model
    selected_model_name = st.selectbox("Pilih Model Algoritma:", list(models.keys()))
    model_pipeline = models[selected_model_name]

    # --- INPUT DATA (Default: Kategori GOOD) ---
    st.subheader("📝 Input Parameter Air")
    col1, col2 = st.columns(2)
    
    with col1:
        ammonia = st.number_input("Ammonia (mg/l)", value=0.03)
        bod = st.number_input("BOD (mg/l)", value=1.3)
        do = st.number_input("Dissolved Oxygen (mg/l)", value=8.1)
        ortho = st.number_input("Orthophosphate (mg/l)", value=0.01)
        
    with col2:
        ph = st.number_input("pH (ph units)", value=8.0)
        temp = st.number_input("Temperature (cel)", value=10.0)
        nitrogen = st.number_input("Nitrogen (mg/l)", value=0.34)
        nitrate = st.number_input("Nitrate (mg/l)", value=1.17)

    # DataFrame Input - PASTIKAN nama kolom sama persis dengan dataset asli
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

    # --- TOMBOL KLASIFIKASI ---
    if st.button("JALANKAN KLASIFIKASI", type="primary", use_container_width=True):
        try:
            # FIX ERROR 'X': Panggil predict langsung dengan input_df sebagai argumen pertama
            # Fit imputer baru dulu agar mengenali format data
            model_pipeline.named_steps['imputer'].fit(input_df)
            
            # Melakukan prediksi
            prediction = model_pipeline.predict(input_df) # Ini adalah X yang diminta
            
            # Mengubah hasil angka ke label teks (Excellent/Good/dll)
            label = le.inverse_transform(prediction)[0]
            
            # Menampilkan Hasil
            if label in ['Excellent', 'Good']:
                st.success(f"### HASIL: {label}")
                st.balloons()
            elif label == 'Fair':
                st.info(f"### HASIL: {label}")
            else:
                st.warning(f"### HASIL: {label}")

            # Probabilitas (Jika tersedia)
            try:
                prob = model_pipeline.predict_proba(input_df)
                st.write(f"Tingkat Keyakinan: **{np.max(prob)*100:.2f}%**")
            except:
                pass
                
        except Exception as e:
            st.error(f"Kesalahan saat pemrosesan: {e}")
else:
    st.warning("Pastikan file 'all_models_components.pkl' sudah ada di folder aplikasi.")
