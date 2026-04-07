import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Kualitas Air",
    page_icon="💧",
    layout="wide"
)

# Judul dan deskripsi
st.title("💧 Klasifikasi Multi-Level Kualitas Air")
st.markdown("""
Aplikasi ini memprediksi kategori kualitas air berdasarkan parameter fisika-kimia air.  
Hasil klasifikasi: **Excellent**, **Good**, **Marginal**, **Fair**, atau **Poor**.  
Pilih model machine learning dan masukkan nilai parameter di bawah.
""")

# Load model dengan caching
@st.cache_resource
def load_models():
    model_path = "saved_models/all_models_components.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan di '{model_path}'. Pastikan file sudah diupload.")
        st.stop()
    try:
        data = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat file model: {e}")
        st.stop()
    
    # Cek kelengkapan komponen
    required = ['random_forest', 'xgboost', 'gradient_boosting', 'label_encoder',
                'selected_features', 'all_feature_names', 'best_model_name', 'model_performance']
    for key in required:
        if key not in data:
            st.error(f"❌ File model tidak valid: key '{key}' tidak ditemukan.")
            st.stop()
    return data

# Load data
model_data = load_models()

# Ekstrak komponen
rf_pipeline = model_data['random_forest']
xgb_pipeline = model_data['xgboost']
gb_pipeline = model_data['gradient_boosting']
encoder = model_data['label_encoder']
selected_features = model_data['selected_features']   # 10 fitur terpilih
all_features = model_data['all_feature_names']        # 12 fitur asli
best_model_name = model_data['best_model_name']
performance = model_data['model_performance']

# Sidebar: pilihan model
st.sidebar.header("⚙️ Pengaturan Model")
model_option = st.sidebar.selectbox(
    "Pilih model untuk klasifikasi:",
    ["Random Forest", "XGBoost", "Gradient Boosting", f"Model Terbaik ({best_model_name})"]
)

# Tampilkan performa model di sidebar
st.sidebar.subheader("📊 Performa Model (Test Accuracy)")
for name, acc in performance.items():
    st.sidebar.write(f"- {name}: **{acc:.4f}**")
st.sidebar.markdown("---")
st.sidebar.info("Masukkan nilai parameter pada kolom input di bawah, lalu tekan tombol 'Klasifikasi'.")

# Input parameter (hanya 10 fitur terpilih, tetapi pipeline membutuhkan 12 fitur)
st.subheader("📝 Masukkan Parameter Air")
st.markdown(f"*Fitur yang digunakan model:* {', '.join(selected_features)}")

# Membagi input menjadi dua kolom
col1, col2 = st.columns(2)
input_values = {}
half = len(selected_features) // 2

with col1:
    for feat in selected_features[:half]:
        input_values[feat] = st.number_input(
            f"**{feat}**",
            value=0.0,
            format="%.6f",
            step=0.01,
            help=f"Masukkan nilai {feat} (mg/l atau satuan yang sesuai)"
        )
with col2:
    for feat in selected_features[half:]:
        input_values[feat] = st.number_input(
            f"**{feat}**",
            value=0.0,
            format="%.6f",
            step=0.01,
            help=f"Masukkan nilai {feat}"
        )

# Tombol klasifikasi
if st.button("🔍 Klasifikasi Kualitas Air", type="primary", use_container_width=True):
    # Buat dataframe dengan semua fitur (12 kolom)
    # Isi fitur yang tidak terpilih dengan nilai 0 (atau bisa median, tetapi 0 cukup untuk demo)
    full_input = {col: 0.0 for col in all_features}
    for feat in selected_features:
        full_input[feat] = input_values[feat]
    input_df = pd.DataFrame([full_input])
    
    # Pilih pipeline sesuai model
    if model_option == "Random Forest":
        pipeline = rf_pipeline
    elif model_option == "XGBoost":
        pipeline = xgb_pipeline
    elif model_option == "Gradient Boosting":
        pipeline = gb_pipeline
    else:  # Model terbaik
        key = best_model_name.lower().replace(" ", "_")
        pipeline = model_data[key]
    
    try:
        # Prediksi
        pred_encoded = pipeline.predict(input_df)[0]
        pred_label = encoder.inverse_transform([pred_encoded])[0]
        
        # Warna hasil
        color_map = {
            "Excellent": "green",
            "Good": "blue",
            "Marginal": "orange",
            "Fair": "darkorange",
            "Poor": "red"
        }
        color = color_map.get(pred_label, "gray")
        
        st.markdown(f"## 🎯 Hasil Klasifikasi: <span style='color:{color}; font-weight:bold'>{pred_label}</span>", unsafe_allow_html=True)
        
        # Probabilitas (jika model mendukung)
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0]
            proba_df = pd.DataFrame({
                "Kelas": encoder.classes_,
                "Probabilitas": proba
            }).sort_values("Probabilitas", ascending=False)
            st.write("📈 Probabilitas per Kelas:")
            st.dataframe(proba_df.style.format({"Probabilitas": "{:.4f}"}), use_container_width=True)
        
        st.caption(f"Menggunakan model: {model_option}")
        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat klasifikasi: {e}")
        st.info("Coba gunakan model lain atau periksa kembali input nilai.")

# Footer
st.markdown("---")
st.caption("Dibangun dengan Streamlit | Model dilatih menggunakan dataset kualitas air Canada")
