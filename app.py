# app.py
# Aplikasi Streamlit untuk Klasifikasi Kualitas Air
# Memuat 3 model: Random Forest, XGBoost, Gradient Boosting

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

# Judul
st.title("💧 Klasifikasi Multi-Level Kualitas Air")
st.markdown("""
Aplikasi ini memprediksi kategori kualitas air (**Excellent, Good, Marginal, Fair, Poor**) 
berdasarkan parameter kimia/fisik air.  
Gunakan tiga model machine learning: **Random Forest**, **XGBoost**, dan **Gradient Boosting**.
""")

# Load model (cached agar tidak reload setiap interaksi)
@st.cache_resource
def load_models():
    model_path = "all_models_components.pkl"
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di {model_path}. Pastikan file sudah ada.")
        st.stop()
    data = joblib.load(model_path)
    return data

try:
    model_data = load_models()
    rf_model = model_data['random_forest']
    xgb_model = model_data['xgboost']
    gb_model = model_data['gradient_boosting']
    encoder = model_data['label_encoder']
    selected_features = model_data['selected_features']  # daftar 10 fitur terpilih
    all_features = model_data['all_feature_names']       # semua fitur asli (12)
    best_model_name = model_data['best_model_name']
    performance = model_data['model_performance']
    
    st.sidebar.success(f"✅ Model berhasil dimuat.\nModel terbaik: {best_model_name}")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Sidebar: pilih model
st.sidebar.header("🔧 Pengaturan")
model_option = st.sidebar.selectbox(
    "Pilih model untuk prediksi:",
    ["Random Forest", "XGBoost", "Gradient Boosting", "Model Terbaik (Otomatis)"]
)

# Tampilkan performa model di sidebar
st.sidebar.subheader("📊 Performa Model (Test Accuracy)")
for model, acc in performance.items():
    st.sidebar.write(f"- {model}: **{acc:.4f}**")

# Input parameter (hanya fitur yang terpilih oleh mutual information)
st.subheader("📝 Masukkan Parameter Air")
st.markdown(f"*Fitur yang digunakan (berdasarkan seleksi fitur):* {', '.join(selected_features)}")

# Buat kolom input dinamis berdasarkan selected_features
input_data = {}
cols = st.columns(3)  # bagi menjadi 3 kolom untuk tata letak
for i, feature in enumerate(selected_features):
    with cols[i % 3]:
        # Tampilkan tooltip deskripsi jika ada (opsional)
        help_text = f"Masukkan nilai {feature}"
        value = st.number_input(
            f"{feature}",
            value=0.0,
            format="%.6f",
            help=help_text,
            step=0.01
        )
        input_data[feature] = value

# Tombol prediksi
if st.button("🔍 Prediksi Kualitas Air", type="primary"):
    # Buat DataFrame dengan semua fitur (lengkapi dengan nilai default 0 untuk fitur tidak terpilih)
    # Karena pipeline model sudah berisi selector yang hanya mengambil 10 fitur, 
    # kita tetap perlu memberikan input untuk semua fitur asli (12) dengan nilai default 0.
    full_input = {col: 0.0 for col in all_features}
    for f in selected_features:
        full_input[f] = input_data[f]
    input_df = pd.DataFrame([full_input])
    
    # Pilih model sesuai opsi
    if model_option == "Random Forest":
        model = rf_model
    elif model_option == "XGBoost":
        model = xgb_model
    elif model_option == "Gradient Boosting":
        model = gb_model
    else:  # Model terbaik
        model = model_data[best_model_name.lower().replace(" ", "_")]
    
    # Prediksi
    pred_encoded = model.predict(input_df)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]
    
    # Tampilkan hasil
    st.subheader("🎯 Hasil Prediksi")
    
    # Warna berdasarkan kategori
    color_map = {
        "Excellent": "green",
        "Good": "blue",
        "Marginal": "orange",
        "Fair": "darkorange",
        "Poor": "red"
    }
    color = color_map.get(pred_label, "gray")
    
    st.markdown(f"### Kualitas Air: **<span style='color:{color}'>{pred_label}</span>**", unsafe_allow_html=True)
    
    # Tampilkan confidence/probability jika model mendukung (Random Forest dan Gradient Boosting)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        st.write("Probabilitas per kelas:")
        proba_df = pd.DataFrame({
            "Kelas": encoder.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)
        st.dataframe(proba_df.style.format({"Probabilitas": "{:.4f}"}))
    
    # Informasi tambahan
    st.info(f"Menggunakan model: **{model_option}**")
    
    # Opsi: tampilkan nilai input yang dimasukkan
    with st.expander("Lihat nilai input"):
        st.json(input_data)

# Footer
st.markdown("---")
st.caption("Dibangun dengan Streamlit | Model dilatih menggunakan dataset kualitas air Canada")
