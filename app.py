import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Klasifikasi Kualitas Air | 3 Algoritma",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Tambahkan Bootstrap Icons dan Font Awesome (opsional) ---
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# CSS profesional (dengan penyesuaian untuk ikon)
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .title {
        color: #1e3a8a;
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-weight: 700;
        text-align: center;
        border-bottom: 2px solid #1e3a8a;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        color: #334155;
        margin-top: -10px;
        margin-bottom: 30px;
        font-size: 1rem;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        height: 100%;
        transition: all 0.2s;
    }
    .card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .model-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f172a;
        text-align: center;
        margin-bottom: 10px;
    }
    .status-label {
        font-size: 1rem;
        color: #475569;
        text-align: center;
    }
    .verdict {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin: 10px 0;
    }
    .confidence {
        text-align: center;
        font-size: 0.9rem;
        color: #2d3e50;
        margin-top: 8px;
    }
    .progress-bar-bg {
        background-color: #e2e8f0;
        border-radius: 12px;
        height: 10px;
        margin: 8px 0;
        overflow: hidden;
    }
    .progress-fill {
        background-color: #2563eb;
        height: 100%;
        width: 0%;
        border-radius: 12px;
    }
    .input-summary {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 20px 0;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }
    .stButton > button {
        background-color: #1e3a8a;
        color: white;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        width: 100%;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1e40af;
    }
    hr {
        margin: 10px 0;
    }
    /* Custom alert tanpa emoji */
    .custom-info {
        background-color: #e6f0fa;
        border-left: 4px solid #1e3a8a;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #0c4e6e;
    }
    </style>
""", unsafe_allow_html=True)

# Memuat semua aset model
@st.cache_resource
def load_all_assets():
    path = "models"
    with open(os.path.join(path, 'feature_info.json'), 'r') as f:
        info = json.load(f)
    
    scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))
    ohe = pickle.load(open(os.path.join(path, 'ohe.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(path, 'label_encoder.pkl'), 'rb'))
    
    models = {
        "CatBoost": pickle.load(open(os.path.join(path, 'catboost.pkl'), 'rb')),
        "LightGBM": pickle.load(open(os.path.join(path, 'lightgbm.pkl'), 'rb')),
        "HistGradientBoosting": pickle.load(open(os.path.join(path, 'histgradientboosting.pkl'), 'rb'))
    }
    
    return info, scaler, ohe, le, models

try:
    info, scaler, ohe, le, models = load_all_assets()
    # Mengganti st.success dengan custom HTML tanpa emoji
    st.markdown("""
    <div class="custom-info" style="background-color:#e0f2e0; border-left-color:#15803d;">
        <i class="bi bi-check-circle-fill" style="color:#15803d;"></i> Semua model dan preprocessor berhasil dimuat.
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.markdown(f"""
    <div class="custom-info" style="background-color:#fee2e2; border-left-color:#b91c1c;">
        <i class="bi bi-exclamation-triangle-fill" style="color:#b91c1c;"></i> Kesalahan fatal: Gagal memuat komponen. Detail: {e}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Header dengan ikon
st.markdown("""
<div class='title'>
    <i class="bi bi-droplet-half" style="font-size: 2rem; margin-right: 10px;"></i> 
    Analisis Klasifikasi Kualitas Air
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class='subtitle'>
    <i class="bi bi-bar-chart-steps"></i> Perbandingan Tiga Algoritma Gradient Boosting 
    | <i class="bi bi-cpu"></i> Prediksi Berbasis Fitur Fisika-Kimia Air
</div>
""", unsafe_allow_html=True)

# Nilai default untuk uji cepat (kondisi air baik)
default_vals = {
    'ph': 7.20,
    'do': 6.50,
    'bod': 2.10,
    'tc': 50.0,
    'tn': 1.20,
    'tp': 0.05,
    'ts': 150.0,
    'turb': 4.50,
    'temp': 25.0
}

# Section dengan ikon
st.markdown("""
<h3><i class="bi bi-pencil-square" style="margin-right: 8px;"></i> 1. Parameter Masukan (Data Laboratorium)</h3>
""", unsafe_allow_html=True)

# Custom info tanpa emoji
st.markdown("""
<div class="custom-info">
    <i class="bi bi-info-circle-fill" style="margin-right: 8px;"></i> 
    Nilai default di bawah mencerminkan kondisi air normal / baik. Anda dapat mengubahnya untuk menguji berbagai skenario kualitas air.
</div>
""", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    user_inputs = {}
    
    # Numerik
    for i, col_name in enumerate(info['numeric_cols']):
        target_col = [col1, col2, col3][i % 3]
        user_inputs[col_name] = target_col.number_input(
            label=f"{col_name.upper()}",
            value=default_vals.get(col_name, 0.0),
            format="%.2f",
            help=f"Masukkan nilai {col_name.upper()} sesuai hasil pengukuran."
        )
    
    st.markdown("---")
    # Kategorikal
    land_use_options = ohe.categories_[0].tolist()
    user_inputs['macro_land_use'] = st.selectbox(
        label="Klasifikasi Penggunaan Lahan (Macro Land Use)",
        options=land_use_options,
        index=0,
        help="Pilih tipe penggunaan lahan di sekitar sumber air."
    )
    
    submitted = st.form_submit_button("Jalankan Klasifikasi")

if submitted:
    # --- Preprocessing ---
    input_df = pd.DataFrame([user_inputs])
    X_num = input_df[info['numeric_cols']]
    X_cat = input_df[['macro_land_use']]
    
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)
    X_final = np.hstack([X_num_scaled, X_cat_encoded])
    
    # --- Ringkasan input ---
    st.markdown("""
    <h3><i class="bi bi-table"></i> 2. Ringkasan Parameter Input</h3>
    """, unsafe_allow_html=True)
    with st.expander("Lihat detail nilai input yang digunakan", expanded=False):
        summary_df = pd.DataFrame([user_inputs]).T.reset_index()
        summary_df.columns = ['Parameter', 'Nilai']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <h3><i class="bi bi-diagram-3"></i> 3. Hasil Klasifikasi Komparatif</h3>
    """, unsafe_allow_html=True)
    
    # Tampilkan hasil dalam 3 kolom
    cols = st.columns(3)
    results = []
    
    for idx, (model_name, model) in enumerate(models.items()):
        # Prediksi kelas
        pred_raw = model.predict(X_final)
        if isinstance(pred_raw, np.ndarray):
            pred_idx = int(pred_raw.flatten()[0]) if pred_raw.ndim > 1 else int(pred_raw[0])
        else:
            pred_idx = int(pred_raw)
        
        label = le.inverse_transform([pred_idx])[0].upper()
        
        # Probabilitas (confidence)
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_final)[0]
            confidence = proba[pred_idx] * 100
        
        results.append({
            "Model": model_name,
            "Prediksi": label,
            "Confidence (%)": confidence if confidence is not None else None,
            "Indeks": pred_idx
        })
        
        # Warna status
        if label in ["EXCELLENT", "GOOD"]:
            color = "#15803d"
            icon = "bi bi-check-circle-fill"
        elif label in ["MODERATE", "FAIR"]:
            color = "#d97706"
            icon = "bi bi-exclamation-triangle-fill"
        else:
            color = "#b91c1c"
            icon = "bi bi-x-circle-fill"
        
        with cols[idx]:
            st.markdown(f"""
            <div class='card'>
                <div class='model-name'><i class="bi bi-ml-model"></i> {model_name}</div>
                <hr>
                <div class='status-label'>Status Kualitas Air</div>
                <div class='verdict' style='color:{color};'>
                    <i class="{icon}" style="font-size: 1.5rem;"></i> {label}
                </div>
            """, unsafe_allow_html=True)
            
            if confidence is not None:
                st.markdown(f"""
                <div class='confidence'>
                    <i class="bi bi-bar-chart"></i> Tingkat Keyakinan: {confidence:.2f}%
                </div>
                <div class='progress-bar-bg'>
                    <div class='progress-fill' style='width:{confidence:.2f}%;'></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='confidence'><i class='bi bi-question-circle'></i> Informasi probabilitas tidak tersedia</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Tabel perbandingan ringkas ---
    st.markdown("""
    <h3><i class="bi bi-table"></i> 4. Perbandingan Ringkas</h3>
    """, unsafe_allow_html=True)
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Model")
    st.dataframe(results_df, use_container_width=True)
    
    # --- Metadata teknis (opsional) ---
    with st.expander("Lihat metadata vektor input (setelah normalisasi & encoding)"):
        st.code(str(X_final), language="python")
    
else:
    st.markdown("""
    <div class="custom-info">
        <i class="bi bi-info-circle"></i> Silakan masukkan parameter atau gunakan nilai default, 
        lalu tekan tombol <strong>Jalankan Klasifikasi</strong> untuk melihat hasil prediksi dari tiga algoritma.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <i class="bi bi-building"></i> Laboratorium Komputasi - Teknik Informatika - Universitas Halu Oleo<br>
    <i class="bi bi-diagram-3"></i> Model gradient boosting: CatBoost, LightGBM, HistGradientBoosting
</div>
""", unsafe_allow_html=True)
