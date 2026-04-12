import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Multi-Class Water Quality | 3 Algoritma",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Tambahkan Bootstrap Icons ---
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# CSS profesional dengan penekanan multi-class
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
        margin-bottom: 20px;
        font-size: 1rem;
    }
    .class-badge {
        display: inline-block;
        background-color: #e2e8f0;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        margin: 0 4px;
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
    .custom-info {
        background-color: #e6f0fa;
        border-left: 4px solid #1e3a8a;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #0c4e6e;
    }
    .comparison-table {
        margin-top: 20px;
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
        "LightGBM": pickle.load(open(os.path.join(path, 'lightgbm.pkl'), 'rb')),
        "CatBoost": pickle.load(open(os.path.join(path, 'catboost.pkl'), 'rb')),
        "HistGradientBoosting": pickle.load(open(os.path.join(path, 'histgradientboosting.pkl'), 'rb'))
    }
    
    return info, scaler, ohe, le, models

try:
    info, scaler, ohe, le, models = load_all_assets()
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

# --- Informasi kelas (dari label encoder) ---
available_classes = le.classes_.tolist()
class_names = [c.upper() for c in available_classes]

# Header dengan penjelasan multi-class
st.markdown(f"""
<div class='title'>
    <i class="bi bi-water" style="font-size: 2rem; margin-right: 10px;"></i> 
    Klasifikasi Multi-Kelas Kualitas Air
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class='subtitle'>
    <i class="bi bi-diagram-3"></i> Perbandingan Tiga Algoritma Gradient Boosting: 
    <strong>LightGBM</strong> | <strong>CatBoost</strong> | <strong>HistGradientBoosting</strong><br>
    <i class="bi bi-tags"></i> Kelas target: 
    {''.join([f'<span class="class-badge">{cls}</span>' for cls in class_names])}
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

st.markdown("""
<h3><i class="bi bi-pencil-square" style="margin-right: 8px;"></i> 1. Parameter Masukan (Data Laboratorium)</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-info">
    <i class="bi bi-info-circle-fill" style="margin-right: 8px;"></i> 
    Nilai default mencerminkan kondisi air normal/baik. Ubah nilai untuk menguji berbagai skenario.
    Sistem akan memprediksi kelas kualitas air (multi-kelas) menggunakan tiga model secara bersamaan.
</div>
""", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    user_inputs = {}
    
    for i, col_name in enumerate(info['numeric_cols']):
        target_col = [col1, col2, col3][i % 3]
        user_inputs[col_name] = target_col.number_input(
            label=f"{col_name.upper()}",
            value=default_vals.get(col_name, 0.0),
            format="%.2f",
            help=f"Masukkan nilai {col_name.upper()}"
        )
    
    st.markdown("---")
    land_use_options = ohe.categories_[0].tolist()
    user_inputs['macro_land_use'] = st.selectbox(
        label="Klasifikasi Penggunaan Lahan (Macro Land Use)",
        options=land_use_options,
        index=0,
        help="Pilih tipe penggunaan lahan di sekitar sumber air."
    )
    
    submitted = st.form_submit_button("Jalankan Klasifikasi")

if submitted:
    # Preprocessing
    input_df = pd.DataFrame([user_inputs])
    X_num = input_df[info['numeric_cols']]
    X_cat = input_df[['macro_land_use']]
    
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)
    X_final = np.hstack([X_num_scaled, X_cat_encoded])
    
    # Ringkasan input
    st.markdown("""
    <h3><i class="bi bi-table"></i> 2. Ringkasan Parameter Input</h3>
    """, unsafe_allow_html=True)
    with st.expander("Lihat detail nilai input yang digunakan", expanded=False):
        summary_df = pd.DataFrame([user_inputs]).T.reset_index()
        summary_df.columns = ['Parameter', 'Nilai']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <h3><i class="bi bi-diagram-3"></i> 3. Hasil Klasifikasi Komparatif (3 Algoritma)</h3>
    """, unsafe_allow_html=True)
    
    # Tampilkan hasil dalam 3 kolom
    cols = st.columns(3)
    results = []
    
    for idx, (model_name, model) in enumerate(models.items()):
        pred_raw = model.predict(X_final)
        if isinstance(pred_raw, np.ndarray):
            pred_idx = int(pred_raw.flatten()[0]) if pred_raw.ndim > 1 else int(pred_raw[0])
        else:
            pred_idx = int(pred_raw)
        
        label = le.inverse_transform([pred_idx])[0].upper()
        
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_final)[0]
            confidence = proba[pred_idx] * 100
        
        results.append({
            "Model": model_name,
            "Prediksi": label,
            "Confidence (%)": confidence if confidence is not None else None,
            "Indeks Kelas": pred_idx
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
                <div class='status-label'>Status Kualitas Air (Multi-Kelas)</div>
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
                st.markdown("<div class='confidence'><i class='bi bi-question-circle'></i> Probabilitas tidak tersedia</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Tabel perbandingan ringkas dan konsistensi ---
    st.markdown("""
    <h3><i class="bi bi-bar-chart-steps"></i> 4. Perbandingan & Konsistensi Prediksi</h3>
    """, unsafe_allow_html=True)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Model")
    st.dataframe(results_df, use_container_width=True)
    
    # Cek apakah semua model memberikan prediksi sama
    unique_preds = results_df['Prediksi'].nunique()
    if unique_preds == 1:
        st.markdown("""
        <div class="custom-info" style="background-color:#e0f2e0; border-left-color:#15803d;">
            <i class="bi bi-check2-all"></i> <strong>Konsisten:</strong> Ketiga model sepakat memprediksi kelas yang sama.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="custom-info" style="background-color:#fff3cd; border-left-color:#d97706;">
            <i class="bi bi-exclamation-triangle"></i> <strong>Perbedaan prediksi:</strong> Model memberikan hasil berbeda. 
            Perhatikan tingkat keyakinan masing-masing model untuk menentukan keputusan akhir.
        </div>
        """, unsafe_allow_html=True)
    
    # Tampilkan probabilitas per kelas jika tersedia (opsional untuk satu model)
    st.markdown("""
    <h3><i class="bi bi-graph-up"></i> 5. Detail Probabilitas per Kelas (Model dengan confidence tertinggi)</h3>
    """, unsafe_allow_html=True)
    # Cari model dengan confidence tertinggi
    best_model_row = results_df.dropna(subset=['Confidence (%)']).iloc[0] if not results_df['Confidence (%)'].isna().all() else None
    if best_model_row is not None:
        best_model_name = best_model_row.name
        best_model = models[best_model_name]
        if hasattr(best_model, "predict_proba"):
            proba_all = best_model.predict_proba(X_final)[0]
            proba_df = pd.DataFrame({
                "Kelas": [c.upper() for c in le.classes_],
                "Probabilitas (%)": proba_all * 100
            }).sort_values("Probabilitas (%)", ascending=False)
            st.dataframe(proba_df, use_container_width=True, hide_index=True)
            # Visualisasi sederhana dengan st.bar_chart
            st.bar_chart(proba_df.set_index("Kelas"))
    else:
        st.info("Tidak ada informasi probabilitas dari model yang tersedia.")
    
    # Metadata teknis
    with st.expander("Lihat metadata vektor input (setelah normalisasi & encoding)"):
        st.code(str(X_final), language="python")
else:
    st.markdown("""
    <div class="custom-info">
        <i class="bi bi-info-circle"></i> Silakan masukkan parameter atau gunakan nilai default, 
        lalu tekan <strong>Jalankan Klasifikasi</strong> untuk membandingkan prediksi dari LightGBM, CatBoost, dan HistGradientBoosting.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <i class="bi bi-building"></i> Laboratorium Komputasi - Teknik Informatika - Universitas Halu Oleo<br>
    <i class="bi bi-diagram-3"></i> Algoritma: LightGBM, CatBoost, HistGradientBoosting | Klasifikasi Multi-Kelas
</div>
""", unsafe_allow_html=True)
