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

# --- Bootstrap Icons & Font Awesome (opsional) ---
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# CSS Profesional dengan warna interaktif & proporsi seimbang
st.markdown("""
<style>
    /* Global */
    .main {
        background: linear-gradient(135deg, #f5f7fc 0%, #eef2f8 100%);
        font-family: 'Inter', sans-serif;
    }
    /* Header */
    .title {
        background: linear-gradient(90deg, #0a2b4e, #1e4a76);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        text-align: center;
        padding-bottom: 10px;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        text-align: center;
        color: #2c3e66;
        font-size: 1rem;
        margin-bottom: 30px;
        border-bottom: 1px solid #cbd5e1;
        display: inline-block;
        padding-bottom: 8px;
        width: auto;
        margin-left: auto;
        margin-right: auto;
    }
    /* Badge kelas */
    .class-badge {
        display: inline-block;
        background: #e2e8f0;
        border-radius: 30px;
        padding: 4px 14px;
        margin: 0 5px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #1e293b;
        box-shadow: inset 0 1px 1px rgba(0,0,0,0.05);
    }
    /* Card model */
    .card {
        background: white;
        border-radius: 20px;
        padding: 1.2rem 0.8rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.25s ease;
        height: 100%;
        border: 1px solid rgba(203,213,225,0.4);
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -12px rgba(0,0,0,0.15);
        border-color: #b9c8e8;
    }
    .model-name {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f2b3d;
        text-align: center;
        background: #f0f4fa;
        display: inline-block;
        padding: 6px 16px;
        border-radius: 40px;
        margin-bottom: 12px;
    }
    .status-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #4b5565;
        margin-top: 8px;
    }
    .verdict {
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 12px 0;
    }
    .confidence {
        font-size: 0.85rem;
        color: #2c3e50;
        font-weight: 500;
    }
    .progress-bar-bg {
        background-color: #e2e8f0;
        border-radius: 20px;
        height: 8px;
        overflow: hidden;
        margin: 10px 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        height: 100%;
        border-radius: 20px;
        width: 0%;
    }
    /* Custom info alert */
    .custom-info {
        background: #f0f9ff;
        border-left: 4px solid #1e4a76;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #1e3a5f;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    /* Tombol */
    .stButton > button {
        background: linear-gradient(90deg, #1e4a76, #0f2b3d);
        color: white;
        font-weight: 600;
        border-radius: 40px;
        padding: 0.6rem 2rem;
        border: none;
        width: 100%;
        transition: 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: scale(1.01);
        background: linear-gradient(90deg, #0f2b3d, #1e4a76);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #5b6e8c;
        font-size: 0.75rem;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #cbd5e1;
    }
    hr {
        margin: 12px 0;
        border-color: #e2e8f0;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 12px;
        font-weight: 500;
    }
    /* Dataframe */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    /* Kolom input seimbang */
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
    }
    .stSelectbox > div > div {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Memuat aset model
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
    <div class="custom-info" style="background:#e0f2e0; border-left-color:#2e7d32;">
        <i class="bi bi-check-circle-fill" style="color:#2e7d32;"></i> Model dan preprocessor berhasil dimuat.
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.markdown(f"""
    <div class="custom-info" style="background:#fee2e2; border-left-color:#c62828;">
        <i class="bi bi-exclamation-triangle-fill" style="color:#c62828;"></i> Kesalahan: {e}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Informasi kelas
class_names = [c.upper() for c in le.classes_]

# Header
st.markdown("""
<div class='title'>
    <i class="bi bi-water" style="font-size: 2.2rem;"></i> Klasifikasi Multi-Kelas Kualitas Air
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 25px;">
    <span class='subtitle' style="display: inline-block;">
        <i class="bi bi-diagram-3"></i> <strong>LightGBM</strong> | <strong>CatBoost</strong> | <strong>HistGradientBoosting</strong>
    </span>
    <br>
    <div style="margin-top: 8px;">
        {''.join([f'<span class="class-badge">{cls}</span>' for cls in class_names])}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Input Form dengan proporsi seimbang ---
st.markdown("""
<div style="margin-top: 20px;">
    <h3><i class="bi bi-pencil-square"></i> 1. Parameter Masukan</h3>
</div>
""", unsafe_allow_html=True)

default_vals = {
    'ph': 7.20, 'do': 6.50, 'bod': 2.10, 'tc': 50.0,
    'tn': 1.20, 'tp': 0.05, 'ts': 150.0, 'turb': 4.50, 'temp': 25.0
}

st.markdown("""
<div class="custom-info">
    <i class="bi bi-info-circle"></i> Gunakan nilai default untuk uji cepat (kondisi air baik). Ubah sesuai data laboratorium.
</div>
""", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2, col3 = st.columns(3, gap="medium")
    user_inputs = {}
    # Numerik
    for i, col_name in enumerate(info['numeric_cols']):
        target_col = [col1, col2, col3][i % 3]
        with target_col:
            user_inputs[col_name] = st.number_input(
                label=f"{col_name.upper()}",
                value=default_vals.get(col_name, 0.0),
                format="%.2f",
                help=f"Masukkan nilai {col_name.upper()}"
            )
    st.markdown("---")
    land_use_options = ohe.categories_[0].tolist()
    user_inputs['macro_land_use'] = st.selectbox(
        label="Penggunaan Lahan (Macro Land Use)",
        options=land_use_options,
        index=0
    )
    submitted = st.form_submit_button("Jalankan Klasifikasi", use_container_width=True)

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
    with st.expander("Lihat detail nilai input", expanded=False):
        summary_df = pd.DataFrame([user_inputs]).T.reset_index()
        summary_df.columns = ['Parameter', 'Nilai']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <h3><i class="bi bi-diagram-3"></i> 3. Hasil Klasifikasi (3 Algoritma)</h3>
    """, unsafe_allow_html=True)

    # Tiga kolom untuk model
    cols = st.columns(3, gap="large")
    results = []

    for idx, (model_name, model) in enumerate(models.items()):
        pred_raw = model.predict(X_final)
        pred_idx = int(pred_raw[0]) if isinstance(pred_raw, np.ndarray) else int(pred_raw)
        label = le.inverse_transform([pred_idx])[0].upper()
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_final)[0]
            confidence = proba[pred_idx] * 100
        results.append({
            "Model": model_name,
            "Prediksi": label,
            "Confidence (%)": confidence if confidence is not None else None
        })
        # Warna berdasarkan label
        if label in ["EXCELLENT", "GOOD"]:
            color = "#15803d"
            icon = "bi bi-check-circle-fill"
            bg_light = "#e0f2e0"
        elif label in ["MODERATE", "FAIR"]:
            color = "#d97706"
            icon = "bi bi-exclamation-triangle-fill"
            bg_light = "#fff3e0"
        else:
            color = "#b91c1c"
            icon = "bi bi-x-circle-fill"
            bg_light = "#fee2e2"
        with cols[idx]:
            st.markdown(f"""
            <div class='card' style="border-top: 4px solid {color};">
                <div style="text-align:center;">
                    <span class='model-name' style="background:{bg_light}; color:{color};">{model_name}</span>
                </div>
                <div class='status-label'>Status Kualitas Air</div>
                <div class='verdict' style='color:{color};'>
                    <i class="{icon}" style="font-size: 1.8rem;"></i> {label}
                </div>
            """, unsafe_allow_html=True)
            if confidence is not None:
                st.markdown(f"""
                <div class='confidence'>
                    <i class="bi bi-bar-chart"></i> Keyakinan: {confidence:.2f}%
                </div>
                <div class='progress-bar-bg'>
                    <div class='progress-fill' style='width:{confidence:.2f}%; background:{color};'></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='confidence'>Probabilitas tidak tersedia</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Perbandingan & konsistensi
    st.markdown("""
    <h3><i class="bi bi-bar-chart-steps"></i> 4. Perbandingan & Konsistensi</h3>
    """, unsafe_allow_html=True)
    results_df = pd.DataFrame(results).set_index("Model")
    st.dataframe(results_df, use_container_width=True)
    if results_df['Prediksi'].nunique() == 1:
        st.markdown("""
        <div class="custom-info" style="background:#e0f2e0; border-left-color:#15803d;">
            <i class="bi bi-check2-all"></i> <strong>Konsisten:</strong> Semua model sepakat dengan prediksi yang sama.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="custom-info" style="background:#fff3cd; border-left-color:#d97706;">
            <i class="bi bi-exclamation-triangle"></i> <strong>Perbedaan prediksi:</strong> Model memberikan hasil berbeda. Perhatikan confidence untuk keputusan akhir.
        </div>
        """, unsafe_allow_html=True)

    # Probabilitas per kelas dari model dengan confidence tertinggi
    st.markdown("""
    <h3><i class="bi bi-graph-up"></i> 5. Detail Probabilitas per Kelas</h3>
    """, unsafe_allow_html=True)
    best = results_df.dropna(subset=['Confidence (%)']).iloc[0] if not results_df['Confidence (%)'].isna().all() else None
    if best is not None:
        best_model_name = best.name
        best_model = models[best_model_name]
        if hasattr(best_model, "predict_proba"):
            proba_all = best_model.predict_proba(X_final)[0] * 100
            proba_df = pd.DataFrame({
                "Kelas": [c.upper() for c in le.classes_],
                "Probabilitas (%)": proba_all
            }).sort_values("Probabilitas (%)", ascending=False)
            col_proba, col_chart = st.columns([1, 1])
            with col_proba:
                st.dataframe(proba_df, use_container_width=True, hide_index=True)
            with col_chart:
                st.bar_chart(proba_df.set_index("Kelas"), color="#1e4a76")
        else:
            st.info("Model tidak menyediakan probabilitas per kelas.")
    else:
        st.info("Tidak ada informasi probabilitas dari model.")

    with st.expander("Metadata vektor input (setelah preprocessing)"):
        st.code(str(X_final), language="python")
else:
    st.markdown("""
    <div class="custom-info">
        <i class="bi bi-info-circle"></i> Masukkan parameter atau gunakan default, lalu tekan <strong>Jalankan Klasifikasi</strong>.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <i class="bi bi-building"></i> Laboratorium Komputasi - Teknik Informatika - Universitas Halu Oleo<br>
    <i class="bi bi-diagram-3"></i> Algoritma: LightGBM, CatBoost, HistGradientBoosting | Klasifikasi Multi-Kelas
</div>
""", unsafe_allow_html=True)
