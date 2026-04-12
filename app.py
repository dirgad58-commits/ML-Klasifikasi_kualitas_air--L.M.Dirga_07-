import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Water Quality Intelligence | Multi-Class ML",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Premium dengan Bootstrap Icons ---
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    * {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background: linear-gradient(145deg, #f4f9ff 0%, #e9f0fa 100%);
    }
    .hero {
        background: linear-gradient(105deg, #0b2b44 0%, #1b4a6e 100%);
        border-radius: 32px;
        padding: 2rem 2rem 1.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.15);
        color: white;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    .hero-badge {
        background: rgba(255,255,240,0.2);
        backdrop-filter: blur(8px);
        border-radius: 60px;
        padding: 0.25rem 1rem;
        display: inline-block;
        font-size: 0.8rem;
        margin-top: 0.8rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3);
        margin-bottom: 1.5rem;
    }
    .model-card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(4px);
        border-radius: 24px;
        padding: 1.2rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
        border: 1px solid rgba(255,255,255,0.5);
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 30px -12px rgba(0,0,0,0.15);
        background: rgba(255,255,255,0.95);
    }
    .model-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0c4a6e;
        background: #eef2ff;
        display: inline-block;
        padding: 4px 16px;
        border-radius: 40px;
        margin-bottom: 12px;
    }
    .verdict {
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin: 10px 0;
    }
    .progress-bar-bg {
        background-color: #e2e8f0;
        border-radius: 30px;
        height: 8px;
        overflow: hidden;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 30px;
        width: 0%;
    }
    .info-box {
        background: #fefce8;
        border-left: 5px solid #eab308;
        padding: 0.8rem 1rem;
        border-radius: 16px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        color: white;
        font-weight: 600;
        border-radius: 44px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 18px rgba(37,99,235,0.3);
    }
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #cbd5e1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255,255,255,0.5);
        border-radius: 40px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 32px;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1><i class="bi bi-water"></i> Water Quality Intelligence</h1>
            <p>Klasifikasi multi-kelas kualitas air dengan 3 algoritma gradient boosting<br>LightGBM · CatBoost · HistGradientBoosting</p>
            <div class="hero-badge"><i class="bi bi-diagram-3"></i> Multi-Class Classification</div>
        </div>
        <div style="font-size: 3rem; opacity: 0.8;">
            <i class="bi bi-cpu"></i>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Load Model ---
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
    <div class="info-box" style="background:#dcfce7; border-left-color:#16a34a;">
        <i class="bi bi-check-circle-fill" style="color:#16a34a;"></i> Model dan preprocessor berhasil dimuat.
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.markdown(f"""
    <div class="info-box" style="background:#fee2e2; border-left-color:#dc2626;">
        <i class="bi bi-exclamation-triangle-fill"></i> Kesalahan: {e}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

class_names = [c.upper() for c in le.classes_]

# --- PRESET LENGKAP (termasuk macro_land_use) ---
presets = {
    "Custom (Manual Input)": None,
    "Sampel A: Excellent Condition": {
        'ph': 7.2, 'do': 8.5, 'bod': 0.5, 'tc': 20.0, 'tn': 0.2, 'tp': 0.01, 'ts': 45.0, 'turb': 0.8, 'temp': 21.0, 'macro_land_use': 'agriculture'
    },
    "Sampel B: Good/Normal Condition": {
        'ph': 7.4, 'do': 6.8, 'bod': 1.8, 'tc': 150.0, 'tn': 0.9, 'tp': 0.06, 'ts': 110.0, 'turb': 4.2, 'temp': 24.5, 'macro_land_use': 'agriculture'
    },
    "Sampel C: Moderate/Medium Pollution": {
        'ph': 7.9, 'do': 4.2, 'bod': 4.5, 'tc': 1200.0, 'tn': 2.8, 'tp': 0.25, 'ts': 320.0, 'turb': 15.0, 'temp': 27.0, 'macro_land_use': 'industrialization'
    },
    "Sampel D: Bad/Heavy Pollution": {
        'ph': 6.2, 'do': 2.1, 'bod': 14.0, 'tc': 25000.0, 'tn': 7.5, 'tp': 1.2, 'ts': 650.0, 'turb': 45.0, 'temp': 29.0, 'macro_land_use': 'industrial'
    },
    "Sampel E: Very Bad/Toxic Condition": {
        'ph': 5.2, 'do': 0.4, 'bod': 35.0, 'tc': 150000.0, 'tn': 18.0, 'tp': 4.5, 'ts': 1450.0, 'turb': 110.0, 'temp': 33.0, 'macro_land_use': 'industrial'
    }
}

# --- Input Form dengan Glass Card ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### <i class='bi bi-sliders2'></i> Parameter Masukan", unsafe_allow_html=True)

col_preset, _ = st.columns([1, 2])
with col_preset:
    selected_preset = st.selectbox(
        "Preset Cepat",
        options=list(presets.keys()),
        index=1  # default Sampel B
    )

# Ambil nilai default dari preset (jika bukan Custom)
if selected_preset != "Custom (Manual Input)":
    default_vals = presets[selected_preset].copy()
else:
    # Nilai default untuk manual (kondisi normal)
    default_vals = {
        'ph': 7.2, 'do': 6.5, 'bod': 2.0, 'tc': 100.0, 'tn': 1.0, 'tp': 0.05, 'ts': 150.0, 'turb': 5.0, 'temp': 25.0, 'macro_land_use': 'agriculture'
    }

with st.form("input_form"):
    col1, col2, col3 = st.columns(3, gap="medium")
    user_inputs = {}
    # Kolom numerik (semua kecuali macro_land_use)
    numeric_cols = [c for c in info['numeric_cols'] if c != 'macro_land_use']
    for i, col_name in enumerate(numeric_cols):
        target_col = [col1, col2, col3][i % 3]
        with target_col:
            user_inputs[col_name] = st.number_input(
                label=f"{col_name.upper()}",
                value=float(default_vals.get(col_name, 0.0)),
                format="%.2f",
                help=f"Masukkan nilai {col_name.upper()}"
            )
    st.markdown("---")
    # Input macro_land_use
    land_use_options = ohe.categories_[0].tolist()
    user_inputs['macro_land_use'] = st.selectbox(
        label="Penggunaan Lahan (Macro Land Use)",
        options=land_use_options,
        index=land_use_options.index(default_vals['macro_land_use']) if default_vals['macro_land_use'] in land_use_options else 0
    )
    submitted = st.form_submit_button("Jalankan Klasifikasi", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Fungsi prediksi yang robust ---
def predict_all_models(models, X_final, le):
    results = []
    for name, model in models.items():
        pred_raw = model.predict(X_final)
        # Handle berbagai bentuk output
        if isinstance(pred_raw, np.ndarray):
            pred_flat = pred_raw.flatten()
            pred_idx = int(pred_flat[0]) if len(pred_flat) > 0 else int(pred_raw)
        elif isinstance(pred_raw, (list, tuple)):
            pred_idx = int(pred_raw[0])
        else:
            pred_idx = int(pred_raw)
        
        label = le.inverse_transform([pred_idx])[0].upper()
        confidence = None
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_final)[0]
            confidence = proba[pred_idx] * 100
        results.append({
            "Model": name,
            "Prediksi": label,
            "Confidence (%)": confidence if confidence is not None else None,
            "probabilities": proba
        })
    return results

if submitted:
    # Preprocessing
    input_df = pd.DataFrame([user_inputs])
    X_num = input_df[info['numeric_cols']]
    X_cat = input_df[['macro_land_use']]
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)
    X_final = np.hstack([X_num_scaled, X_cat_encoded])

    results = predict_all_models(models, X_final, le)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'probabilities'} for r in results]).set_index("Model")

    # --- Tabs untuk tampilan rapi ---
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Class Probabilities", "Input & Metadata"])

    with tab1:
        st.markdown("### Hasil Klasifikasi 3 Algoritma")
        cols = st.columns(3, gap="large")
        for idx, res in enumerate(results):
            model_name = res["Model"]
            label = res["Prediksi"]
            confidence = res["Confidence (%)"]
            # Warna berdasarkan label (sesuaikan dengan kelas yang ada)
            if label in ["EXCELLENT", "GOOD"]:
                color = "#15803d"
                icon = "bi bi-check-circle-fill"
                bg_light = "#dcfce7"
            elif label in ["MODERATE", "MEDIUM", "FAIR"]:
                color = "#d97706"
                icon = "bi bi-exclamation-triangle-fill"
                bg_light = "#fef3c7"
            else:
                color = "#b91c1c"
                icon = "bi bi-x-circle-fill"
                bg_light = "#fee2e2"
            with cols[idx]:
                st.markdown(f"""
                <div class='model-card' style="border-top: 4px solid {color};">
                    <div style="text-align:center;">
                        <span class='model-name' style="background:{bg_light}; color:{color};">{model_name}</span>
                    </div>
                    <div class='status-label'>Status Kualitas Air</div>
                    <div class='verdict' style='color:{color};'>
                        <i class="{icon}" style="font-size: 1.5rem;"></i> {label}
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

        # Konsistensi
        st.markdown("### Konsistensi Prediksi")
        if results_df['Prediksi'].nunique() == 1:
            st.markdown("""
            <div class="info-box" style="background:#dcfce7; border-left-color:#15803d;">
                <i class="bi bi-check2-all"></i> <strong>Konsisten:</strong> Semua model sepakat dengan prediksi yang sama.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="background:#fef3c7; border-left-color:#d97706;">
                <i class="bi bi-exclamation-triangle"></i> <strong>Perbedaan prediksi:</strong> Model memberikan hasil berbeda. Perhatikan confidence untuk keputusan akhir.
            </div>
            """, unsafe_allow_html=True)

        st.dataframe(results_df, use_container_width=True)

    with tab2:
        st.markdown("### Probabilitas per Kelas dari Model dengan Confidence Tertinggi")
        # Pilih model dengan confidence tertinggi
        best_row = results_df.dropna(subset=['Confidence (%)']).iloc[0] if not results_df['Confidence (%)'].isna().all() else None
        if best_row is not None:
            best_model_name = best_row.name
            best_model = models[best_model_name]
            if hasattr(best_model, "predict_proba"):
                proba_all = best_model.predict_proba(X_final)[0] * 100
                proba_df = pd.DataFrame({
                    "Kelas": [c.upper() for c in le.classes_],
                    "Probabilitas (%)": proba_all
                }).sort_values("Probabilitas (%)", ascending=False)
                st.markdown(f"<p><i class='bi bi-star-fill'></i> Model terbaik: <strong>{best_model_name}</strong> (confidence tertinggi)</p>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(proba_df, use_container_width=True, hide_index=True)
                with col2:
                    fig = px.bar(proba_df, x="Kelas", y="Probabilitas (%)", color="Probabilitas (%)",
                                 color_continuous_scale="Blues", title="Distribusi Probabilitas")
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model tidak menyediakan probabilitas per kelas.")
        else:
            st.info("Tidak ada informasi probabilitas dari model.")

    with tab3:
        st.markdown("### Ringkasan Parameter Input")
        summary_df = pd.DataFrame([user_inputs]).T.reset_index()
        summary_df.columns = ['Parameter', 'Nilai']
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with st.expander("Metadata vektor input (setelah normalisasi & encoding)"):
            st.code(str(X_final), language="python")

else:
    st.markdown("""
    <div class="info-box">
        <i class="bi bi-info-circle"></i> Silakan masukkan parameter atau gunakan preset cepat, lalu tekan <strong>Jalankan Klasifikasi</strong>.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <i class="bi bi-building"></i> Laboratorium Komputasi - Teknik Informatika - Universitas Halu Oleo<br>
    <i class="bi bi-diagram-3"></i> Algoritma: LightGBM, CatBoost, HistGradientBoosting | Klasifikasi Multi-Kelas
</div>
""", unsafe_allow_html=True)
