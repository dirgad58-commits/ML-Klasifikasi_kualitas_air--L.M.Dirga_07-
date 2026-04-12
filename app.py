import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Sistem Analisis Kualitas Air - Universitas Halu Oleo",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS KUSTOM (Tema Terang & Profesional)
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
    <style>
    /* Mengatur latar belakang utama menjadi abu-abu sangat muda (bukan hitam) */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Utama */
    .main-header {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 4px solid #1E3A8A;
    }
    
    .uho-title {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    
    /* Card Style untuk Hasil */
    .algo-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .algo-card:hover {
        transform: translateY(-5px);
    }
    
    /* Badge Status */
    .status-badge {
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
        margin-top: 10px;
    }
    
    .excellent { background-color: #DCFCE7; color: #166534; }
    .good { background-color: #DBEAFE; color: #1E40AF; }
    .moderate { background-color: #FEF3C7; color: #92400E; }
    .poor { background-color: #FEE2E2; color: #991B1B; }

    /* Input Styling */
    .stNumberInput label, .stSelectbox label {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. FUNGSI LOAD DATA
@st.cache_resource
def load_assets():
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

# Inisialisasi
try:
    info, scaler, ohe, le, models = load_assets()
except Exception as e:
    st.error(f"Sistem gagal memuat modul model di folder '/models'.")
    st.stop()

# 4. HEADER UI
st.markdown("""
    <div class="main-header">
        <h1 class="uho-title">WATER QUALITY CLASSIFICATION SYSTEM</h1>
        <p style="color: #64748B; font-size: 1.1rem;">Implementasi Algoritma Gradient Boosting untuk Pemantauan Mutu Air</p>
    </div>
""", unsafe_allow_html=True)

# 5. INPUT SECTION (Tengah Halaman)
tab1, tab2, tab3 = st.tabs([
    "📝 Input Parameter Laboratorium", 
    "📊 Hasil Klasifikasi Komparatif", 
    "🔬 Detail Vektor & Probabilitas"
])

with tab1:
    st.markdown("### Masukkan Data Hasil Pengujian")
    st.caption("Gunakan nilai default di bawah ini untuk simulasi cepat (Kondisi Air Normal).")
    
    with st.form("input_form"):
        # Penentuan nilai default untuk uji cepat
        default_vals = {
            'ph': 7.20, 'do': 6.50, 'bod': 2.10, 'tc': 50.0, 
            'tn': 1.20, 'tp': 0.05, 'ts': 150.0, 'turb': 4.50, 'temp': 25.0
        }
        
        # Grid input 3 kolom
        col1, col2, col3 = st.columns(3)
        user_inputs = {}
        
        for i, col_name in enumerate(info['numeric_cols']):
            target_col = [col1, col2, col3][i % 3]
            user_inputs[col_name] = target_col.number_input(
                f"{col_name.upper()}", 
                value=default_vals.get(col_name, 0.0), 
                format="%.2f",
                help=f"Parameter konsentrasi {col_name}"
            )
            
        st.markdown("---")
        land_use_options = ohe.categories_[0].tolist()
        user_inputs['macro_land_use'] = st.selectbox(
            "Macro Land Use (Penggunaan Lahan Sekitar)", 
            land_use_options,
            index=0
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("ANALISIS KUALITAS AIR")

# 6. PROCESSING & OUTPUT
if submit:
    # Preprocessing
    input_df = pd.DataFrame([user_inputs])
    X_num = input_df[info['numeric_cols']]
    X_cat = input_df[['macro_land_use']]
    
    X_num_scaled = scaler.transform(X_num)
    X_cat_encoded = ohe.transform(X_cat)
    X_final = np.hstack([X_num_scaled, X_cat_encoded])

    # Tampilkan Hasil di Tab 2
    with tab2:
        st.markdown("### Perbandingan Performa Algoritma")
        res_cols = st.columns(3)
        
        all_results = []
        
        for idx, (name, model) in enumerate(models.items()):
            # Prediksi
            raw_pred = model.predict(X_final)
            if isinstance(raw_pred, np.ndarray):
                pred_idx = int(raw_pred.flatten()[0]) if raw_pred.ndim > 1 else int(raw_pred[0])
            else:
                pred_idx = int(raw_pred)
            
            label = le.inverse_transform([pred_idx])[0].upper()
            
            # Tentukan warna badge
            badge_class = "moderate"
            if "EXCELENT" in label or "GOOD" in label: badge_class = "excellent"
            elif "BAD" in label: badge_class = "poor"
            
            # Hitung Probabilitas
            prob = model.predict_proba(X_final)[0][pred_idx] * 100

            with res_cols[idx]:
                st.markdown(f"""
                    <div class="algo-card">
                        <p style="color: #64748B; font-weight: bold; margin-bottom: 5px;">{name}</p>
                        <hr style="margin: 10px 0;">
                        <p style="font-size: 0.8rem; margin-bottom: 0;">Status Klasifikasi:</p>
                        <div class="status-badge {badge_class}">{label}</div>
                        <p style="margin-top: 15px; font-size: 0.9rem; color: #475569;">
                            Confidence: <strong>{prob:.2f}%</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
            all_results.append({"Model": name, "Status": label, "Confidence": prob})

    # Detail di Tab 3
    with tab3:
        st.markdown("### Distribusi Probabilitas (Model Terbaik: CatBoost)")
        best_model = models["CatBoost"]
        probs = best_model.predict_proba(X_final)[0]
        classes = [c.upper() for c in le.classes_]
        
        fig = go.Figure([go.Bar(x=classes, y=probs*100, marker_color='#1E3A8A')])
        fig.update_layout(
            title="Probabilitas Tiap Kelas (%)",
            template="plotly_white",
            height=400,
            yaxis_title="Persentase (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Lihat Metadata Vektor Input"):
            st.code(str(X_final))

else:
    with tab2:
        st.info("Silakan lengkapi data pada tab 'Input Parameter' dan klik tombol Analisis.")

# 7. FOOTER
st.markdown(f"""
    <div style="margin-top: 50px; padding: 20px; text-align: center; color: #94A3B8; border-top: 1px solid #E2E8F0;">
        <p>Aplikasi Lab Komputasi | Informatika Universitas Halu Oleo | 2024</p>
        <small>Target Akurasi Model: {info['best_accuracy']:.2%}</small>
    </div>
""", unsafe_allow_html=True)
