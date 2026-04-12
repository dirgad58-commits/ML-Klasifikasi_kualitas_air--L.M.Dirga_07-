import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Multi-Class Water Quality | 3 Algoritma",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background:
            radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent 30%),
            radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 25%),
            linear-gradient(135deg, #f6f9ff 0%, #eef4fb 100%);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,64,175,0.92));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 28px;
        padding: 28px 28px 22px 28px;
        box-shadow: 0 18px 40px rgba(15,23,42,0.20);
        color: white;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        margin-top: 0.6rem;
        color: rgba(255,255,255,0.85);
        font-size: 0.98rem;
    }

    .pill {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        color: #fff;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.78rem;
        margin: 0.25rem 0.22rem 0 0;
        backdrop-filter: blur(8px);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.2rem 0 0.85rem 0;
        padding-left: 0.2rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(148,163,184,0.22);
        backdrop-filter: blur(12px);
        border-radius: 22px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: 0 10px 24px rgba(15,23,42,0.06);
        height: 100%;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 10px 20px rgba(15,23,42,0.05);
        height: 100%;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
    }

    .metric-note {
        font-size: 0.84rem;
        color: #475569;
        margin-top: 0.25rem;
    }

    .model-card {
        border-radius: 22px;
        padding: 1rem;
        background: white;
        border: 1px solid rgba(148,163,184,0.20);
        box-shadow: 0 12px 26px rgba(15,23,42,0.06);
        height: 100%;
        transition: transform 0.20s ease, box-shadow 0.20s ease;
    }

    .model-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 30px rgba(15,23,42,0.10);
    }

    .model-badge {
        display: inline-block;
        padding: 0.38rem 0.8rem;
        border-radius: 999px;
        font-size: 0.80rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .verdict {
        font-size: 1.7rem;
        font-weight: 900;
        margin: 0.35rem 0 0.5rem 0;
        line-height: 1.1;
    }

    .subtext {
        color: #475569;
        font-size: 0.88rem;
        margin-top: 0.2rem;
    }

    .progress-wrap {
        background: #e2e8f0;
        height: 10px;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 0.6rem;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
    }

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        color: #1e3a8a;
        box-shadow: 0 6px 16px rgba(37,99,235,0.06);
    }

    .success-box {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        color: #166534;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        color: #92400e;
    }

    .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        color: #991b1b;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148,163,184,0.25);
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        padding: 0.72rem 1rem;
        background: linear-gradient(135deg, #1d4ed8, #0f172a);
        color: white;
        box-shadow: 0 10px 20px rgba(15,23,42,0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(15,23,42,0.20);
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: 12px;
    }

    .stExpander {
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 16px;
        background: rgba(255,255,255,0.65);
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px;
        padding: 0.9rem;
        box-shadow: 0 8px 20px rgba(15,23,42,0.04);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def get_preset_defaults(preset_name: str):
    presets = {
        "Air Baik": {
            "ph": 7.20, "do": 6.80, "bod": 1.80, "tc": 35.0,
            "tn": 0.90, "tp": 0.04, "ts": 120.0, "turb": 3.20, "temp": 25.0
        },
        "Air Sedang": {
            "ph": 6.90, "do": 5.50, "bod": 3.50, "tc": 120.0,
            "tn": 1.80, "tp": 0.10, "ts": 220.0, "turb": 8.50, "temp": 27.0
        },
        "Air Buruk": {
            "ph": 5.80, "do": 3.20, "bod": 7.80, "tc": 420.0,
            "tn": 4.20, "tp": 0.28, "ts": 680.0, "turb": 25.0, "temp": 31.0
        }
    }
    return presets.get(preset_name, presets["Air Baik"])

def safe_pred_to_int(pred):
    arr = np.array(pred).ravel()
    try:
        return int(arr[0])
    except Exception:
        return int(pred)

def class_color(label_upper: str):
    if label_upper in ["EXCELLENT", "GOOD"]:
        return {
            "color": "#15803d",
            "bg": "#dcfce7",
            "icon": "bi bi-check-circle-fill"
        }
    elif label_upper in ["MODERATE", "FAIR"]:
        return {
            "color": "#d97706",
            "bg": "#fef3c7",
            "icon": "bi bi-exclamation-triangle-fill"
        }
    else:
        return {
            "color": "#dc2626",
            "bg": "#fee2e2",
            "icon": "bi bi-x-circle-fill"
        }

def render_model_card(model_name, label, confidence):
    style = class_color(label)
    conf_text = f"{confidence:.2f}%" if confidence is not None else "Tidak tersedia"
    progress_width = confidence if confidence is not None else 0
    progress_color = style["color"]

    st.markdown(f"""
    <div class="model-card" style="border-top: 5px solid {style['color']};">
        <div class="model-badge" style="background:{style['bg']}; color:{style['color']};">
            {model_name}
        </div>
        <div class="subtext">Status kualitas air</div>
        <div class="verdict" style="color:{style['color']};">
            <i class="{style['icon']}"></i> {label}
        </div>
        <div class="subtext"><i class="bi bi-bar-chart-fill"></i> Keyakinan: {conf_text}</div>
        <div class="progress-wrap">
            <div class="progress-fill" style="width:{progress_width}%; background:{progress_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# LOAD ASSETS
# =========================================================
@st.cache_resource
def load_all_assets():
    path = "models"
    with open(os.path.join(path, "feature_info.json"), "r") as f:
        info = json.load(f)

    scaler = pickle.load(open(os.path.join(path, "scaler.pkl"), "rb"))
    ohe = pickle.load(open(os.path.join(path, "ohe.pkl"), "rb"))
    le = pickle.load(open(os.path.join(path, "label_encoder.pkl"), "rb"))

    models = {
        "LightGBM": pickle.load(open(os.path.join(path, "lightgbm.pkl"), "rb")),
        "CatBoost": pickle.load(open(os.path.join(path, "catboost.pkl"), "rb")),
        "HistGradientBoosting": pickle.load(open(os.path.join(path, "histgradientboosting.pkl"), "rb"))
    }
    return info, scaler, ohe, le, models

try:
    info, scaler, ohe, le, models = load_all_assets()
except Exception as e:
    st.markdown(
        f'<div class="error-box"><i class="bi bi-exclamation-triangle-fill"></i> Gagal memuat model/preprocessor: {e}</div>',
        unsafe_allow_html=True
    )
    st.stop()

class_names = [c.upper() for c in le.classes_]
land_use_options = list(ohe.categories_[0]) if hasattr(ohe, "categories_") else ["UNKNOWN"]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 💧 Kontrol")
preset = st.sidebar.selectbox(
    "Preset input cepat",
    ["Air Baik", "Air Sedang", "Air Buruk", "Manual"],
    index=0
)

show_technical = st.sidebar.toggle("Tampilkan detail teknis", value=True)
show_raw_vector = st.sidebar.toggle("Tampilkan vektor hasil preprocessing", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### Algoritma aktif")
st.sidebar.markdown(
    "• LightGBM\n\n• CatBoost\n\n• HistGradientBoosting"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Kelas output")
for cls in class_names:
    st.sidebar.markdown(f"- {cls}")

default_vals = get_preset_defaults(preset if preset != "Manual" else "Air Baik")

# =========================================================
# HEADER
# =========================================================
st.markdown(f"""
<div class="hero">
    <div class="hero-title">
        <i class="bi bi-droplet-half"></i> Klasifikasi Multi-Kelas Kualitas Air
    </div>
    <div class="hero-subtitle">
        Antarmuka interaktif untuk membandingkan prediksi <b>LightGBM</b>, <b>CatBoost</b>, dan <b>HistGradientBoosting</b>.
    </div>
    <div style="margin-top: 0.8rem;">
        {''.join([f'<span class="pill">{cls}</span>' for cls in class_names])}
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab_input, tab_hasil, tab_info = st.tabs(["🧪 Input & Prediksi", "📊 Hasil Analisis", "ℹ️ Informasi"])

with tab_input:
    st.markdown('<div class="section-title">Parameter Masukan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box"><i class="bi bi-info-circle-fill"></i> Gunakan preset untuk uji cepat atau ubah nilai secara manual sesuai data laboratorium.</div>',
        unsafe_allow_html=True
    )

    with st.form("input_form"):
        c1, c2, c3 = st.columns(3, gap="medium")
        user_inputs = {}

        numeric_cols = info["numeric_cols"]

        for i, col_name in enumerate(numeric_cols):
            target_col = [c1, c2, c3][i % 3]
            with target_col:
                user_inputs[col_name] = st.number_input(
                    label=f"{col_name.upper()}",
                    value=float(default_vals.get(col_name, 0.0)),
                    format="%.4f",
                    help=f"Masukkan nilai {col_name.upper()}",
                    key=f"input_{col_name}"
                )

        st.markdown("#### Parameter kategorikal")
        user_inputs["macro_land_use"] = st.selectbox(
            label="Macro Land Use",
            options=land_use_options,
            index=0,
            help="Pilih jenis penggunaan lahan yang paling sesuai"
        )

        submitted = st.form_submit_button("🚀 Jalankan Klasifikasi")

with tab_hasil:
    if submitted:
        # =========================================================
        # PREPROCESSING
        # =========================================================
        input_df = pd.DataFrame([user_inputs])
        X_num = input_df[info["numeric_cols"]]
        X_cat = input_df[["macro_land_use"]]

        X_num_scaled = scaler.transform(X_num)
        X_cat_encoded = ohe.transform(X_cat)
        X_final = np.hstack([X_num_scaled, X_cat_encoded])

        # =========================================================
        # PREDICTION
        # =========================================================
        results = []
        pred_labels = []

        for model_name, model in models.items():
            pred_raw = model.predict(X_final)
            pred_idx = safe_pred_to_int(pred_raw)
            label = le.inverse_transform([pred_idx])[0].upper()
            pred_labels.append(label)

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_final)[0]
                confidence = float(proba[pred_idx] * 100)

            results.append({
                "Model": model_name,
                "Prediksi": label,
                "Confidence (%)": confidence
            })

        results_df = pd.DataFrame(results).set_index("Model")
        majority_vote = results_df["Prediksi"].mode().iloc[0]
        agreement_count = int(results_df["Prediksi"].eq(majority_vote).sum())
        agreement_pct = agreement_count / len(results_df) * 100

        best_row = None
        if results_df["Confidence (%)"].notna().any():
            best_row = results_df.dropna(subset=["Confidence (%)"]).sort_values("Confidence (%)", ascending=False).iloc[0]

        # =========================================================
        # RINGKASAN METRIK
        # =========================================================
        st.markdown('<div class="section-title">Ringkasan Cepat</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4, gap="medium")

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediksi Dominan</div>
                <div class="metric-value">{majority_vote}</div>
                <div class="metric-note">Hasil voting dari 3 model</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tingkat Kesepakatan</div>
                <div class="metric-value">{agreement_pct:.0f}%</div>
                <div class="metric-note">{agreement_count} dari {len(results_df)} model setuju</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            top_conf = f"{best_row['Confidence (%)']:.2f}%" if best_row is not None else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence Tertinggi</div>
                <div class="metric-value">{top_conf}</div>
                <div class="metric-note">{best_row.name if best_row is not None else 'Tidak tersedia'}</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Macro Land Use</div>
                <div class="metric-value" style="font-size:1.1rem;">{user_inputs['macro_land_use']}</div>
                <div class="metric-note">Fitur kategorikal terpilih</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Hasil per Model")
        cA, cB, cC = st.columns(3, gap="large")

        model_order = list(models.keys())
        for idx, model_name in enumerate(model_order):
            row = results_df.loc[model_name]
            label = row["Prediksi"]
            conf = row["Confidence (%)"]
            with [cA, cB, cC][idx]:
                render_model_card(model_name, label, conf)

        st.markdown("### Perbandingan Prediksi")
        st.dataframe(results_df, use_container_width=True)

        if results_df["Prediksi"].nunique() == 1:
            st.markdown(
                '<div class="success-box"><i class="bi bi-check2-circle"></i> Semua model memberikan prediksi yang sama. Ini menunjukkan konsistensi yang tinggi.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="warning-box"><i class="bi bi-exclamation-triangle-fill"></i> Model memberikan prediksi yang berbeda. Gunakan confidence tertinggi dan pola probabilitas untuk interpretasi akhir.</div>',
                unsafe_allow_html=True
            )

        # =========================================================
        # PROBABILITAS TERBAIK
        # =========================================================
        st.markdown("### Probabilitas per Kelas")
        if best_row is not None:
            best_model_name = best_row.name
            best_model = models[best_model_name]

            if hasattr(best_model, "predict_proba"):
                proba_all = best_model.predict_proba(X_final)[0] * 100
                proba_df = pd.DataFrame({
                    "Kelas": [c.upper() for c in le.classes_],
                    "Probabilitas (%)": proba_all
                }).sort_values("Probabilitas (%)", ascending=False)

                left, right = st.columns([1, 1], gap="large")
                with left:
                    st.dataframe(proba_df, use_container_width=True, hide_index=True)

                with right:
                    fig = px.bar(
                        proba_df,
                        x="Probabilitas (%)",
                        y="Kelas",
                        orientation="h",
                        text=proba_df["Probabilitas (%)"].map(lambda x: f"{x:.1f}%"),
                        title=f"Distribusi Probabilitas - {best_model_name}",
                        template="plotly_white"
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        height=420,
                        margin=dict(l=10, r=10, t=50, b=10),
                        xaxis_title="Probabilitas (%)",
                        yaxis_title=""
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model terbaik tidak menyediakan `predict_proba()`.")
        else:
            st.info("Tidak ada informasi probabilitas yang tersedia.")

        # =========================================================
        # DETAIL TEKNIS
        # =========================================================
        if show_technical:
            with st.expander("Detail teknis preprocessing dan input"):
                st.markdown("#### Ringkasan input asli")
                summary_df = pd.DataFrame([user_inputs]).T.reset_index()
                summary_df.columns = ["Parameter", "Nilai"]
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                st.markdown("#### Vektor hasil preprocessing")
                if show_raw_vector:
                    st.code(np.array2string(X_final, precision=4, suppress_small=True), language="python")
                else:
                    st.caption("Aktifkan opsi **Tampilkan vektor hasil preprocessing** di sidebar untuk melihat nilainya.")

                st.markdown("#### Informasi fitur numerik")
                st.write(info["numeric_cols"])

    else:
        st.markdown(
            '<div class="info-box"><i class="bi bi-info-circle-fill"></i> Isi parameter pada tab <b>Input & Prediksi</b>, lalu tekan <b>Jalankan Klasifikasi</b>.</div>',
            unsafe_allow_html=True
        )

with tab_info:
    st.markdown("### Tentang Sistem")
    st.markdown(
        """
        Aplikasi ini membandingkan tiga model machine learning untuk klasifikasi kualitas air multi-kelas.
        Alur umumnya:
        1. Input parameter kualitas air
        2. Preprocessing numerik dan kategorikal
        3. Prediksi oleh tiga model
        4. Perbandingan hasil dan confidence
        5. Visualisasi probabilitas kelas
        """
    )

    st.markdown("### Model yang digunakan")
    model_df = pd.DataFrame({
        "Model": list(models.keys()),
        "Keterangan": [
            "Gradient boosting yang cepat dan kuat untuk data tabular",
            "Boosting yang stabil dan cocok untuk fitur campuran",
            "Gradient boosting berbasis histogram untuk efisiensi tinggi"
        ]
    })
    st.dataframe(model_df, use_container_width=True, hide_index=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    <i class="bi bi-building"></i> Laboratorium Komputasi - Teknik Informatika - Universitas Halu Oleo<br>
    <i class="bi bi-diagram-3"></i> LightGBM | CatBoost | HistGradientBoosting
</div>
""", unsafe_allow_html=True)
