import warnings
import matplotlib
matplotlib.use("Agg")

import streamlit as st

warnings.filterwarnings("ignore")


# ========= PAGE CONFIG & GLOBAL STYLE =========
st.set_page_config(
    page_title="Rice Land Suitability Dashboard",
    layout="wide",
    page_icon="🌾",
)

# (opsional) CSS global kamu taruh di sini
st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        font-size: 15px;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid #eee;
    }
    .badge {
        font-size: 11px;
        background: #e6f4ea;
        color: #137333;
        border-radius: 999px;
        padding: 0.15rem 0.6rem;
        margin-left: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========= HEADER / HOME CONTENT =========
st.markdown(
    """
    <div class="main-title">
        🌾 Rice Land Suitability Dashboard
        <span class="badge">RF · MLP · Hybrid</span>
    </div>
    <div class="main-subtitle">
        Dashboard ini membantu mengklasifikasikan kesesuaian lahan sawah untuk padi
        berdasarkan pH tanah, suhu tanah, dan kelembaban udara.
        Gunakan menu di sidebar untuk berpindah antar halaman.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
    ### Alur Utama Dashboard

    1. **Pendahuluan & Load Data**  
       - Upload file CSV berisi data IoT (suhu tanah, kelembaban udara, pH tanah).  
       - Cek format dataset dan validasi awal.

    2. **Deskriptif Data**  
       - Lihat statistik dasar & distribusi fitur.

    3. **Distribusi Label & Imputasi**  
       - Lihat hasil rule-based labeling dan penanganan missing value.

    4. **Split Data, SMOTE & Normalisasi**  
       - Proses pembagian train–test, SMOTE, dan scaling.

    5. **Model & Evaluasi (RF, MLP, Hybrid)**  
       - Bandingkan performa model.

    6. **Interpretabilitas (SHAP, LIME, FI)**  
       - Jelaskan kontribusi fitur terhadap prediksi.

    7. **Prediksi Interaktif**  
       - Input nilai manual dan lihat hasil prediksi model.
    """
)
st.markdown('</div>', unsafe_allow_html=True)

# ========= OPTIONAL: LINK ANTAR PAGE (kalau Streamlit baru) =========
# Kalau pakai Streamlit 1.20+ bisa pakai st.page_link:
try:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Navigasi Cepat", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.page_link("pages/1_Pendahuluan_Load_Data.py", label="1. Pendahuluan & Load Data", icon="📂")
        # nanti tinggal tambah:
        # st.page_link("pages/2_Deskriptif_Data.py", label="2. Deskriptif Data", icon="📊")

    st.markdown(
        """
        Atau gunakan sidebar di kiri (default Streamlit) untuk memilih halaman.
        """,
    )
    st.markdown('</div>', unsafe_allow_html=True)
except Exception:
    # Kalau Streamlit versi lama dan belum support st.page_link, abaikan saja.
    pass
