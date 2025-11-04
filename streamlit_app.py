import os, sys, datetime as dt
import streamlit as st

# === INIT DJANGO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iprs.settings")
import django
django.setup()

st.set_page_config(page_title="IPRS", layout="wide")

from page.analisis import get_analisis_page
from page.unggah_dataset import get_unggah_dataset_page
# ====== STYLE ======
st.markdown(
    """
        <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 26px !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 24px !important;
            font-weight: 600 !important;
        }
        </style>
    """, 
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("FPRS")
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

sidebar_map = {
    "Beranda": "Beranda",
    "Unggah Dataset": "Unggah Dataset",
    "Analisis": "Analisis",
    "Tentang": "Tentang",
}

clicked_page = None
for label, target in sidebar_map.items():
    if st.sidebar.button(label, key=f"nav_{label}"):
        clicked_page = target

# hanya ubah page jika tombol berbeda
if clicked_page and st.session_state.page != clicked_page:
    st.session_state.page = clicked_page
    # st.rerun()

# for label, target in sidebar_map.items():
#     if st.sidebar.button(label):
#         st.session_state.page = target

page = st.session_state.page

# =========================
# HALAMAN BERANDA
# =========================
if page == "Beranda":
    st.title("Sistem Rekomendasi Pemain Sepak Bola")
    st.markdown(
        """
        ---
        ### Selamat datang di Sistem Rekomendasi Pemain Sepak Bola ⚽  
        Gunakan sistem ini untuk menemukan pemain rekomendasi yang mirip dengan pemain acuan yang anda pilih berdasarkan statistik performa
        """
    )
    st.markdown(
        """
        ---
        ### Fitur yang terdapat di website ini
        """
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📂 Unggah Dataset")
        st.write("Unggah data liga dan statistik pemain untuk memulai analisis.")
    with col2:
        st.subheader("📊 Analisis")
        st.write("Lihat hasil clustering dan temukan pemain rekomendasi.")
    with col3:
        st.subheader("👨‍💻 Tentang")
        st.write("Lihat lebih lanjut tentang pembuat dan website.")

    st.markdown("---")

    if st.button("Mulai Analisis Sekarang 🚀"):
        st.session_state.page = "Analisis"        

# =============================
# HALAMAN UNGGAH DATASET
# =============================
elif page == "Unggah Dataset":
    get_unggah_dataset_page(st)

# ================================
# HALAMAN ANALISIS
# ================================
elif page == "Analisis":
    get_analisis_page(st)

# ===============================
# HALAMAN Tentang
# ===============================
elif page == "Tentang":
    st.header("Tentang Saya")

    c1, c2 = st.columns([1,7])

    with c1:
        st.image("media/PASFOTO_STUDIO.jpg", width=100)

    with c2:
        st.markdown("""
            **Richard Souwiko**  
        """)
        st.markdown("""
            Jurusan Teknik Informatika,
        """)
        st.markdown("""
            Universitas Tarumanagara
        """)

    st.header("Tentang Sistem")

    c3, c4 = st.columns([2,1])

    with c3:
        st.markdown("""
        Sistem rancangan ini dikembangkan sebagai sistem rekomendasi pemain sepak bola berbasis statistik.
        Tujuannya adalah membantu tim - tim liga Indonesia menemukan pemain lokal yang performanya mirip dengan pemain asing,
        menggunakan algoritma **Mean Shift** dan **Cosine Similarity**.
    """)

    st.markdown("""
    Sistem ini dibangun dengan:
    - **Python** digunakan sebagai bahasa pemrograman utama
    - **Streamlit** digunakan untuk membangun UI website
    - **Django** digunakan sebagai ORM untuk mengelola database
    - **PostgreSQL** digunakan sebagai database untuk menyimpan data liga dan statistik pemain    

    """)

    st.header("Sumber Data")
    st.write("Data diambil dari : [Sumber data](https://www.sofascore.com/tournament/football/indonesia/liga-1/1015#id:65049)")

    st.markdown("---")

    st.header("Kontak")
    st.markdown("[richard.s050804@gmail.com](https://mail.google.com/mail/?view=cm&fs=1&to=richard.s050804@gmail.com) | [GitHub](https://github.com/RichardxSW)")

