# streamlit_app.py
import os, sys, datetime as dt
import pandas as pd
import streamlit as st

# === Bootstrap Django ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iprs.settings")

import django
django.setup()

# === Import services & utils ===
from players.services import (
    insert_dataset_and_players, get_seasons, get_players_by_season, make_template_excel_bytes
)

# =========================
# NAVIGASI SIDEBAR (BUTTON)
# =========================
st.sidebar.title("Navigasi")
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

col_map = {
    "🏠 Beranda": "Beranda",
    "📤 Unggah Dataset": "Unggah Dataset",
    "📊 Riwayat & Analisis": "Riwayat & Analisis",
    "ℹ️ About": "About",
}
for label, target in col_map.items():
    if st.sidebar.button(label):
        st.session_state.page = target

page = st.session_state.page

# =========================
# PAGES
# =========================
if page == "Beranda":
    st.title("Perancangan Sistem Rekomendasi Pemain (Mean Shift) — Liga 1")
    st.caption("Streamlit + Django ORM")
    st.markdown(
        """
        - Unggah dataset pemain per musim → simpan ke PostgreSQL  
        - Riwayat per musim → pilih pemain  
        - (Berikutnya) Analisis cluster & rekomendasi
        """
    )

elif page == "Unggah Dataset":
    st.header("Unggah Dataset Pemain")

    st.subheader("Template Dataset")
    st.write("Wajib **player_name**; opsional **team**, **position**, dan kolom fitur numerik.")
    st.download_button(
        "Download Template Excel",
        data=make_template_excel_bytes(),
        file_name=f"template_dataset_{dt.datetime.now():%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
    st.subheader("Form Unggah")
    with st.form("upload_form"):
        league_name = st.text_input("Nama Liga", value="Liga 1 Indonesia")
        season = st.text_input("Musim", placeholder="mis. 2024/2025")
        file = st.file_uploader("Unggah file CSV / Excel", type=["csv", "xlsx"])
        submitted = st.form_submit_button("Simpan ke Database")

    if submitted:
        try:
            if not league_name or not season or not file:
                st.error("Lengkapi liga, musim, dan file.")
            else:
                df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
                insert_dataset_and_players(league_name, season, df)
                st.success(f"Sukses menyimpan dataset: {league_name} – {season}.")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

elif page == "Riwayat & Analisis":
    st.header("Riwayat & Analisis")
    seasons = get_seasons()
    if not seasons:
        st.warning("Belum ada data musim. Unggah dataset terlebih dahulu.")
    else:
        selected_season = st.selectbox("Pilih Musim", seasons, index=0)
        players = get_players_by_season(selected_season) if selected_season else []
        if players:
            selected_player = st.selectbox("Pilih Pemain", players, index=0)
        else:
            selected_player = None
            st.info("Tidak ada pemain untuk musim tersebut.")

        st.markdown("---")
        if selected_season and selected_player:
            st.subheader("Ringkasan")
            st.write(f"**Musim**: {selected_season}")
            st.write(f"**Pemain**: {selected_player}")
            st.caption("Nanti: grafik cluster, top-N pemain mirip, dan skor jarak.")

elif page == "About":
    st.header("About")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit…")
