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
    get_player_detail, insert_dataset_and_players, get_seasons, get_players_by_season, make_template_excel_bytes
)

# ====== Style ======
st.markdown(
    """
        <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            padding-left: 0px;
            padding-right: 0px;
        }
        </style>
    """, 
    unsafe_allow_html=True
)

# =========================
# NAVIGASI SIDEBAR (BUTTON)
# =========================
# st.sidebar.title("Navigasi")
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
    st.title("Sistem Rekomendasi Pemain Sepak Bola Indonesia")
    st.caption("Liga 1 Indonesia")
    st.markdown(
        """
        - Unggah dataset pemain per musim di halaman Unggah Dataset
        - Pilih musim dan cari pemain rekomendasi di halaman Riwayat & Analisis
        - Lihat tentang pembuat di halaman About
        """
    )

elif page == "Unggah Dataset":
    # st.header("Unggah Dataset")

    st.header("Template Dataset")

    data = {
        "Player": ["Marc Klok", ""],
        "Team": ["Persib Bandung", ""],
        "Nationality": ["Indonesia", ""],
        "Position": ["DM", ""],
        "Age": [25, ""],
        "Appearance": [34, ""],
        "Total Minute": [3060, ""],
        "Total Goal": [10, ""],
        "Goal/game": [1, ""],
        "Shot/game": [1, ""],
        "SoT/game": [1, ""],
        "Assist": [5, ""],
        "Assist/game": [1, ""],
        "Success Dribble/game": [8, ""],
        "Key Pass/game": [5, ""],
        "Successful Pass/game": [20, ""],
        "Long Ball/game": [10, ""],
        "Successful Crossing/game": [10, ""],
        "Ball Recovered/game": [10, ""],
        "Dribbled Past/game": [5, ""],
        "Clearance/game": [5,""],
        "Error leading to shot": [5, ""],
        "Error leading to shot/game": [5, ""],
        "Total duel won/game": [5, ""],
        "Aerial duel won/game": [5, ""],
    }

    df = pd.DataFrame(data)
    st.dataframe(df)
    # st.write("Wajib **player_name**; opsional **team**, **position**, dan kolom fitur numerik.")
    st.download_button(
        "Download Template",
        data=make_template_excel_bytes(),
        file_name=f"template_dataset_{dt.datetime.now():%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("---")
    st.header("Unggah Dataset")
    with st.form("upload_form"):
        league_name = st.text_input("Nama Liga", value="Liga 1 Indonesia")
        season = st.text_input("Musim", placeholder=f"misal {"2024/2025"}")
        file = st.file_uploader("Unggah file dataset", type="xlsx")
        submitted = st.form_submit_button("Simpan")

    if submitted:
        try:
            if not league_name or not season or not file:
                st.error("Lengkapi nama liga, musim, dan file dataset.")
            else:
                df = pd.read_excel(file)
                insert_dataset_and_players(league_name, season, df)
                st.success(f"Sukses menyimpan dataset: {league_name} – {season}.")
        except KeyError as ke:
            st.error(str(ke))
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

elif page == "Riwayat & Analisis":
    st.header("Riwayat & Analisis")
    seasons = get_seasons()
    if not seasons:
        st.warning("Belum ada dataset yang diunggah. Unggah dataset terlebih dahulu di halaman Unggah Dataset.")
    else:
        selected_season = st.selectbox("Pilih Musim", seasons, index=0)
        players = get_players_by_season(selected_season) if selected_season else []
        if players:
            selected_player = st.selectbox("Pilih Pemain Acuan", players, index=0)
        else:
            selected_player = None
            st.info("Tidak ada pemain yang ditemukan")

        st.markdown("---")
        if selected_season and selected_player:
            detail = get_player_detail(selected_season, selected_player)

            if detail:
                st.subheader("Tentang Pemain")
                # st.write(f"**Musim**: {selected_season}")
                # st.write(f"**Pemain**: {selected_player}")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"Team: {detail.get("team")}")
                    st.write(f"Nationality: {detail.get("nationality")}")
                    st.write(f"Position: {detail.get("position")}")

                with col2:
                    st.write(f"Age: {detail.get("age")}")
                    st.write(f"Appearance: {detail.get("appearance")}")
                    st.write(f"Total minutes played: {detail.get("total_minute")}")

                st.markdown("---")
                st.subheader("Statistik Pemain Acuan")
                
                col3, col4, col5 = st.columns(3)

                with col3:
                    st.write(f"Goal: {detail.get("total_goal")}")
                    st.write(f"Assist: {detail.get("assist")}")
                    st.write(f"Shot per game: {detail.get("shot_per_game")}")
                    st.write(f"Shot on target per game: {detail.get("shot_per_game")}")
                    st.write(f"Successful dribble per game: {detail.get("successful_dribble_per_game")}")

                with col4:
                    st.write(f"Successful pass per game: {detail.get("successful_pass_per_game")}")
                    st.write(f"Key pass per game: {detail.get("key_pass_per_game")}")
                    st.write(f"Long ball pass per game: {detail.get("long_ball_per_game")}")
                    st.write(f"Successful crossing per game: {detail.get("successful_crossing_per_game")}")

                with col5:
                    st.write(f"Ball recovered per game: {detail.get("ball_recovered_per_game")}")
                    st.write(f"Dribbled past per game: {detail.get("dribbled_past_per_game")}")
                    st.write(f"Clearance per game: {detail.get("clearance_per_game")}")
                    st.write(f"Error leading to shot per game: {detail.get("error_per_game"):.1f}")
                    st.write(f"Total duel won per game: {detail.get("total_duel_per_game"):.1f}")
                    st.write(f"Aerial duel won per game: {detail.get("aerial_duel_per_game"):.1f}")

                st.markdown("---")
                st.subheader("Pemain Rekomendasi")

elif page == "About":
    st.header("About")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit…")
