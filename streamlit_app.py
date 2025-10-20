# streamlit_app.py
import os, sys, datetime as dt
import pandas as pd
import streamlit as st
from django.core.exceptions import ValidationError
import matplotlib.pyplot as plt
import numpy as np
import time

# === Bootstrap Django ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iprs.settings")

import django
django.setup()

from players.clustering import run_meanshift, run_meanshift_by_position
# === Import services & utils ===
from players.services import (
    delete_dataset, get_list_of_dataset, get_player_detail, get_recommend_similar_players, insert_dataset_and_players, get_seasons, get_players_by_season, make_template_excel_bytes
)

st.set_page_config(page_title="IPRS", layout="wide")

# ====== Style ======
st.markdown(
    """
        <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            padding-left: 3rem;
            padding-right: 3rem;
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
    "📊 Analisis": "Analisis",
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
        - Pilih musim dan cari pemain rekomendasi di halaman Analisis
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
        season = st.text_input("Musim", placeholder=f"misal 2024/2025")
        file = st.file_uploader("Unggah file dataset", type="xlsx")
        submitted = st.form_submit_button("Simpan")

    if submitted:
        try:
            if not league_name:
                st.error("Isi nama liga terlebih dahulu.")
            if not season:
                st.error("Isi musim terlebih dahulu.")
            if not file:
                st.error("Unggah file dataset terlebih dahulu.")
            else:
                df = pd.read_excel(file)
                insert_dataset_and_players(league_name, season, df)
                st.success(f"Sukses menyimpan dataset: {league_name} – {season}.")
                st.rerun()
        except KeyError as ke:
            st.error(str(ke))
        except ValueError as ve:
            st.error(str(ve))
        except ValidationError as validation_error:
            st.error(str(validation_error))
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    st.markdown("---")

    datasets = get_list_of_dataset()

    if not datasets:
        st.info("Belum ada data yang tersimpan")
    else:
        col_head1, col_head2, col_head3, col_head4, col_head5 = st.columns([3, 2, 2, 2, 1])
        col_head1.write("**Liga**")
        col_head2.write("**Musim**")
        col_head3.write("**Jumlah Pemain**")
        col_head4.write("**Diunggah**")
        col_head5.write("**Hapus**")

        for ds in datasets:
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            col1.write(ds["league_name"])
            col2.write(ds["season"])
            col3.write(ds["player_count"])
            col4.write(pd.to_datetime(ds["uploaded_at"]).strftime("%d-%m-%Y %H:%M"))

            if col5.button("🗑️", key=f"del_{ds['id']}"):
                ok = delete_dataset(ds["id"])
                if ok:
                    st.success(f"Data {ds['league_name']} musim ({ds['season']}) berhasil dihapus.")
                    st.rerun()
                else:
                    st.error("Gagal menghapus data liga.")

elif page == "Analisis":
    st.header("Analisis")
    seasons = get_seasons()
    if not seasons:
        st.warning("Belum ada data liga yang diunggah. Unggah dataset terlebih dahulu di halaman Unggah Dataset.")
    else:
        season_choices = ["Pilih Musim"] + seasons
        selected_season = st.selectbox("Pilih Musim", season_choices, index=0)

        selected_position = None
        selected_player = None

        if selected_season != "Pilih Musim":            
            if st.button("Clustering"):
                with st.spinner("Sedang menjalankan clustering..."):
                    result = run_meanshift_by_position(selected_season)
                st.session_state.cluster_result = result
                st.session_state.selected_season = selected_season                
                st.success("Clustering berhasil.")

        results = st.session_state.get("cluster_result")
        if results and st.session_state.get("selected_season") == selected_season:
            with st.expander("Hasil Clustering"):
                for group_name, res in results.items():
                    st.markdown(f"## {group_name}")
                    if not res:
                        st.warning(f"Tidak cukup data untuk posisi {group_name.lower()}.")
                        continue

                    df_eval = res["res_table"]
                    st.dataframe(df_eval)

                    X2 = res["emb2d"]

                    def plot_clusters(x2, labels, title):
                        fig, ax = plt.subplots(figsize=(5,4))
                        sc = ax.scatter(x2[:, 0], x2[:, 1], c=labels, s=28, alpha=0.9)
                        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
                        ax.set_title(title)
                        uniq, counts = np.unique(labels, return_counts=True)
                        ax.legend(sc.legend_elements()[0], [f"C{c}: {n}" for c,n in zip(uniq, counts)], loc="best")
                        st.pyplot(fig)
                        plt.close(fig)

                    # Silhouette terbaik
                    if res["best_sil"]:
                        b = res["best_sil"]
                        st.markdown("#### Nilai Silhouette Terbaik")
                        st.caption(f"bandwidth={b['bw']:.1f} | Jumlah Cluster={b['n_clusters']} | Silhouette={b['sil']:.4f} | DBI={b['dbi']:.4f}")
                        plot_clusters(X2, b["labels"], f"{group_name}")

                    # DBI terbaik (jika beda bandwidth)
                    # if res["best_dbi"] and not res["same_bw"]:
                    #     b = res["best_dbi"]
                    #     st.markdown("#### DBI Terbaik")
                    #     st.caption(f"bw={b['bw']:.2f}, clusters={b['n_clusters']}, dbi={b['dbi']:.4f}, silhouette={b['sil']}")
                    #     plot_clusters(X2, b["labels"], f"{group_name} – DBI Terbaik (bw={b['bw']:.2f})")

                if st.button("🔄 Reset Hasil"):
                    st.session_state.cluster_results = None
                    st.session_state.selected_season = None
                    st.rerun()
    
            position_choices = [
                "Pilih posisi pemain acuan",
                "ST",
                "LW",
                "RW",
                "AM",
                "CM",
                "LM",
                "RM",
                "DM",
                "CB",
                "LB",
                "RB"
            ]

            selected_position = st.selectbox("Pilih posisi pemain acuan", position_choices, index=0)

            if selected_position != "Pilih posisi pemain acuan":
                players = get_players_by_season(selected_season, selected_position) if selected_season and selected_position else []
                player_option = ["Pilih Pemain Acuan"] + players
                selected_player = st.selectbox("Pilih Pemain Acuan", player_option, index=0)
            else:
                selected_player = None
                selected_position = None

        # st.markdown("---")
        if selected_season and selected_position and selected_player:
            detail = get_player_detail(selected_season, selected_player)

            if detail:
                st.subheader("Tentang Pemain")
                # st.write(f"**Musim**: {selected_season}")
                # st.write(f"**Pemain**: {selected_player}")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"Team: {detail.get('team')}")
                    st.write(f"Nationality: {detail.get('nationality')}")
                    st.write(f"Position: {detail.get('position')}")

                with col2:
                    st.write(f"Age: {detail.get('age')}")
                    st.write(f"Appearance: {detail.get('appearance')}")
                    st.write(f"Total minutes played: {detail.get('total_minute')}")

                st.markdown("---")
                st.subheader("Statistik Pemain Acuan")
                
                col3, col4, col5 = st.columns(3)

                with col3:
                    st.write(f"Goal: {detail.get('total_goal')}")
                    st.write(f"Assist: {detail.get('assist')}")
                    st.write(f"Shot per game: {detail.get('shot_per_game'):.2f}")
                    st.write(f"Shot on target per game: {detail.get('shot_per_game'):.2f}")
                    st.write(f"Successful dribble per game: {detail.get('successful_dribble_per_game'):.2f}")

                with col4:
                    st.write(f"Successful pass per game: {detail.get('successful_pass_per_game'):.2f}")
                    st.write(f"Key pass per game: {detail.get('key_pass_per_game'):.2f}")
                    st.write(f"Long ball pass per game: {detail.get('long_ball_per_game'):.2f}")
                    st.write(f"Successful crossing per game: {detail.get('successful_crossing_per_game'):.2f}")

                with col5:
                    st.write(f"Ball recovered per game: {detail.get('ball_recovered_per_game'):.2f}")
                    st.write(f"Dribbled past per game: {detail.get('dribbled_past_per_game'):.2f}")
                    st.write(f"Clearance per game: {detail.get('clearance_per_game'):.2f}")
                    st.write(f"Error leading to shot per game: {detail.get('error_per_game'):.2f}")
                    st.write(f"Total duel won per game: {detail.get('total_duel_per_game'):.2f}")
                    st.write(f"Aerial duel won per game: {detail.get('aerial_duel_per_game'):.2f}")

                st.markdown("---")
                st.subheader("Pemain Rekomendasi")

                recommend_count = st.slider(
                    "Jumlah pemain rekomendasi",
                    min_value=1,
                    max_value=5,
                    step=1
                )

                only_indo = st.checkbox("Pemain Indonesia saja", value=False)
                
                if recommend_count and st.button("Cari pemain rekomendasi"):
                    try:
                        recs = get_recommend_similar_players(
                            season=selected_season,
                            position_code=selected_position,   # "ST", "CM", ...
                            anchor_player=selected_player,
                            top_n=recommend_count,
                            only_indonesian=only_indo,
                        )
                        if recs.empty:
                            st.info("Tidak ada rekomendasi yang cocok untuk konfigurasi ini.")
                        else:
                            # tampilkan tabel ringkas
                            cols_show = ["player", "team", "position", "nationality", "similarity"]
                            cols_show = [c for c in cols_show if c in recs.columns]
                            st.dataframe(
                                recs[cols_show].assign(similarity=lambda d: d["similarity"].round(4)),                                
                            )
                    except Exception as e:
                        st.error(f"Gagal mencari rekomendasi: {e}")


elif page == "About":
    st.header("About")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit…")
