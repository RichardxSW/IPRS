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

from players.clustering import get_player_features_df, run_meanshift, run_meanshift_by_position
# === Import services & utils ===
from players.services import (
    delete_dataset, get_list_of_dataset, get_player_detail, insert_dataset_and_players, get_seasons, get_players_by_season, make_template_excel_bytes
)
from players.recommend import get_recommend_similar_players, prepare_comparison_long_df

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
st.sidebar.title("IPRS")
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
        col_head1, col_head2, col_head3, col_head4, col_head5 = st.columns([3, 2, 2, 2, 2])
        col_head1.write("**Liga**")
        col_head2.write("**Musim**")
        col_head3.write("**Jumlah Pemain**")
        col_head4.write("**Diunggah**")
        # col_head5.write("**Hapus**")

        for ds in datasets:
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
            col1.write(ds["league_name"])
            col2.write(ds["season"])
            col3.write(ds["player_count"])
            col4.write(pd.to_datetime(ds["uploaded_at"]).strftime("%d-%m-%Y %H:%M"))

            # if col5.button("🗑️", key=f"del_{ds['id']}"):
            if col5.button("Hapus", key=f"del_{ds['id']}"):
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

            selected_position = st.selectbox("Pilih Posisi Pemain Acuan", position_choices, index=0)

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
                    st.write(f"Total Goal: {detail.get('total_goal')}")
                    st.write(f"Total Assist: {detail.get('assist')}")
                    st.write(f"Shot/Game: {detail.get('shot_per_game'):.2f}")
                    st.write(f"Shot On Target/Game: {detail.get('sot_per_game'):.2f}")
                    st.write(f"Successful Dribble/game: {detail.get('successful_dribble_per_game'):.2f}")

                with col4:
                    st.write(f"Successful Pass/Game: {detail.get('successful_pass_per_game'):.2f}")
                    st.write(f"Key Pass/Game: {detail.get('key_pass_per_game'):.2f}")
                    st.write(f"Long Ball Pass/Game: {detail.get('long_ball_per_game'):.2f}")
                    st.write(f"Successful Crossing/Game: {detail.get('successful_crossing_per_game'):.2f}")

                with col5:
                    st.write(f"Ball Recovered/Game: {detail.get('ball_recovered_per_game'):.2f}")
                    st.write(f"Dribbled Past/Game: {detail.get('dribbled_past_per_game'):.2f}")
                    st.write(f"Clearance/Game: {detail.get('clearance_per_game'):.2f}")
                    st.write(f"Error Leading to Shot: {detail.get('error')}")
                    st.write(f"Total Duel Won/Game: {detail.get('total_duel_per_game'):.2f}")
                    st.write(f"Aerial Duel Won/Game: {detail.get('aerial_duel_per_game'):.2f}")

                st.markdown("---")
                st.subheader("Pemain Rekomendasi")

                recommend_count = st.slider(
                    "Jumlah pemain rekomendasi",
                    min_value=1,
                    max_value=5,
                    step=1
                )

                col7, col8, col9, col10 = st.columns(4)

                with col7:
                    only_indo = st.checkbox("Pemain Indonesia saja", value=False)

                with col8:
                    filter_position = st.checkbox("Posisi yang sama saja", value=False)
                
                if recommend_count and st.button("Cari pemain rekomendasi"):                    
                    recs = get_recommend_similar_players(
                        season=selected_season,
                        position_code=selected_position,
                        anchor_player=selected_player,
                        top_n=recommend_count,
                        only_indonesian=only_indo,
                        filter_position=filter_position
                    )
                    st.session_state["recs_df"] = recs
                    st.session_state["feat_df"] = get_player_features_df(selected_season)
                    st.session_state["cmp_target"] = None

                    if recs.empty:
                        st.info("Tidak ada pemain rekomendasi yang cocok untuk konfigurasi ini.")
                        st.session_state["recs_df"] = None
                        st.session_state["feat_df"] = None
                        st.session_state["cmp_target"] = None
                  
                import altair as alt

                FEATURES_TO_COMPARE = [
                    # isi sesuai kolom fitur kamu, contoh:
                    "age", "appearance", "total_minute",
                    "total_goal", "assist", "shot_per_game",
                    # "tackles_p90", "interceptions_p90", "carries_p90"
                ]

                recs_df = st.session_state.get("recs_df")
                feat_df = st.session_state.get("feat_df")

                if recs_df is not None and not recs_df.empty:
                    # ====== header kolom ======
                    col_head1, col_head2, col_head3, col_head4, col_head5, col_head6 = st.columns([3, 3, 2, 2, 2, 2])
                    col_head1.write("**Pemain**")
                    col_head2.write("**Tim**")
                    col_head3.write("**Posisi**")
                    col_head4.write("**Nationality**")
                    col_head5.write("**Kemiripan**")
                    # col_head6.write("**Aksi**")

                    cols_show = ["player", "team", "position", "nationality", "similarity"]
                    cols_show = [c for c in cols_show if c in recs_df.columns]

                    for i, ds in recs_df[cols_show].iterrows():
                        col1, col2, col3, col4, col5, col6 = st.columns([3, 3, 2, 2, 2, 2])

                        sim_pct = float(ds.get("similarity", 0.0))
                        sim_pct = max(0.0, min(1.0, sim_pct)) * 100.0  

                        col1.write(ds.get("player", "-"))
                        col2.write(ds.get("team", "-"))
                        col3.write(ds.get("position", "-"))
                        col4.write(ds.get("nationality", "-"))
                        col5.write(f"{sim_pct:.2f}%")

                        # tombol bandingkan per baris
                        if col6.button("Bandingkan", key=f"cmp_{i}_{ds.get('player','')}"):
                            st.session_state["cmp_target"] = ds['player']

                    if st.session_state.get("cmp_target"):
                        target_player = st.session_state["cmp_target"]
                        with st.expander(f"Perbandingan {selected_player} dengan {target_player}", expanded=True):                                
                            try:
                                long_df = prepare_comparison_long_df(
                                    feat_df=feat_df,
                                    anchor_player=selected_player,
                                    target_player=target_player,
                                    features=FEATURES_TO_COMPARE
                                )
                            except ValueError as e:
                                st.error(str(e))
                            else:
                                # expander tepat di bawah baris ini
                                if not FEATURES_TO_COMPARE:
                                    st.warning("Daftar fitur kosong. Isi `FEATURES_TO_COMPARE` dulu.")
                                else:
                                    # render bar chart per-fitur dalam kolom (mis. 3 kolom per baris)
                                    N_COLS = 2
                                    fitur_list = list(long_df["Fitur"].unique())
                                    for start in range(0, len(fitur_list), N_COLS):
                                        cols_plot = st.columns(N_COLS)
                                        batch = fitur_list[start:start+N_COLS]
                                        for c, fitur in zip(cols_plot, batch):
                                            sub = long_df[long_df["Fitur"] == fitur]
                                            chart = (
                                                alt.Chart(sub)
                                                .mark_bar()
                                                .encode(
                                                    x=alt.X("Pemain:N", axis=alt.Axis(title=None, labelAngle=0)),
                                                    y=alt.Y("Nilai:Q", axis=alt.Axis(title=None)),
                                                    tooltip=["Pemain:N", "Nilai:Q"]
                                                )
                                                .properties(
                                                    title={
                                                        "text": str(fitur),
                                                        "anchor": "middle",
                                                        "align": "center",                                                        
                                                    },                                                    
                                                    width="container", 
                                                    height=220
                                                )
                                            )
                                            with c:
                                                st.altair_chart(chart, use_container_width=True)
                # else:
                #     st.info("Klik **Cari pemain rekomendasi**.")
elif page == "About":
    st.header("About")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit…")
