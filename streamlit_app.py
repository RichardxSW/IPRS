import os, sys, datetime as dt
import pandas as pd
import streamlit as st
from django.core.exceptions import ValidationError
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

from players.bar_chart import BarDataMissing, build_cluster_feature_bar_df, get_features_for_group

# === INIT DJANGO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iprs.settings")
import django
django.setup()

from players.clustering import FEATURE_LABELS, FEATURES_BY_POSITION, build_cluster_members_df, get_player_features_df, run_meanshift_by_position
from players.services import (
    POSITION_CHOICES, delete_dataset, get_clubs_by_season, get_list_of_season, get_player_detail, get_players_by_season_and_club, insert_dataset_and_players, get_seasons, get_players_by_season, make_template_excel_bytes
)
from players.recommend import FEATURES_TO_COMPARE, get_recommend_similar_players, prepare_comparison_chart_data

st.set_page_config(page_title="IPRS", layout="wide")

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
        </style>
    """, 
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("IPRS")
if "page" not in st.session_state:
    st.session_state.page = "Beranda"

sidebar_map = {
    "Beranda": "Beranda",
    "Unggah Dataset": "Unggah Dataset",
    "Analisis": "Analisis",
    "About": "About",
}

for label, target in sidebar_map.items():
    if st.sidebar.button(label):
        st.session_state.page = target

page = st.session_state.page

# =========================
# HALAMAN BERANDA
# =========================
if page == "Beranda":
    st.title("Sistem Rekomendasi Pemain Sepak Bola Indonesia")
    st.markdown(
        """
        ---
        ### Fitur yang terdapat di website ini
        """
    )
    st.write("- **Unggah Dataset** → Download template dataset dan menyimpan data liga dan pemain.")
    st.write("- **Analisis** → Pilih musim, lakukan clustering, mencari pemain rekomendasi, dan membandingkan pemain acuan dan pemain rekmendasi.")
    st.write("- **About** → Lihat lebih lanjut tentang pembuat dan website.")
    st.markdown("---")

# HALAMAN UNGGAH DATASET
elif page == "Unggah Dataset":
    st.header("Template Dataset")

    # TEMPLATE DATA
    data = {
        "Player": ["Marc Klok", np.nan],
        "Team": ["Persib Bandung", np.nan],
        "Nationality": ["Indonesia",np.nan],
        "Position": ["DM", np.nan],
        "Age": [25, np.nan],
        "Appearance": [34, np.nan],
        "Total Minute": [3060, np.nan],
        "Total Goal": [10, np.nan],
        "Goal/game": [1,np.nan],
        "Shot/game": [1, np.nan],
        "SoT/game": [1, np.nan],
        "Assist": [5, np.nan],
        "Assist/game": [1, np.nan],
        "Success Dribble/game": [8, np.nan],
        "Key Pass/game": [5, np.nan],
        "Successful Pass/game": [20, np.nan],
        "Long Ball/game": [10, np.nan],
        "Successful Crossing/game": [10, np.nan],
        "Ball Recovered/game": [10, np.nan],
        "Dribbled Past/game": [5, np.nan],
        "Clearance/game": [5,np.nan],
        "Error leading to shot": [5, np.nan],
        "Error leading to shot/game": [5, np.nan],
        "Total duel won/game": [5, np.nan],
        "Aerial duel won/game": [5, np.nan],
    }

# DOWNLOAD FILE TEMPLATE DATASET
    df = pd.DataFrame(data)
    st.dataframe(df)
    st.download_button(
        "Download Template",
        data=make_template_excel_bytes(),
        file_name=f"template_dataset_{dt.datetime.now():%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# UPLOAD FILE DATASET
    st.markdown("---")
    st.header("Unggah Dataset")
    with st.form("upload_form"):
        league_name = st.text_input("Nama Liga", value="Liga 1 Indonesia")
        season = st.text_input("Musim", placeholder=f"misal 2024/2025", value="2024/2025")
        file = st.file_uploader("Unggah file dataset", type="xlsx")
        submitted = st.form_submit_button("Simpan")

    if submitted:
        try:
            # VALIDASI DATA
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

    seasons = get_list_of_season()

    if not seasons:
        st.info("Belum ada data yang tersimpan")
    else:
        # DATA MUSIM YANG SUDAH DIUNGGAH
        col_head1, col_head2, col_head3, col_head4, col_head5 = st.columns([3, 2, 2, 2, 2])
        col_head1.write("**Liga**")
        col_head2.write("**Musim**")
        col_head3.write("**Jumlah Pemain**")
        col_head4.write("**Diunggah**")

        for ds in seasons:
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
            col1.write(ds["league_name"])
            col2.write(ds["season"])
            col3.write(ds["player_count"])
            col4.write(pd.to_datetime(ds["uploaded_at"]).strftime("%d-%m-%Y %H:%M"))

            if col5.button("Hapus", key=f"del_{ds['id']}"):
                ok = delete_dataset(ds["id"])
                if ok:
                    st.success(f"Data {ds['league_name']} musim ({ds['season']}) berhasil dihapus.")
                    st.rerun()
                else:
                    st.error("Gagal menghapus data liga.")

# HALAMAN ANALISIS
elif page == "Analisis":
    st.header("Analisis")

    # ==== HELPER UNTUK RESET STATE ====
    st.session_state.setdefault("recommend_state", None)
    st.session_state.setdefault("features", None)
    st.session_state.setdefault("compare_recommend", None)
    st.session_state.setdefault("cluster_result", None)
    st.session_state.setdefault("selected_season", None)

    def _clear_reco_state():
        st.session_state["recommend_state"] = None
        st.session_state["features"] = None
        st.session_state["compare_recommend"] = None

    def _clear_cluster_state():
        st.session_state["cluster_result"] = None
        st.session_state["selected_season"] = None

    seasons = get_seasons()
    if not seasons:
        st.warning("Belum ada data liga yang diunggah. Unggah dataset terlebih dahulu di halaman Unggah Dataset.")
    else:
        # DROPDOWN PILIH MUSIM
        season_choices = ["Pilih Musim"] + seasons
        selected_season = st.selectbox("Pilih Musim", season_choices, index=0)

        selected_position = None
        selected_player = None

        if selected_season != "Pilih Musim":            
            if st.button("Clustering"):
                _clear_reco_state()
                with st.spinner("Sedang menjalankan clustering..."):
                    result = run_meanshift_by_position(selected_season)
                st.session_state.cluster_result = result
                st.session_state.selected_season = selected_season                
                st.success("Clustering berhasil.")

        # HASIL CLUSTERING
        results = st.session_state.get("cluster_result")
        if results and st.session_state.get("selected_season") == selected_season:
            with st.expander("Hasil Clustering"):
                group_items = list(results.items())
                N_COLS = 3

                def plot_clusters(X_pca, labels, title):
                    fig, ax = plt.subplots(figsize=(4,3))
                    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=28, alpha=0.9)
                    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
                    ax.set_title(title)
                    uniq, counts = np.unique(labels, return_counts=True)
                    ax.legend(sc.legend_elements()[0], [f"C{c}: {n}" for c,n in zip(uniq, counts)], loc="best")
                    st.pyplot(fig)
                    plt.close(fig)

                # tampilkan per posisi
                for start in range(0, len(group_items), N_COLS):
                    columns = st.columns(N_COLS)
                    for c, (group_name, results) in zip(columns, group_items[start:start+N_COLS]):
                        with c:
                            st.markdown(f"### {group_name}")
                            if not results:
                                st.warning(f"Tidak cukup data untuk posisi {group_name.lower()}.")
                                continue

                            df_eval = results["cluster_result"]
                            X_pca = results["X_pca"]
                            best = results["best_silhouette"]

                            # tabel hasil loop clustering dengan bandwidth dari 0,5 - 5
                            if best:
                                st.caption(
                                    f"BW={best['bw']:.1f} | Clusters={best['n_clusters']} | "
                                    f"Sil={best['silhouette']:.4f} | DBI={best['dbi']:.4f}"
                                )
                            if isinstance(df_eval, pd.DataFrame) and not df_eval.empty:
                                st.data_editor(
                                    df_eval.reset_index(drop=True),
                                    hide_index=True,
                                    disabled=True,
                                    height=180
                                )

                            # menampilkan scatter plot nilai silhouette terbaik
                            if best and X_pca is not None:
                                st.write("Nilai Silhouette Terbaik")
                                plot_clusters(X_pca, best["labels"], f"{group_name}")

                                # === TABEL DAFTAR PEMAIN TIAP CLUSTER ===
                                df_members = build_cluster_members_df(results, best)
                                if isinstance(df_members, pd.DataFrame) and not df_members.empty:
                                    st.write("Daftar Pemain")
                                    st.data_editor(
                                        df_members,
                                        hide_index=True,
                                        disabled=True,
                                        height=260,
                                    )
                                else:
                                    st.info("Daftar pemain per cluster tidak ditemukan.")

                            # init bar chart tiap fitur
                            try:
                                features = get_features_for_group(group_name, FEATURES_BY_POSITION)
                                chart_data = build_cluster_feature_bar_df(results, features)
                            except BarDataMissing as e:
                                st.info(f"Bar chart tidak dapat ditampilkan: {e}")
                            else:
                                chart_data["Fitur"] = pd.Categorical(
                                    chart_data["Fitur"],
                                    categories=features,
                                    ordered=True
                                )
                                clusters_order = sorted(chart_data["Cluster"].unique(), key=lambda s: int(s[1:]))  # "C0","C1",...
                                chart_data["Cluster"] = pd.Categorical(chart_data["Cluster"], categories=clusters_order, ordered=True)

                                # tampilkan tiap fitur satu bar chart
                                for fitur in features:
                                    sub = chart_data[chart_data["Fitur"] == fitur]
                                    chart_title = FEATURE_LABELS.get(fitur, fitur)
                                    chart = (
                                        alt.Chart(sub)
                                        .mark_bar()
                                        .encode(
                                            x=alt.X("Cluster:N", axis=alt.Axis(title=None, labelAngle=0)),
                                            y=alt.Y("Mean:Q", axis=alt.Axis(title=None)),
                                            color=alt.Color("Cluster:N", title=None, scale=alt.Scale(scheme="tableau10")),
                                            tooltip=[
                                                "Cluster:N",
                                                alt.Tooltip("Mean:Q", title="Rata-rata", format=".3f")
                                            ],
                                        )
                                        .properties(
                                            title={"text": chart_title, "anchor": "middle", "align": "center"},
                                            width="container",
                                            height=220
                                        )
                                    )
                                    st.altair_chart(chart)

                if st.button("🔄 Reset Hasil Clustering"):
                    _clear_reco_state()
                    _clear_cluster_state()
                    st.rerun()            

            selected_position = st.selectbox("Pilih Posisi Pemain Acuan", POSITION_CHOICES, index=0)

            if selected_position != "Pilih posisi pemain acuan":
                st.session_state.setdefault("club_filter", "Semua")

                club_options = ["Semua"] + get_clubs_by_season(season=selected_season)                    
                
                selected_club = st.selectbox("Pilih Klub (opsional)", options=club_options, index=0)
                st.session_state["club_filter"] = selected_club

                if selected_season and selected_position:
                    if selected_club and selected_club != "Semua":
                        players = get_players_by_season_and_club(selected_season, selected_position, selected_club)
                    else:                        
                        players = get_players_by_season(selected_season, selected_position)
                else:
                    players = []

                player_option = ["Pilih Pemain Acuan"] + (players if players else [])
                selected_player = st.selectbox("Pilih Pemain Acuan", player_option, index=0)
            else:
                selected_player = None
                selected_position = None

        # -------------------- RESET STATE SAAT PILIHAN BERUBAH --------------------
        st.session_state.setdefault("prev_season", None)
        st.session_state.setdefault("prev_position", None)
        st.session_state.setdefault("prev_anchor", None)
        st.session_state.setdefault("prev_club_filter", None)

        def _clear_reco_state():
            st.session_state["recommend_state"] = None
            st.session_state["features"] = None
            st.session_state["compare_recommend"] = None

        def _clear_cluster_state():
            st.session_state["cluster_result"] = None
            st.session_state["selected_season"] = None

        season_changed   = (st.session_state["prev_season"]   != selected_season)
        position_changed = (st.session_state["prev_position"] != selected_position)
        anchor_changed   = (st.session_state["prev_anchor"]   != selected_player)
        club_changed     = (st.session_state["prev_club_filter"] != st.session_state.get("club_filter"))

        if season_changed:
            _clear_reco_state()
            _clear_cluster_state()
            if selected_season == "Pilih Musim":
                st.session_state["prev_season"] = selected_season
                st.session_state["prev_position"] = None
                st.session_state["prev_anchor"] = None

        if position_changed:
            _clear_reco_state()
            if selected_position == "Pilih posisi pemain acuan":
                st.session_state["prev_position"] = selected_position
                st.session_state["prev_anchor"] = None

        if club_changed:
            _clear_reco_state()
            st.session_state["prev_anchor"] = None

        if anchor_changed:
            _clear_reco_state()

        st.session_state["prev_season"]   = selected_season
        st.session_state["prev_position"] = selected_position
        st.session_state["prev_anchor"]   = selected_player
        st.session_state["prev_club_filter"] = st.session_state.get("club_filter")
        # -------------------------------------------------------------------------
        
        # DETAIL PEMAIN ACUAN
        if selected_season and selected_position and selected_player and selected_player != "Pilih Pemain Acuan":
            detail = get_player_detail(selected_season, selected_player)

            if detail:
                st.subheader("Tentang Pemain")
                st.write(f"Pemain: **{selected_player}**")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"Team: **{detail.get('team')}**")
                    st.write(f"Nationality: **{detail.get('nationality')}**")
                    st.write(f"Position: **{detail.get('position')}**")

                with col2:
                    st.write(f"Age: **{detail.get('age')}**")
                    st.write(f"Appearance: **{detail.get('appearance')}**")
                    st.write(f"Total Minutes Played: **{detail.get('total_minute')}**")

                st.markdown("---")
                st.subheader("Statistik Pemain Acuan")
                
                col3, col4, col5 = st.columns(3)

                with col3:
                    st.write(f"Total Goal: **{detail.get('total_goal')}**")
                    st.write(f"Total Assist: **{detail.get('assist')}**")
                    st.write(f"Shot/Game: **{detail.get('shot_per_game'):.2f}**")
                    st.write(f"Shot On Target/Game: **{detail.get('sot_per_game'):.2f}**")
                    st.write(f"Successful Dribble/Game: **{detail.get('successful_dribble_per_game'):.2f}**")

                with col4:
                    st.write(f"Successful Pass/Game: **{detail.get('successful_pass_per_game'):.2f}**")
                    st.write(f"Key Pass/Game: **{detail.get('key_pass_per_game'):.2f}**")
                    st.write(f"Long Ball Pass/Game: **{detail.get('long_ball_per_game'):.2f}**")
                    st.write(f"Successful Crossing/Game: **{detail.get('successful_crossing_per_game'):.2f}**")

                with col5:
                    st.write(f"Ball Recovered/Game: **{detail.get('ball_recovered_per_game'):.2f}**")
                    st.write(f"Dribbled Past/Game: **{detail.get('dribbled_past_per_game'):.2f}**")
                    st.write(f"Clearance/Game: **{detail.get('clearance_per_game'):.2f}**")
                    st.write(f"Error Leading to Shot: **{detail.get('error')}**")
                    st.write(f"Total Duel Won/Game: **{detail.get('total_duel_per_game'):.2f}**")
                    st.write(f"Aerial Duel Won/Game: **{detail.get('aerial_duel_per_game'):.2f}**")

                st.markdown("---")
                st.subheader("Pemain Rekomendasi")
                
                st.session_state.setdefault("recommend_state", None)
                st.session_state.setdefault("features", None)
                st.session_state.setdefault("compare_recommend", None)

                recommend_count = st.slider(
                    "Jumlah pemain rekomendasi",
                    min_value=1,
                    max_value=10,
                    step=1,
                    value=10
                )

                col7, col8, col9, col10 = st.columns(4)

                with col7:
                    only_indo = st.checkbox("Pemain Indonesia saja", value=False)

                with col8:
                    filter_position = st.checkbox("Posisi yang sama saja", value=False)

                with col9:
                    diff_club = st.checkbox("Klub yang berbeda saja", value=False)
                
                if recommend_count and st.button("Cari pemain rekomendasi"):
                    recommend_players = get_recommend_similar_players(
                        season=selected_season,
                        position_code=selected_position,
                        anchor_player=selected_player,
                        recommend_count=recommend_count,
                        only_indonesian=only_indo,
                        filter_position=filter_position,
                        diff_club=diff_club
                    )

                    if recommend_players.empty:
                        st.info("Tidak ada pemain rekomendasi yang cocok untuk konfigurasi ini.")
                        st.session_state["recommend_state"] = None
                        st.session_state["features"] = None
                        st.session_state["compare_recommend"] = None
                    else:
                        st.session_state["recommend_state"] = recommend_players
                        st.session_state["features"] = get_player_features_df(selected_season)
                        st.session_state["compare_recommend"] = None                                                                                

                recommend_state = st.session_state.get("recommend_state")
                features = st.session_state.get("features")

                # DAFTAR PEMAIN REKOMENDASI
                if recommend_state is not None and isinstance(recommend_state, pd.DataFrame) and not recommend_state.empty:
                    st.subheader("Hasil Pemain Rekomendasi")
                    col_head1, col_head2, col_head3, col_head4, col_head5, col_head6 = st.columns([3, 3, 2, 2, 2, 2])
                    col_head1.write("**Pemain**")
                    col_head2.write("**Tim**")
                    col_head3.write("**Posisi**")
                    col_head4.write("**Nationality**")
                    col_head5.write("**Kemiripan**")

                    recommend_columns = ["player", "team", "position", "nationality", "similarity"]
                    recommend_column = [c for c in recommend_columns if c in recommend_state.columns]

                    for i, ds in recommend_state[recommend_column].iterrows():
                        col1, col2, col3, col4, col5, col6 = st.columns([3, 3, 2, 2, 2, 2])

                        similarity = float(ds.get("similarity", 0.0))
                        similarity = max(0.0, min(1.0, similarity)) * 100.0  

                        col1.write(ds.get("player", "-"))
                        col2.write(ds.get("team", "-"))
                        col3.write(ds.get("position", "-"))
                        col4.write(ds.get("nationality", "-"))
                        col5.write(f"{similarity:.2f}%")

                        # BUTTON BANDINGKAN PER PEMAIN
                        if col6.button("Bandingkan", key=f"compare_{i}_{ds.get('player','')}"):
                            st.session_state["compare_recommend"] = ds['player']
                    
                    target_player = st.session_state["compare_recommend"]
                    if target_player and features is not None:
                        with st.expander(f"Perbandingan {selected_player} dengan {target_player}", expanded=True):                                
                            try:
                                chart_data = prepare_comparison_chart_data(
                                    features=features,
                                    anchor_player=selected_player,
                                    target_player=target_player,
                                    features_to_compare=FEATURES_TO_COMPARE
                                )
                            except ValueError as e:
                                st.error(str(e))
                            else:
                                if not FEATURES_TO_COMPARE:
                                    st.warning("Daftar fitur kosong. Isi `FEATURES_TO_COMPARE` dulu.")
                                else:
                                    # MENAMPILKAN BAR CHART
                                    N_COLS = 2
                                    fitur_list = list(chart_data["Fitur"].unique())
                                    for start in range(0, len(fitur_list), N_COLS):
                                        columns_plot = st.columns(N_COLS)
                                        batch = fitur_list[start:start+N_COLS]
                                        for c, fitur in zip(columns_plot, batch):
                                            sub = chart_data[chart_data["Fitur"] == fitur]
                                            chart_title = FEATURE_LABELS.get(fitur, fitur)
                                            chart = (
                                                alt.Chart(sub)
                                                .mark_bar()
                                                .encode(
                                                    x=alt.X("Pemain:N", axis=alt.Axis(title=None, labelAngle=0)),
                                                    y=alt.Y("Nilai:Q", axis=alt.Axis(title=None)),
                                                    color=alt.Color("Pemain:N", title=None, scale=alt.Scale(scheme="tableau10")),
                                                    tooltip=["Pemain:N", "Nilai:Q"],
                                                )
                                                .properties(
                                                    title={
                                                        "text": chart_title,
                                                        "anchor": "middle",
                                                        "align": "center",                                                        
                                                    },                                                    
                                                    width="container", 
                                                    height=220
                                                )
                                            )
                                            with c:
                                                st.altair_chart(chart)

# HALAMAN ABOUT
elif page == "About":
    st.header("Tentang Saya")

    c1, c2 = st.columns([1,7])

    with c1:
        st.image("PASFOTO_STUDIO.jpg", width=100)

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

    st.header("Tentang Website")

    c3, c4 = st.columns([2,1])

    with c3:
        st.markdown("""
        Website ini dikembangkan sebagai sistem rekomendasi pemain sepak bola berbasis statistik.
        Tujuannya adalah membantu tim - tim liga Indonesia menemukan pemain lokal yang performanya mirip dengan pemain asing,
        menggunakan algoritma **Mean Shift** dan **Cosine Similarity**.
    """)

    st.markdown("""
    Website ini dibangun dengan:
    - **Python** sebagai bahasa pemrograman utama
    - **Streamlit** digunakan untuk membangun UI website
    - **Django** digunakan sebagai backend service
    - **PostgreSQL** digunakan sebagai database untuk menyimpan data statistik pemain    

    """)

    st.header("Sumber Data")
    st.write("Data diambil dari : [Sumber data](https://www.sofascore.com/tournament/football/indonesia/liga-1/1015#id:65049)")

    st.markdown("---")

    st.header("Kontak")
    st.markdown("[richard.s050804@gmail.com](https://mail.google.com/mail/?view=cm&fs=1&to=richard.s050804@gmail.com) | [GitHub](https://github.com/RichardxSW)")

