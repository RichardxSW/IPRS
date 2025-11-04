from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from players.bar_chart import BarDataMissing, get_cluster_feature_chart_data, get_features_for_group
from players.clustering import FEATURE_LABELS, FEATURES_BY_POSITION, get_cluster_members_data, get_player_features_data, run_meanshift_by_position
from players.recommend import FEATURES_TO_COMPARE, get_comparison_chart_data, get_recommend_similar_players
from players.services import POSITION_CHOICES, clear_cluster_state, clear_recommend_state, get_clubs_by_season, get_leagues, get_player_detail, get_players_by_season, get_players_by_season_and_club, get_seasons
import altair as alt


def get_analisis_page(st):    
    st.header("Analisis")

    # UNTUK RESET STATE
    st.session_state.setdefault("prev_league", None)
    st.session_state.setdefault("recommend_state", None)
    st.session_state.setdefault("features", None)
    st.session_state.setdefault("compare_recommend", None)
    st.session_state.setdefault("cluster_result", None)
    st.session_state.setdefault("selected_season", None)

    # UNTUK DROPDOWN PILIH LIGA
    league_choices = ["Pilih Liga"] + get_leagues()
    selected_league = st.selectbox("Pilih Liga", league_choices, index=0)

    st.session_state.setdefault("prev_league", None)
    league_changed = (st.session_state["prev_league"] != selected_league)

    if league_changed:
        clear_recommend_state(st)
        clear_cluster_state(st)
        st.session_state["prev_league"] = selected_league
        st.session_state["prev_season"] = None
        st.session_state["prev_position"] = None
        st.session_state["prev_anchor"] = None
        st.session_state["prev_club_filter"] = None

        if "selected_season" in st.session_state:
            st.session_state["selected_season"] = "Pilih Musim"

    if not league_choices:
        st.warning("Belum ada data liga yang diunggah. Unggah dataset terlebih dahulu di halaman Unggah Dataset.")
    else:
        if selected_league != "Pilih Liga":        
            seasons = get_seasons(selected_league)
            if not seasons:
                st.warning("Tidak ada data musim yang tersedia untuk liga ini.")
            else:
                # DROPDOWN PILIH MUSIM
                season_choices = ["Pilih Musim"] + seasons
                selected_season = st.selectbox("Pilih Musim", season_choices, index=0, key=f"season_{selected_league}")

                selected_position = None
                selected_player = None

                if selected_season != "Pilih Musim":
                    # BUTTON UNTUK MEMULAI CLUSTERING
                    if st.button("Clustering"):
                        clear_recommend_state(st)
                        with st.spinner("Sedang menjalankan clustering..."):
                            result = run_meanshift_by_position(selected_league, selected_season)
                        st.session_state.cluster_result = result
                        st.session_state.selected_season = selected_season                
                        st.success("Clustering berhasil.")

                # HASIL CLUSTERING
                results = st.session_state.get("cluster_result")
                if results and st.session_state.get("selected_season") == selected_season:
                    with st.expander("Hasil Clustering", expanded=False):
                        group_items = list(results.items())
                        group_names = [g for g, _ in group_items]

                        tabs = st.tabs(group_names)

                        for (group_name, group_result), tab in zip(group_items, tabs):
                            with tab:
                                st.markdown(f"### Hasil Clustering {group_name}")
                                if not group_result:
                                    st.warning(f"Tidak cukup data untuk posisi {group_name.lower()}.")
                                    continue

                                cluster_result = group_result.get("cluster_result")
                                X_pca   = group_result.get("X_pca")
                                best_result    = group_result.get("best_silhouette")

                                # NILAI SILHOUETTE TERBAIK
                                if best_result:
                                    st.write(
                                        f"BW = {best_result['bw']:.1f} | Clusters = {best_result['n_clusters']} | "
                                        f"Silhouette = {best_result['silhouette']:.4f} | DBI = {best_result['dbi']:.4f}"
                                    )
                                st.markdown("---")

                                c1, c2, c3 = st.columns(3, vertical_alignment="top")

                                # TABEL HASIL EVALUASI CLUSTERING 
                                with c1:
                                    st.write("##### Evaluasi Clustering")
                                    if isinstance(cluster_result, pd.DataFrame) and not cluster_result.empty:
                                        st.data_editor(
                                            cluster_result.reset_index(drop=True),
                                            hide_index=True,
                                            disabled=True,
                                            height=380,
                                            # width=290
                                        )
                                    else:
                                        st.info("Belum ada evaluasi bandwidth.")

                                # GRAFIK HASIL CLUSTERING DENGAN NILAI SILHOUETTE TERBAIK
                                with c2:
                                    st.write("##### Nilai Silhouette Terbaik")
                                    if best_result and X_pca is not None:
                                        fig, ax = plt.subplots(figsize=(4,3))
                                        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=best_result["labels"], s=16, alpha=0.85)
                                        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
                                        ax.set_title(f"{group_name}")
                                        uniq, counts = np.unique(best_result["labels"], return_counts=True)
                                        ax.legend(sc.legend_elements()[0], [f"C{c}: {n}" for c, n in zip(uniq, counts)], loc="best")
                                        st.pyplot(fig, width=720)
                                        plt.close(fig)
                                    else:
                                        st.info("Plot tidak tersedia.")

                                # TABEL DAFTAR PEMAIN TIAP CLUSTER
                                with c3:
                                    st.write("##### Daftar Pemain")
                                    if best_result:
                                        df_members = get_cluster_members_data(group_result, best_result)
                                        if isinstance(df_members, pd.DataFrame) and not df_members.empty:
                                            st.data_editor(
                                                df_members,
                                                hide_index=True,
                                                disabled=True,
                                                height=420,
                                            )
                                        else:
                                            st.info("Daftar pemain per cluster tidak ditemukan.")
                                    else:
                                        st.info("Nilai silhouette terbaik tidak ditemukan.")

                                # GRAFIK PERBANDINGAN STATISTIK TIAP CLUSTER
                                st.write("---")
                                st.write("##### Perbandingan Statistik")
                                try:
                                    features   = get_features_for_group(group_name, FEATURES_BY_POSITION)
                                    chart_data = get_cluster_feature_chart_data(group_result, features)
                                except BarDataMissing as e:
                                    st.info(f"Bar chart tidak dapat ditampilkan: {e}")
                                    continue

                                chart_data["Fitur"] = pd.Categorical(chart_data["Fitur"], categories=features, ordered=True)
                                clusters_order = sorted(chart_data["Cluster"].unique(), key=lambda s: int(s[1:]))
                                chart_data["Cluster"] = pd.Categorical(chart_data["Cluster"], categories=clusters_order, ordered=True)

                                c1, c2, c3 = st.columns(3, vertical_alignment="top")
                                
                                if len(features) > 0:
                                    columns= [c1, c2, c3]
                                    for i, fitur in enumerate(features):
                                        sub = chart_data[chart_data["Fitur"] == fitur]

                                        if sub.empty:
                                            with columns[i % 3]:
                                                st.empty()
                                        
                                        # BAR CHART
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
                                                # width=420,
                                                height=220,
                                            )
                                        )
                                        with columns[i % 3]:
                                            st.altair_chart(chart)
                                else:
                                    st.info("Tidak ada statistik yang ditemukan.")

                    # RESET HASIL CLUSTERING
                    if st.button("Reset Hasil Clustering"):
                        clear_recommend_state(st)
                        clear_cluster_state(st)
                        st.rerun()            

                    # DROPDOWN PILIH POSISI PEMAIN ACUAN
                    selected_position = st.selectbox("Pilih Posisi Pemain Acuan", POSITION_CHOICES, index=0)

                    if selected_position != "Pilih posisi pemain acuan":
                        st.session_state.setdefault("club_filter", "Semua")

                        # DROPDOWN PILIH KLUB PEMAIN ACUAN
                        club_options = ["Semua"] + get_clubs_by_season(selected_league, season=selected_season)                        
                        selected_club = st.selectbox("Pilih Klub Pemain Acuan", options=club_options, index=0)
                        st.session_state["club_filter"] = selected_club

                        if selected_season and selected_position:
                            if selected_club and selected_club != "Semua":
                                # BACA DATA PEMAIN DENGAN FILTER KLUB
                                players = get_players_by_season_and_club(selected_league, selected_season, selected_position, selected_club)
                            else:
                                # BACA DATA PEMAIN TANPA FILTER KLUB                        
                                players = get_players_by_season(selected_league, selected_season, selected_position)
                        else:
                            players = []
                        
                        # PILIH PEMAIN ACUAN
                        player_option = ["Pilih Pemain Acuan"] + (players if players else [])
                        selected_player = st.selectbox("Pilih Pemain Acuan", player_option, index=0)
                    else:
                        selected_player = None
                        selected_position = None

# =========================================================================================================================
                # UNTUK RESET STATE
                st.session_state.setdefault("prev_season", None)
                st.session_state.setdefault("prev_position", None)
                st.session_state.setdefault("prev_anchor", None)
                st.session_state.setdefault("prev_club_filter", None)

                season_changed   = (st.session_state["prev_season"]   != selected_season)
                position_changed = (st.session_state["prev_position"] != selected_position)
                anchor_changed   = (st.session_state["prev_anchor"]   != selected_player)
                club_changed     = (st.session_state["prev_club_filter"] != st.session_state.get("club_filter"))

                if season_changed:
                    clear_recommend_state(st)
                    clear_cluster_state(st)
                    if selected_season == "Pilih Musim":
                        st.session_state["prev_season"] = selected_season
                        st.session_state["prev_position"] = None
                        st.session_state["prev_anchor"] = None

                if position_changed:
                    clear_recommend_state(st)
                    if selected_position == "Pilih posisi pemain acuan":
                        st.session_state["prev_position"] = selected_position
                        st.session_state["prev_anchor"] = None

                if club_changed:
                    clear_recommend_state(st)
                    st.session_state["prev_anchor"] = None

                if anchor_changed:
                    clear_recommend_state(st)

                st.session_state["prev_season"]   = selected_season
                st.session_state["prev_position"] = selected_position
                st.session_state["prev_anchor"]   = selected_player
                st.session_state["prev_club_filter"] = st.session_state.get("club_filter")
# =========================================================================================================================
                
                # STATISTIK PEMAIN ACUAN
                if selected_season and selected_position and selected_player and selected_player != "Pilih Pemain Acuan":
                    detail = get_player_detail(selected_season, selected_player)

                    if detail:
                        st.markdown("---")
                        st.subheader("Tentang Pemain")
                        st.write(f"## {selected_player}")
                        col1, col2, _ = st.columns(3)

                        with col1:
                            st.metric(f"Team", detail.get('team'), border=True)
                            st.metric(f"Nationality", detail.get('nationality'), border=True)
                            st.metric(f"Position", detail.get('position'), border=True)

                        with col2:
                            st.metric(f"Age", detail.get('age'), border=True)
                            st.metric(f"Appearance", detail.get('appearance'), border=True)
                            st.metric(f"Total Minutes Played", detail.get('total_minute'), border=True)

                        st.markdown("---")
                        st.subheader("Statistik Pemain Acuan")
                        
                        col3, col4, col5 = st.columns(3)

                        with col3:
                            st.metric(f"Total Goal", detail.get('total_goal'), border=True)
                            st.metric(f"Total Assist", detail.get('assist'), border=True)
                            st.metric(f"Shot/Game", detail.get('shot_per_game'), border=True)
                            st.metric(f"Shot On Target/Game", detail.get('sot_per_game'), border=True)
                            st.metric(f"Successful Dribble/Game", detail.get('successful_dribble_per_game'), border=True)

                        with col4:
                            st.metric(f"Successful Pass/Game", detail.get('successful_pass_per_game'), border=True)
                            st.metric(f"Key Pass/Game", detail.get('key_pass_per_game'), border=True)
                            st.metric(f"Long Ball Pass/Game", detail.get('long_ball_per_game'), border=True)
                            st.metric(f"Successful Crossing/Game", detail.get('successful_crossing_per_game'), border=True)
                            st.metric(f"Ball Recovered/Game", detail.get('ball_recovered_per_game'), border=True)

                        with col5:
                            st.metric(f"Dribbled Past/Game", detail.get('dribbled_past_per_game'), border=True)
                            st.metric(f"Clearance/Game", detail.get('clearance_per_game'), border=True)
                            st.metric(f"Error Leading to Shot", detail.get('error'), border=True)
                            st.metric(f"Total Duel Won/Game", detail.get('total_duel_per_game'), border=True)
                            st.metric(f"Aerial Duel Won/Game", detail.get('aerial_duel_per_game'), border=True)

                        st.markdown("---")
                        st.subheader("Pemain Rekomendasi")
                        
                        st.session_state.setdefault("recommend_state", None)
                        st.session_state.setdefault("features", None)
                        st.session_state.setdefault("compare_recommend", None)

                        # JUMLAH PEMAIN REKOMENDASI
                        recommend_count = st.slider(
                            "Jumlah pemain rekomendasi",
                            min_value=1,
                            max_value=20,
                            step=1,
                            value=10
                        )

                        col7, col8, col9, col10 = st.columns(4)

                        # FILTER PEMAIN INDONESIA SAJA
                        with col7:
                            only_indo = st.checkbox("Pemain Indonesia saja", value=False)

                        # FILTER POSISI YANG SAMA DENGAN PEMAIN ACUAN SAJA
                        with col8:
                            filter_position = st.checkbox("Posisi yang sama saja", value=False)

                        # FILTER KLUB YG BERBEDA DENGAN PEMAIN ACUAN
                        with col9:
                            diff_club = st.checkbox("Klub yang berbeda saja", value=False)
                        
                        # BUTTON CARI PEMAIN REKOMENDASI
                        if recommend_count and st.button("Cari pemain rekomendasi"):
                            # MENGAMBIL DATA PEMAIN REKOMENDASI
                            recommend_players = get_recommend_similar_players(
                                result=results,                        
                                position_code=selected_position,
                                anchor_player=selected_player,
                                recommend_count=recommend_count,
                                only_indonesian=only_indo,
                                filter_position=filter_position,
                                diff_club=diff_club
                            )

                            # YANG DITAMPILKAN JIKA TIDAK ADA PEMAIN REKOMENDASI YANG DITEMUKAN
                            if recommend_players.empty:
                                st.info("Tidak ada pemain rekomendasi yang cocok untuk konfigurasi ini.")
                                st.session_state["recommend_state"] = None
                                st.session_state["features"] = None
                                st.session_state["compare_recommend"] = None
                            else:
                                st.session_state["recommend_state"] = recommend_players
                                st.session_state["features"] = get_player_features_data(selected_league, selected_season)
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

                            recommend_columns = ["player_name", "team", "position", "nationality", "similarity"]
                            recommend_column = [c for c in recommend_columns if c in recommend_state.columns]

                            for i, ds in recommend_state[recommend_column].iterrows():
                                col1, col2, col3, col4, col5, col6 = st.columns([3, 3, 2, 2, 2, 2])

                                similarity = float(ds.get("similarity", 0.0))
                                similarity = max(0.0, min(1.0, similarity)) * 100.0  

                                col1.write(ds.get("player_name", "-"))
                                col2.write(ds.get("team", "-"))
                                col3.write(ds.get("position", "-"))
                                col4.write(ds.get("nationality", "-"))
                                col5.write(f"{similarity:.2f}%")

                                # BUTTON BANDINGKAN PER PEMAIN
                                if col6.button("Bandingkan", key=f"compare_{i}_{ds.get('player_name','')}"):
                                    st.session_state["compare_recommend"] = ds['player_name']
                            
                            # PERBANDINGAN STATISTIK PEMAIN ACUAN DENGAN PEMAIN REKOMENDASI
                            target_player = st.session_state["compare_recommend"]
                            if target_player and features is not None:
                                with st.expander(f"Perbandingan {selected_player} dengan {target_player}", expanded=True):                                
                                    try:
                                        # MENGAMBIL DATA PERBANDINGAN STATISTIK
                                        chart_data = get_comparison_chart_data(
                                            features=features,
                                            anchor_player=selected_player,
                                            target_player=target_player,
                                            features_to_compare=FEATURES_TO_COMPARE
                                        )
                                    except ValueError as e:
                                        st.error(str(e))
                                    else:                               
                                        # MENAMPILKAN BAR CHART PERBANDINGAN STATISTIK
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
    
    return st