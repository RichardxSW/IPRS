import math
import re
import altair as alt
import numpy as np
import pandas as pd
from players.clustering import FEATURE_LABELS, POSITION_GROUPS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# FITUR YANG AKAN DITAMPILKAN DALAM HASIL PERBANDINGAN
FEATURES_TO_COMPARE = [
    "total_goal", "assist", "shot_per_game", "sot_per_game", 
    "total_duel_per_game", "aerial_duel_per_game",
    "successful_dribble_per_game", "key_pass_per_game",
    "successful_pass_per_game", "long_ball_per_game", "successful_crossing_per_game",
    "ball_recovered_per_game", "dribbled_past_per_game", "clearance_per_game",
    "error", 
]

# MENGKATEGORIKAN POSISI KE PENYERANG, GELANDANG ATAU BERTAHAN
def group_for_position(position_code: str) -> str | None:
    position = str(position_code).upper().strip()
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return None

# FORMAT POSISI 
def format_position(position_str) -> set[str]:    
    if position_str is None:
        return set()
    if isinstance(position_str, float) and math.isnan(position_str):
        return set()
    string = str(position_str).upper()
    position = [i for i in re.split(r"[^A-Z]+", string) if i]
    return set(position)

# FILTER POSISI
def matches_position(position_str: str, anchor_position: set[str]) -> bool:
    positions = format_position(position_str)
    return bool(positions & anchor_position)

# MENCARI PEMAIN REKOMENDASI DAN MENGHITUNG COSINE SIMILARITY
def get_recommend_similar_players(
    result,
    position_code: str,
    anchor_player: str,
    recommend_count: int = 10,
    only_indonesian: bool = False,
    filter_position: bool = False,
    diff_club: bool = False
):
    group = group_for_position(position_code)

    if not group:
        raise ValueError("Posisi tidak valid.")

    results = result.get(group)

    if not results or not results.get("best_silhouette"):
        return pd.DataFrame()

    labels = results["best_silhouette"]["labels"]
    X_scaled = results["X_scaled"]
    players = results["players"].copy()

    # PEMAIN ACUAN
    anchor = players["player_name"].str.lower() == str(anchor_player).lower()

    if not anchor.any():
        return pd.DataFrame()

    anchor_player = int(players[anchor].index[0])
    anchor_cluster = int(labels[anchor_player])
    anchor_position_str = players.loc[anchor_player, "position"]
    anchor_position = format_position(anchor_position_str)

    # ambil pemain dalam cluster yang sama dengan pemain acuan
    same_cluster = np.where(labels == anchor_cluster)[0]
    if same_cluster.size <= 1:
        return pd.DataFrame()

    # filter Pemain Indonesia saja
    if "nationality" in players.columns and only_indonesian:
        same_cluster = [i for i in same_cluster if str(players.loc[i, "nationality"]).strip().lower() == "indonesia"]
        if len(same_cluster) <= 1:
            return pd.DataFrame()

    # FILTER POSISI YANG SAMA DENGAN PEMAIN ACUAN
    if "position" in players.columns and filter_position:
        same_cluster = [j for j in same_cluster if matches_position(players.loc[j, "position"], anchor_position)]

    # DATA PEMAIN ACUAN
    anchor_data = X_scaled[anchor_player:anchor_player+1]

    # DATA PEMAIN DALAM CLUSTER YANG SAMA
    same_cluster_data = X_scaled[same_cluster]

    # hitung cosine similarity antara pemain acuan dan pemain dalam cluster yg sama
    cos_sim = cosine_similarity(anchor_data, same_cluster_data).ravel()

    recommend_result = players.iloc[same_cluster].copy()
    recommend_result["similarity"] = cos_sim

    # MEMBUANG PEMAIN ACUAN
    recommend_result = recommend_result[recommend_result.index != anchor_player]

    # FILTER KLUB
    if diff_club and "team" in players.columns:
        anchor_team = str(players.loc[anchor_player, "team"]).strip().lower()
        recommend_result = recommend_result[recommend_result["team"].str.strip().str.lower() != anchor_team]
        
    # SORT DARI NILAI COSINE SIM PALING GEDE DAN NAMPILIN SESUAI JUMLAH YANG DIINPUT        
    recommend_result = recommend_result.sort_values("similarity", ascending=False).head(recommend_count)

    return recommend_result.reset_index(drop=True)

# ==================================================================================================================================

def build_comparison_section(
    st,
    features_df: pd.DataFrame,
    anchor_player: str,
    target_player: str,
    features_to_compare: list[str],
):
    cols = ["player_name", *features_to_compare]
    anchor_data = features_df.loc[features_df["player_name"] == anchor_player, cols]
    target_data = features_df.loc[features_df["player_name"] == target_player, cols]

    if anchor_data.empty or target_data.empty:
        raise ValueError("Data pemain tidak ditemukan.")

    anchor_data = anchor_data.iloc[0]
    target_data = target_data.iloc[0]

    label = [FEATURE_LABELS.get(f, f) for f in features_to_compare]

    rows = []
    for feature in features_to_compare:
        value_anchor = float(anchor_data[feature])
        value_target = float(target_data[feature])
        diff_raw  = value_target - value_anchor
        
        max_value = max(abs(value_anchor), abs(value_target), 1e-6)
        diff_scaled = np.clip(diff_raw / max_value, -1.0, 1.0)

        rows.append({
            "Statistik": FEATURE_LABELS.get(feature, feature),
            "Nilai Pemain Acuan": value_anchor,
            "Nilai Pemain Rekomendasi": value_target,
            "Selisih": diff_raw,
            "Bar Kiri": (-abs(diff_scaled)) if diff_scaled < 0 else 0.0,
            "Bar Kanan": (abs(diff_scaled)) if diff_scaled > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    dom = [-1, 1]
    bar_height = 32 * len(features_to_compare)

    ax_none_x = alt.Axis(title=None, labels=False, ticks=False, domain=False)
    ax_none_y = alt.Axis(title=None, labels=False, ticks=False, domain=False)

    # ===== Kolom 1: nama statistik =====
    stats = (
        alt.Chart(df)
        .mark_text(
            align="center", 
            dx=0, 
            fontWeight="bold",
            fontSize=16,
            tooltip=False
        )
        .encode(
            y=alt.Y("Statistik:N", sort=label, axis=ax_none_y),
            text="Statistik:N",
            color=alt.value("white"),            
        )
        .properties(
            height=bar_height
        )
    )

    # ===== Kolom 2: bar chart
    bg = (
        alt.Chart(df)
        .mark_bar(color="#eeeeee", tooltip=False)
        .encode(
            y=alt.Y("Statistik:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("bg_min:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            x2="bg_max:Q",
        )
        .transform_calculate(bg_min="-1", bg_max="1")
    )

    mid = (
        alt.Chart(df)
        .mark_rule(color="#222", strokeWidth=1)
        .encode(x=alt.X("zero:Q", scale=alt.Scale(domain=dom), axis=ax_none_x))
        .transform_calculate(zero="0")
    )

    left_bar = (
        alt.Chart(df)
        .mark_bar(color="#f59e0b", tooltip=False)
        .encode(
            y=alt.Y("Statistik:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("zero:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            x2="Bar Kiri:Q",
        )
        .transform_calculate(zero="0")
    )

    right_bar = (
        alt.Chart(df)
        .mark_bar(color="#3b82f6", tooltip=False)
        .encode(
            y=alt.Y("Statistik:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("zero:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            x2="Bar Kanan:Q",
        )
        .transform_calculate(zero="0")
    )

    left_txt = (
        alt.Chart(df)
        .mark_text(align="right", dx=-8, fontWeight="bold", fontSize=16, tooltip=False)
        .encode(
            y=alt.Y("Statistik:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("xpos:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            text=alt.Text("Nilai Pemain Acuan:Q", format=".1f"),
            color=alt.value("#f9fafb"),
        )
        .transform_calculate(xpos="-1")
    )

    right_txt = (
        alt.Chart(df)
        .mark_text(align="left", dx=8, fontWeight="bold", fontSize=16, tooltip=False)
        .encode(
            y=alt.Y("Statistik:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("xpos:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            text=alt.Text("Nilai Pemain Rekomendasi:Q", format=".1f"),
            color=alt.value("#f9fafb"),
        )
        .transform_calculate(xpos="1")
    )

    bars_chart = (bg + left_bar + right_bar + mid + left_txt + right_txt).properties(
        height=bar_height
    )

    # ===== Kolom 3: Nilai Selisih
    diff_value = (
        alt.Chart(df)
        .transform_calculate(
            diff_label=(
                "datum.Selisih == 0 ? '-' : format(abs(datum.Selisih), '.1f')"
            ),
            diff_color=(
                "datum.Selisih == 0 ? '#FFFFFF' : " #PUTIH
                "datum.Selisih > 0 ? '#3B82F6' : '#F59E0B'" #KIRI ORANGE, KANAN BIRU
            )
        )
        .mark_text(align="center", dx=0, fontWeight="bold", fontSize=16, tooltip=False)
        .encode(
            y=alt.Y("Statistik:N", sort=features_to_compare, axis=ax_none_y),
            text="diff_label:N",
            color=alt.Color("diff_color:N", scale=None)
            
        )
        .properties(height=bar_height)
    )

    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.altair_chart(stats)
    with col2:
        st.altair_chart(bars_chart)
    with col3:
        st.altair_chart(diff_value)
    return st