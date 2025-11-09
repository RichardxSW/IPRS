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
    # "age", "appearance", "total_minute",
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

# MEMBACA FITUR UNTUK YANG DIPAKAI UNTUK PEMAIN ACUAN DAN PEMAIN REKOMENDASI
def get_feature_data(feat_df: pd.DataFrame, anchor_player: str, target_player: str, features: list[str]) -> tuple[pd.Series, pd.Series]:
    position_features = ["player_name", *features]
    anchor = feat_df.loc[feat_df["player_name"] == anchor_player, position_features]
    recommend = feat_df.loc[feat_df["player_name"] == target_player, position_features]
    if anchor.empty:
        raise ValueError(f"Data fitur pemain acuan '{anchor_player}' tidak ditemukan.")
    if recommend.empty:
        raise ValueError(f"Data fitur pemain rekomendasi '{target_player}' tidak ditemukan.")
    return anchor.iloc[0], recommend.iloc[0]

# ==================================================================================================================================
# UNTUK MEMBUAT CHART PERBANDINGAN
def get_comparison_data(anchor_row: pd.Series, recommend_row: pd.Series, features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "Fitur": features,
        anchor_row["player_name"]: [anchor_row[f] for f in features],
        recommend_row["player_name"]: [recommend_row[f] for f in features],
    })
    chart_data = df.melt(id_vars="Fitur", var_name="Pemain", value_name="Nilai")
    return chart_data

def get_comparison_chart_data(features: pd.DataFrame, anchor_player: str, target_player: str, features_to_compare: list[str]) -> pd.DataFrame:
    anchor_row, recommend_row = get_feature_data(features, anchor_player, target_player, features_to_compare)
    return get_comparison_data(anchor_row, recommend_row, features_to_compare)

# ==================================================================================================================================

def _feature_scale_per_feature(df: pd.DataFrame, features: list[str], q=0.98):
    scale = {}
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        vmin, vmax = np.nanmin(s), np.nanquantile(s, q)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = np.nanmax(s)
        scale[f] = max(vmax - vmin, 1e-6)
    return scale

def fm_compare_hconcat(
    st,
    features_df: pd.DataFrame,
    anchor_player: str,
    target_player: str,
    features_to_compare: list[str],
):
    cols = ["player_name", *features_to_compare]
    A = features_df.loc[features_df["player_name"] == anchor_player, cols]
    B = features_df.loc[features_df["player_name"] == target_player, cols]

    if A.empty or B.empty:
        raise ValueError("Data pemain tidak ditemukan.")

    A = A.iloc[0]
    B = B.iloc[0]

    # scale = _feature_scale_per_feature(features_df, features_to_compare, q=0.98)

    label = [FEATURE_LABELS.get(f, f) for f in features_to_compare]

    rows = []
    for f in features_to_compare:
        va = float(A[f])
        vb = float(B[f])
        d_raw  = vb - va

        
        vmax = max(abs(va), abs(vb), 1e-6)
        d_norm = np.clip(d_raw / vmax, -1.0, 1.0)

        rows.append({
            "Fitur": FEATURE_LABELS.get(f, f),
            "val_left": va,
            "val_right": vb,
            "diff_raw": d_raw,
            "left_fill": (-abs(d_norm)) if d_norm < 0 else 0.0,
            "right_fill": (abs(d_norm)) if d_norm > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    dom = [-1, 1]
    bar_h = 32 * len(features_to_compare)

    ax_none_x = alt.Axis(title=None, labels=False, ticks=False, domain=False)
    ax_none_y = alt.Axis(title=None, labels=False, ticks=False, domain=False)

    # ===== Kolom 1: nama statistik =====
    names_chart = (
        alt.Chart(df)
        .mark_text(
            align="center", 
            dx=0, 
            fontWeight="bold",
            fontSize=16
        )
        .encode(
            y=alt.Y("Fitur:N", sort=label, axis=ax_none_y),
            text="Fitur:N",
            color=alt.value("white"),
        )
        .properties(
            height=bar_h
        )
    )

    # ===== Kolom 2: bar chart
    bg = (
        alt.Chart(df)
        .mark_bar(color="#eeeeee")
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
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
        .mark_bar(color="#f59e0b")
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("zero:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            x2="left_fill:Q",
        )
        .transform_calculate(zero="0")
    )

    right_bar = (
        alt.Chart(df)
        .mark_bar(color="#3b82f6")
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("zero:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            x2="right_fill:Q",
        )
        .transform_calculate(zero="0")
    )

    left_txt = (
        alt.Chart(df)
        .mark_text(align="right", dx=-8, fontWeight="bold", fontSize=16)
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("xpos:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            text=alt.Text("val_left:Q", format=".1f"),
            color=alt.value("#f9fafb"),
        )
        .transform_calculate(xpos="-1.05")
    )

    right_txt = (
        alt.Chart(df)
        .mark_text(align="left", dx=8, fontWeight="bold", fontSize=16)
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("xpos:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            text=alt.Text("val_right:Q", format=".1f"),
            color=alt.value("#f9fafb"),
        )
        .transform_calculate(xpos="1.05")
    )

    bars_chart = (bg + left_bar + right_bar + mid + left_txt + right_txt).properties(
        height=bar_h
    )

    # ===== Kolom 3: Nilai Selisih
    delta_chart = (
        alt.Chart(df)
        .transform_calculate(
            diff_label=(
                "datum.diff_raw == 0 ? '-' : format(abs(datum.diff_raw), '.1f')"
            ),
            diff_color=(
                "datum.diff_raw == 0 ? '#FFFFFF' : "
                "datum.diff_raw > 0 ? '#3B82F6' : '#F59E0B'"
            )
        )
        .mark_text(align="center", dx=0, fontWeight="bold", fontSize=16)
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            text="diff_label:N",
            color=alt.Color("diff_color:N", scale=None)
            
        )
        .properties(height=bar_h)
    )

    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.altair_chart(names_chart)
    with col2:
        st.altair_chart(bars_chart)
    with col3:
        st.altair_chart(delta_chart)
    return st