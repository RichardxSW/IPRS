import math
import re
import numpy as np
import pandas as pd
from players.clustering import POSITION_GROUPS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# FITUR YANG AKAN DITAMPILKAN DALAM HASIL PERBANDINGAN
FEATURES_TO_COMPARE = [
    "age", "appearance", "total_minute",
    "total_goal", "assist", "shot_per_game",
    "sot_per_game", "successful_dribble_per_game", "key_pass_per_game",
    "successful_pass_per_game", "long_ball_per_game", "successful_crossing_per_game",
    "ball_recovered_per_game", "dribbled_past_per_game", "clearance_per_game",
    "error", "total_duel_per_game", "aerial_duel_per_game"
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