import math
import re
import numpy as np
import pandas as pd
from players.clustering import POSITION_GROUPS, run_meanshift_by_position
from sklearn.metrics.pairwise import cosine_similarity

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
def _group_for_position(position_code: str) -> str | None:
    position = str(position_code).upper().strip()
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return None

# Ubah posisi jadi (huruf besar, spasi, /, -, koma).
def _format_position(position_str) -> set[str]:    
    if position_str is None:
        return set()
    if isinstance(position_str, float) and math.isnan(position_str):
        return set()
    string = str(position_str).upper()
    position = [i for i in re.split(r"[^A-Z]+", string) if i]
    return set(position)

# FILTER POSISI
def _matches_position(position_str: str, anchor_position: set[str]) -> bool:
    positions = _format_position(position_str)
    return bool(positions & anchor_position)

# MENCARI PEMAIN REKOMENDASI DAN MENGHITUNG COSINE SIMILARITY
def get_recommend_similar_players(
    season: str,
    position_code: str,
    anchor_player: str,
    recommend_count: int = 10,
    only_indonesian: bool = False,
    filter_position: bool = False,
    diff_club: bool = False
):
    group = _group_for_position(position_code)
    if not group:
        raise ValueError("Kode posisi tidak valid.")

    all_results = run_meanshift_by_position(season)
    results = all_results.get(group)
    if not results or not results.get("best_silhouette"):
        return pd.DataFrame()

    labels = results["best_silhouette"]["labels"]
    X_scaled = results["X_scaled"]
    players = results["players"].copy()

    # cari pemain acuan
    anchor = players["player"].str.lower() == str(anchor_player).lower()
    if not anchor.any():
        return pd.DataFrame()

    anchor_idx = int(players[anchor].index[0])
    anchor_cluster = int(labels[anchor_idx])
    anchor_position_str = players.loc[anchor_idx, "position"]
    anchor_position = _format_position(anchor_position_str)

    # ambil hanya pemain dalam cluster yang sama
    same_idx = np.where(labels == anchor_cluster)[0]
    if same_idx.size <= 1:
        return pd.DataFrame()

    # filter Pemain Indonesia saja
    if "nationality" in players.columns and only_indonesian:
        same_idx = [i for i in same_idx if str(players.loc[i, "nationality"]).strip().lower() == "indonesia"]
        if len(same_idx) <= 1:
            return pd.DataFrame()

    # FILTER POSISI YANG SAMA DENGAN PEMAIN ACUAN
    if "position" in players.columns and filter_position:
        same_idx = [j for j in same_idx if _matches_position(players.loc[j, "position"], anchor_position)]


    # hitung cosine similarity antara pemain acuan dan pemain dalam cluster yg sama
    anchor_vec = X_scaled[anchor_idx:anchor_idx+1]
    cluster_vecs = X_scaled[same_idx]
    sims = cosine_similarity(anchor_vec, cluster_vecs).ravel()

    out = players.iloc[same_idx].copy()
    out["similarity"] = sims

    # MEMBUANG PEMAIN ACUAN
    out = out[out.index != anchor_idx]

    # FILTER KLUB
    if diff_club and "team" in players.columns:
        anchor_team = str(players.loc[anchor_idx, "team"]).strip().lower()
        out = out[out["team"].str.strip().str.lower() != anchor_team]
        
    out = out.sort_values("similarity", ascending=False).head(recommend_count)

    return out.reset_index(drop=True)

# MEMBACA FITUR UNTUK YANG DIPAKAI UNTUK PEMAIN ACUAN DAN PEMAIN REKOMENDASI
def get_feature_rows(feat_df: pd.DataFrame, anchor_player: str, target_player: str, features: list[str]) -> tuple[pd.Series, pd.Series]:
    position_features = ["player", *features]
    anchor = feat_df.loc[feat_df["player"] == anchor_player, position_features]
    recommend = feat_df.loc[feat_df["player"] == target_player, position_features]
    if anchor.empty:
        raise ValueError(f"Data fitur pemain acuan '{anchor_player}' tidak ditemukan.")
    if recommend.empty:
        raise ValueError(f"Data fitur pemain rekomendasi '{target_player}' tidak ditemukan.")
    return anchor.iloc[0], recommend.iloc[0]

# ==================================================================================================================================
# UNTUK MEMBUAT CHART PERBANDINGAN
def build_long_compare_df(anchor_row: pd.Series, recommend_row: pd.Series, features: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "Fitur": features,
        anchor_row["player"]: [anchor_row[f] for f in features],
        recommend_row["player"]: [recommend_row[f] for f in features],
    })
    chart_data = df.melt(id_vars="Fitur", var_name="Pemain", value_name="Nilai")
    return chart_data

def prepare_comparison_chart_data(features: pd.DataFrame, anchor_player: str, target_player: str, features_to_compare: list[str]) -> pd.DataFrame:
    anchor_row, recommend_row = get_feature_rows(features, anchor_player, target_player, features_to_compare)
    return build_long_compare_df(anchor_row, recommend_row, features_to_compare)

# ==================================================================================================================================