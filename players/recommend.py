import math
import re
import numpy as np
import pandas as pd
from players.clustering import POS_GROUPS, run_meanshift_by_position
from sklearn.metrics.pairwise import cosine_similarity


FEATURES_TO_COMPARE = [
    # isi sesuai kolom fitur kamu, contoh:
    "age", "appearance", "total_minute",
    "total_goal", "assist", "shot_per_game",
    "sot_per_game", "successful_dribble_per_game", "key_pass_per_game",
    "successful_pass_per_game", "long_ball_per_game", "successful_crossing_per_game",
    "ball_recovered_per_game", "dribbled_past_per_game", "clearance_per_game",
    "error", "total_duel_per_game", "aerial_duel_per_game"
    # "tackles_p90", "interceptions_p90", "carries_p90"
]

def _group_for_position(pos_code: str) -> str | None:
    p = str(pos_code).upper().strip()
    for g, arr in POS_GROUPS.items():
        if p in arr:
            return g
    return None

def _pos_tokens(pos_str) -> set[str]:
    """Ubah string posisi jadi set token huruf besar (tahan spasi, /, -, koma, dll)."""
    if pos_str is None:
        return set()
    if isinstance(pos_str, float) and math.isnan(pos_str):
        return set()
    s = str(pos_str).upper()
    tokens = [t for t in re.split(r"[^A-Z]+", s) if t]
    return set(tokens)

def _matches_position(pos_str: str, anchor_tokens: set[str]) -> bool:
    """True kalau ada minimal satu token posisi yang sama dengan anchor."""
    tokens = _pos_tokens(pos_str)
    return bool(tokens & anchor_tokens)

def get_recommend_similar_players(
    season: str,
    position_code: str,
    anchor_player: str,
    top_n: int = 5,
    only_indonesian: bool = False,
    filter_position: bool = False
):
    """
    Rekomendasi berbasis cosine:
    - pakai grup sesuai position_code,
    - pakai konfigurasi cluster dengan silhouette terbaik,
    - top-N dari cluster yang sama dengan anchor.
    """
    group = _group_for_position(position_code)
    if not group:
        raise ValueError("Kode posisi tidak valid.")

    all_results = run_meanshift_by_position(season)
    res = all_results.get(group)
    if not res or not res.get("best_sil"):
        return pd.DataFrame()

    labels = res["best_sil"]["labels"]
    Xs = res["Xs"]
    meta = res["meta"].copy()  # id, player, team, position, nationality, ...

    # cari pemain acuan
    anchor_mask = meta["player"].str.lower() == str(anchor_player).lower()
    if not anchor_mask.any():
        return pd.DataFrame()

    anchor_idx = int(meta[anchor_mask].index[0])
    anchor_cluster = int(labels[anchor_idx])
    anchor_pos_str = meta.loc[anchor_idx, "position"]
    anchor_tokens = _pos_tokens(anchor_pos_str)

    # ambil hanya pemain dalam cluster yang sama
    same_idx = np.where(labels == anchor_cluster)[0]
    if same_idx.size <= 1:
        return pd.DataFrame()  # cluster cuma anchor sendiri

    # filter Pemain Indonesia saja
    if "nationality" in meta.columns and only_indonesian:
        same_idx = [i for i in same_idx if str(meta.loc[i, "nationality"]).strip().lower() == "indonesia"]
        if len(same_idx) <= 1:
            return pd.DataFrame()
        
    # if "position" in meta.columns and filter_position:
    #     same_idx = [j for j in same_idx if str(meta.loc[j, "position"]).strip() == position_code]

    if "position" in meta.columns and filter_position:
        same_idx = [j for j in same_idx if _matches_position(meta.loc[j, "position"], anchor_tokens)]

    # hitung cosine similarity antara pemain acuan dan pemain dalam cluster yg sama
    anchor_vec = Xs[anchor_idx:anchor_idx+1]
    cluster_vecs = Xs[same_idx]
    sims = cosine_similarity(anchor_vec, cluster_vecs).ravel()

    out = meta.iloc[same_idx].copy()
    out["similarity"] = sims
    # buang pemain acuan
    out = out[out.index != anchor_idx]
    out = out.sort_values("similarity", ascending=False).head(top_n)

    return out.reset_index(drop=True)

def get_feature_rows(feat_df: pd.DataFrame, anchor_player: str, target_player: str, features: list[str]) -> tuple[pd.Series, pd.Series]:
    """
    Ambil baris fitur untuk anchor & target. Raise ValueError jika tidak ada.
    """
    cols = ["player", *features]
    a = feat_df.loc[feat_df["player"] == anchor_player, cols]
    t = feat_df.loc[feat_df["player"] == target_player, cols]
    if a.empty:
        raise ValueError(f"Data fitur anchor '{anchor_player}' tidak ditemukan.")
    if t.empty:
        raise ValueError(f"Data fitur '{target_player}' tidak ditemukan.")
    return a.iloc[0], t.iloc[0]

def build_long_compare_df(anchor_row: pd.Series, target_row: pd.Series, features: list[str]) -> pd.DataFrame:
    """
    Kembalikan long-form dataframe: kolom = Fitur, Pemain, Nilai.
    Cocok untuk di-plot grouped bar (Anchor vs Target) per fitur.
    """
    df = pd.DataFrame({
        "Fitur": features,
        anchor_row["player"]: [anchor_row[f] for f in features],
        target_row["player"]: [target_row[f] for f in features],
    })
    long_df = df.melt(id_vars="Fitur", var_name="Pemain", value_name="Nilai")
    return long_df

def prepare_comparison_long_df(feat_df: pd.DataFrame, anchor_player: str, target_player: str, features: list[str]) -> pd.DataFrame:
    """
    Satu pintu: ambil baris → bentuk long-form → kembalikan ke UI.
    """
    a_row, t_row = get_feature_rows(feat_df, anchor_player, target_player, features)
    return build_long_compare_df(a_row, t_row, features)