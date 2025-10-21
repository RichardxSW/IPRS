# players/services.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .models import Player

# =============================
# DEFINISI POSISI & FITUR
# =============================
POS_GROUPS = {
    "Forward": ["ST", "LW", "RW"],
    "Midfielder": ["AM", "CM", "DM", "LM", "RM"],
    "Defender": ["CB", "LB", "RB"],
}

FEATURES_BY_POS = {
    "Forward": [
        "goal_per_game", "shot_per_game", "sot_per_game",
        "assist_per_game", "successful_dribble_per_game", "successful_crossing_per_game",
        "key_pass_per_game", "total_duel_per_game", "aerial_duel_per_game"
    ],
    "Midfielder": [
        "shot_per_game", "sot_per_game", "assist_per_game", "key_pass_per_game", "successful_pass_per_game",
        "long_ball_per_game", "successful_dribble_per_game",
        "ball_recovered_per_game", "total_duel_per_game", "dribbled_past_per_game", "clearance_per_game"
    ],
    "Defender": [
        "clearance_per_game", "ball_recovered_per_game",
        "dribbled_past_per_game", "successful_dribble_per_game",
        "long_ball_per_game", "aerial_duel_per_game", "total_duel_per_game"
    ],
}

META_COLS = [
    "id", "player", "team", "position", "nationality",
    "age", "appearance", "total_minute",
    "total_goal", "assist", "error"
]

# =============================
# UTILITAS
# =============================
def get_player_features_df(season: str) -> pd.DataFrame:
    all_feats = sorted({f for feats in FEATURES_BY_POS.values() for f in feats})
    qs = (
        Player.objects
        .filter(dataset__season=season)
        .values(*META_COLS, *all_feats)
        .order_by("player")
    )
    return pd.DataFrame(list(qs))

def _prepare_matrix(df: pd.DataFrame, feat_cols):
    X = (
        df[feat_cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xs)
    return Xs, X2

# =============================
# CLUSTERING
# =============================
def run_meanshift(df: pd.DataFrame, feat_cols):
    """Loop bandwidth 0.5–10 dengan error handling."""
    Xs, X2 = _prepare_matrix(df, feat_cols)
    bandwidths = np.arange(0.5, 5.5, 0.5)
    results = []

    for bw in bandwidths:
        labels, n_clusters, = None, 0
        for bin_seed in (True, False):
            try:
                ms = MeanShift(bandwidth=float(bw), bin_seeding=bin_seed, cluster_all=True)
                labels = ms.fit_predict(Xs)
                n_clusters = len(np.unique(labels))
                break
            except ValueError:
                raise Exception("Clustering gagal")

        sil, dbi = None, None
        if labels is not None and n_clusters >= 2:
            try:
                sil = float(silhouette_score(Xs, labels))
            except Exception:
                pass
            try:
                dbi = float(davies_bouldin_score(Xs, labels))
            except Exception:
                pass

        results.append({
            "bw": float(bw),
            "labels": labels,
            "n_clusters": n_clusters,
            "sil": sil,
            "dbi": dbi,            
        })

    df_eval = pd.DataFrame([{
        "Bandwidth": r["bw"],
        "Jumlah Cluster": r["n_clusters"],
        "Silhouette": r["sil"],
        "DBI": r["dbi"],        
    } for r in results])

    valid_sil = [r for r in results if r["sil"] is not None]
    valid_dbi = [r for r in results if r["dbi"] is not None]
    best_sil = max(valid_sil, key=lambda r: r["sil"]) if valid_sil else None
    best_dbi = min(valid_dbi, key=lambda r: r["dbi"]) if valid_dbi else None
    same_bw = best_sil and best_dbi and best_sil["bw"] == best_dbi["bw"]

    return {
        "res_table": df_eval,
        "emb2d": X2,
        "Xs": Xs,
        "meta": df.reset_index(drop=True),
        "best_sil": best_sil,
        "best_dbi": best_dbi,
        "same_bw": same_bw,
    }

def run_meanshift_by_position(season: str):
    """
    Jalankan clustering per kategori posisi
    dengan fitur yang disesuaikan untuk tiap kategori.
    """
    df_all = get_player_features_df(season)
    if df_all.empty:
        return {"Forward": None, "Midfielder": None, "Defender": None}

    results = {}
    for group, positions in POS_GROUPS.items():
        df_pos = df_all[df_all["position"].apply(
            lambda p: any(pos in str(p).upper().replace(" ", "").split(",") or
                          pos in str(p).upper().replace(" ", "") for pos in positions)
        )]
        feat_cols = FEATURES_BY_POS[group]
        if len(df_pos) < 3:
            results[group] = None
            continue
        results[group] = run_meanshift(df_pos, feat_cols)

    return results
