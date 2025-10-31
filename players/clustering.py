import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .models import Player

# =============================
# FITUR YANG TERPAKAI UNTUK CLUSTERING
# =============================
POSITION_GROUPS = {
    "Pemain Penyerang": ["ST", "LW", "RW"],
    "Pemain Gelandang": ["AM", "CM", "DM", "LM", "RM"],
    "Pemain Bertahan": ["CB", "LB", "RB"],
}

FEATURES_BY_POSITION = {
    "Pemain Penyerang": [
        "goal_per_game", "shot_per_game", "sot_per_game",
        "assist_per_game", "successful_dribble_per_game", "successful_crossing_per_game",
        "key_pass_per_game", "total_duel_per_game", "aerial_duel_per_game"
    ],
    "Pemain Gelandang": [
        "goal_per_game", "shot_per_game", "sot_per_game", 
        "assist_per_game", "key_pass_per_game", "successful_pass_per_game",
        "long_ball_per_game", "successful_dribble_per_game", "aerial_duel_per_game", "successful_crossing_per_game",
        "ball_recovered_per_game", "total_duel_per_game", "dribbled_past_per_game", "clearance_per_game"
    ],
    "Pemain Bertahan": [
        "clearance_per_game", "ball_recovered_per_game",
        "dribbled_past_per_game", "successful_dribble_per_game",
        "aerial_duel_per_game", "total_duel_per_game"
    ],
}

# FITUR YANG TIDAK TERPAKAI UNTUK CLUSTERING
META_COLUMNS = [
    "id", "player", "team", "position", "nationality",
    "age", "appearance", "total_minute",
    "total_goal", "assist", "error"
]

# LABEL FITUR UNTUK DITAMPILKAN 
FEATURE_LABELS = {
    "age": "Age",
    "appearance": "Appearances",
    "total_minute": "Total Minutes Played",
    "clearance_per_game": "Clearance/Game",
    "ball_recovered_per_game": "Ball Recovery/Game",    
    "aerial_duel_per_game": "Aerial Duel Won/Game",
    "total_duel_per_game": "Total Duel Won/Game",
    "successful_pass_per_game": "Successful Pass/Game",
    "long_ball_per_game": "Long Pass/Game",
    "dribbled_past_per_game": "Dribbled Past/Game",    
    "successful_dribble_per_game": "Successful Dribble/Game",
    "total_goal": "Goals",
    "goal_per_game": "Goal/Game",
    "assist_per_game": "Assist/Game",
    "assist": "Assists",
    "shot_per_game": "Shot/Game",
    "sot_per_game": "Shot on Target/Game",
    "key_pass_per_game": "Key Pass/Game",
    "successful_crossing_per_game": "Successful Crossing/Game",
    "error": "Error Leading to Shot"
}

# =============================
# MENGAMBIL DATA FITUR FITUR PEMAIN
# =============================
def get_player_features_df(season: str) -> pd.DataFrame:
    all_feature = sorted({f for feats in FEATURES_BY_POSITION.values() for f in feats})
    qs = (
        Player.objects
        .filter(season__season=season)
        .values(*META_COLUMNS, *all_feature)
        .order_by("player")
    )
    return pd.DataFrame(list(qs))

# PREPROCESSING DATA
def _prepare_matrix(df: pd.DataFrame, feat_cols):
    X = (
        df[feat_cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )

    X_scaled = StandardScaler().fit_transform(X)    
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    return X_scaled, X_pca

# =============================
# MEAN SHIFT CLUSTERING
# =============================
def run_meanshift(df: pd.DataFrame, features):
    #PREPROSES
    X_scaled, X_pca = _prepare_matrix(df, features)

    # BANDWIDTH
    bandwidths = np.arange(0.5, 5.5, 0.5)

    results = []

    #LOOP MEANSHIFT 
    for bw in bandwidths:
        labels, n_clusters, = None, 0        
        for bin_seed in [True]:
            try:
                ms = MeanShift(bandwidth=float(bw), bin_seeding=bin_seed, cluster_all=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    labels = ms.fit_predict(X_scaled)
                # JUMLAH CLUSTER
                n_clusters = len(np.unique(labels))
                break
            except ValueError:
                raise Exception("Clustering gagal")

        # EVALUASI
        silhouette, dbi = None, None
        if labels is not None and n_clusters >= 2:
            try:
                # NILAI SILHOUTTE
                silhouette = float(silhouette_score(X_pca, labels))
            except Exception:
                pass
            try:
                # NILAI DBI
                dbi = float(davies_bouldin_score(X_scaled, labels))
            except Exception:
                pass

        results.append({
            "bw": float(bw),
            "labels": labels,
            "n_clusters": n_clusters,
            "silhouette": silhouette,
            "dbi": dbi,            
        })

    cluster_result = pd.DataFrame([{
        "Bandwidth": r["bw"],
        "Jumlah Cluster": r["n_clusters"],
        "Silhouette": r["silhouette"],
        "DBI": r["dbi"],        
    } for r in results])

    valid_silhouette = [r for r in results if r["silhouette"] is not None]
    valid_dbi = [r for r in results if r["dbi"] is not None]
    best_silhouette = max(valid_silhouette, key=lambda r: r["silhouette"]) if valid_silhouette else None
    best_dbi = min(valid_dbi, key=lambda r: r["dbi"]) if valid_dbi else None
    same_bw = best_silhouette and best_dbi and best_silhouette["bw"] == best_dbi["bw"]

    return {
        "cluster_result": cluster_result,
        "X_pca": X_pca,
        "X_scaled": X_scaled,
        "players": df.reset_index(drop=True),
        "features": df[features].reset_index(drop=True),
        "best_silhouette": best_silhouette,
        "best_dbi": best_dbi,
        "same_bw": same_bw,
    }

# MENJALANKAN MEAN SHIFT PER KATEGORI POSISI PEMAIN
def run_meanshift_by_position(season: str):
    players = get_player_features_df(season)
    if players.empty:
        return {"Forward": None, "Midfielder": None, "Defender": None}

    results = {}
    for group, positions in POSITION_GROUPS.items():
        player_by_position = players[players["position"].apply(
            lambda p: any(pos in str(p).upper().replace(" ", "").split(",") or
                          pos in str(p).upper().replace(" ", "") for pos in positions)
        )]
        features = FEATURES_BY_POSITION[group]
        if len(player_by_position) < 3:
            results[group] = None
            continue
        results[group] = run_meanshift(player_by_position, features)

    return results

# UNTUK MEMBUAT DAFTAR PEMAIN PER CLUSTER
def build_cluster_members_df(result: dict, best: dict):
    if not best or "labels" not in best:
        return None

    labels = np.asarray(best["labels"])
    players = result.get("players")
    if not isinstance(players, pd.DataFrame) or "player" not in players.columns:
        return None

    players = players.reset_index(drop=True)
    if len(players) != len(labels):
        return None

    player_name = players[["player"]].copy()
    try:
        player_name["ClusterId"] = labels.astype(int)
    except Exception:
        player_name["ClusterId"] = labels

    grouped = (
        player_name.groupby("ClusterId")["player"]
        .apply(list)
        .to_dict()
    )

    ordered_ids = sorted(grouped.keys())
    max_len = max(len(v) for v in grouped.values()) if grouped else 0

    data = {}
    for i in ordered_ids:
        col_name = f"Cluster {i}"
        vals = grouped[i]
        if len(vals) < max_len:
            vals = vals + [""] * (max_len - len(vals))
        data[col_name] = vals

    return pd.DataFrame(data)
