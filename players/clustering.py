import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .models import Player, League
from django.db.models import Case, When, Value, F, CharField
from django.db.models.functions import Concat

# KATEGORI POSISI PEMAIN
POSITION_GROUPS = {
    "Pemain Penyerang": ["ST", "LW", "RW"],
    "Pemain Gelandang": ["AM", "CM", "DM", "LM", "RM"],
    "Pemain Bertahan": ["CB", "LB", "RB"],
}

# STATISTIK YANG DIGUNAKAN UNTUK CLUSTERING PER KATEGORI POSISI PEMAIN
FEATURES_BY_POSITION = {
    "Pemain Penyerang": [
        "goal_per_game", "shot_per_game", "sot_per_game", "assist_per_game", "successful_dribble_per_game", 
        "successful_crossing_per_game", "key_pass_per_game", 
        "aerial_duel_per_game", "total_duel_per_game",
    ],
    "Pemain Gelandang": [
        "goal_per_game", "shot_per_game", "sot_per_game", "assist_per_game", "successful_dribble_per_game", 
        "successful_crossing_per_game", "key_pass_per_game", "successful_pass_per_game", "long_ball_per_game", 
        "aerial_duel_per_game",  "total_duel_per_game", "ball_recovered_per_game", "dribbled_past_per_game", "clearance_per_game"
    ],
    "Pemain Bertahan": [
        "successful_dribble_per_game", "successful_crossing_per_game", "key_pass_per_game",
        "aerial_duel_per_game", "total_duel_per_game", "ball_recovered_per_game", "dribbled_past_per_game", "clearance_per_game",         
    ],
}

# VARIABEL YANG TIDAK TERPAKAI UNTUK CLUSTERING
META_COLUMNS = [
    "id", "player_name", "team", "position", "nationality",
    "age", "appearance", "total_minute",
    "total_goal", "assist", "error"
]

# LABEL VARIABEL UNTUK DITAMPILKAN 
FEATURE_LABELS = {
    "age": "Age",
    "appearance": "Appearances",
    "total_minute": "Total Minutes Played",
    "clearance_per_game": "Clearance / Game",
    "ball_recovered_per_game": "Ball Recovery / Game",    
    "aerial_duel_per_game": "Aerial Duel Won / Game",
    "total_duel_per_game": "Total Duel Won / Game",
    "successful_pass_per_game": "Successful Pass / Game",
    "long_ball_per_game": "Long Pass / Game",
    "dribbled_past_per_game": "Dribbled Past / Game",    
    "successful_dribble_per_game": "Successful Dribble / Game",
    "total_goal": "Total Goals",
    "goal_per_game": "Goal / Game",
    "assist_per_game": "Assist / Game",
    "assist": "Total Assists",
    "shot_per_game": "Shot / Game",
    "sot_per_game": "Shot on Target / Game",
    "key_pass_per_game": "Key Pass / Game",
    "successful_crossing_per_game": "Successful Crossing / Game",
    "error": "Errors Leading to Shot"
}


# MENGAMBIL DATA FITUR FITUR PEMAIN
def get_player_features_data(selected_league, season: str) -> pd.DataFrame:
    all_feature = sorted({f for feats in FEATURES_BY_POSITION.values() for f in feats})
    qs = (
        Player.objects
        .filter(league__league_name=selected_league.strip(), league__season=season.strip())
        .annotate(
            player_name=Case(
                When(
                    naturalisasi=True,
                    then=Concat(F('player'), Value(' ( Naturalisasi )'))
                ),\
                default=F('player'),
                output_field=CharField()
            )
        )
        .values(*META_COLUMNS, *all_feature)
        .order_by("player_name")
    )
    return pd.DataFrame(list(qs))

# PREPROCESSING DATA
def preprocessing_data(df: pd.DataFrame, feat_cols):
    # VALIDASI DATA 
    X = (
        df[feat_cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )

    # ZSCORE
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    return X_scaled, X_pca

# METODE MEAN SHIFT
def run_meanshift(df: pd.DataFrame, features):
    #PREPROSES
    X_scaled, X_pca = preprocessing_data(df, features)

    # BANDWIDTH
    bandwidths = np.arange(0.5, 5.5, 0.5)

    results = []

    # MENGUJI CLUSTERING DENGAN BANDWIDTH 0,5 - 5
    for bw in bandwidths:
        labels, n_clusters, = None, 0        
        for bin_seed in [True]:
            try:
                ms = MeanShift(bandwidth=float(bw), bin_seeding=bin_seed, cluster_all=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    labels = ms.fit_predict(X_scaled)
                n_clusters = len(np.unique(labels))
                break
            except ValueError:
                raise Exception("Clustering gagal")

        # EVALUASI
        silhouette, dbi = None, None
        if labels is not None and n_clusters >= 2:
            try:
                # NILAI SILHOUTTE
                silhouette = float(silhouette_score(X_scaled, labels))
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
        "Cluster": r["n_clusters"],
        "Silhouette": f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "-",
        "DBI": f"{r['dbi']:.4f}" if r["dbi"] is not None else "-",
    } for r in results])

    valid_silhouette = [r for r in results if r["silhouette"] is not None and r['silhouette'] != '-']
    
    best_silhouette = max(valid_silhouette, key=lambda r: r["silhouette"]) if valid_silhouette else None

    return {
        "cluster_result": cluster_result,
        "X_pca": X_pca,
        "X_scaled": X_scaled,
        "players": df.reset_index(drop=True),
        "features": df[features].reset_index(drop=True),
        "best_silhouette": best_silhouette,
    }

# MENJALANKAN MEAN SHIFT PER KATEGORI POSISI PEMAIN
def run_meanshift_by_position(selected_league, season: str):
    players = get_player_features_data(selected_league, season)
    if players.empty:
        return {"Pemain Penyerang": None, "Pemain Gelandang": None, "Pemain Bertahan": None}

    results = {}
    for group, positions in POSITION_GROUPS.items():
        player_by_position = players[players["position"].apply(
            lambda p: any(pos in str(p).upper().replace(" ", "").split(",") or
                          pos in str(p).upper().replace(" ", "") for pos in positions)
        )]
        features = FEATURES_BY_POSITION[group]  

        results[group] = run_meanshift(player_by_position, features)

    return results

# UNTUK MEMBUAT DAFTAR PEMAIN PER CLUSTER
def get_cluster_members_data(result: dict, best: dict):
    if not best or "labels" not in best:
        return None

    labels = np.asarray(best["labels"])
    players = result.get("players")
    if not isinstance(players, pd.DataFrame) or "player_name" not in players.columns:
        return None

    players = players.reset_index(drop=True)
    if len(players) != len(labels):
        return None

    player_name = players[["player_name"]].copy()
    try:
        player_name["ClusterId"] = labels.astype(int)
    except Exception:
        player_name["ClusterId"] = labels

    grouped = (
        player_name.groupby("ClusterId")["player_name"]
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
