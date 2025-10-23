import pandas as pd
from typing import Dict, List, Any


class BarDataMissing(Exception):
    pass

#MEMBACA FITUR PER POSISI YANG AKAN DIPAKAI
def get_features_for_group(group_name: str, features_by_pos: Dict[str, List[str]]) -> List[str]:
    feats = features_by_pos.get(group_name, [])
    if not feats:
        raise BarDataMissing(f"Tidak ada fitur untuk posisi '{group_name}'.")
    return feats

# BAR CHART UNTUK HASIL CLUSTERING
def build_cluster_feature_bar_df(res: Dict[str, Any], feature_cols: List[str]) -> pd.DataFrame:
    """
    Menghasilkan dataframe berisi rata-rata fitur per cluster.
    Format: Cluster | Fitur | Mean
    """
    if "best_sil" not in res or not res["best_sil"]:
        raise BarDataMissing("best_sil tidak ditemukan.")
    labels = res["best_sil"].get("labels")
    feat_df = res.get("feature_df")
    if feat_df is None:
        raise BarDataMissing("feature_df tidak tersedia.")

    tmp = feat_df[feature_cols].copy()
    tmp["cluster"] = labels

    agg = (
        tmp.groupby("cluster", as_index=False)[feature_cols]
        .mean(numeric_only=True)
        .sort_values("cluster")
    )

    long_df = agg.melt(id_vars="cluster", var_name="Fitur", value_name="Mean")
    long_df["Cluster"] = long_df["cluster"].apply(lambda c: f"C{int(c)}")
    long_df.drop(columns=["cluster"], inplace=True)
    return long_df
