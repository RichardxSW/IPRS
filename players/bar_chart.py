import pandas as pd
from typing import Dict, List, Any

class BarDataMissing(Exception):
    pass

#MEMBACA FITUR PER POSISI YANG AKAN DIPAKAI
def get_features_for_group(group_name: str, features_by_position: Dict[str, List[str]]) -> List[str]:
    features = features_by_position.get(group_name, [])

    if not features:
        raise BarDataMissing(f"Tidak ada fitur untuk posisi '{group_name}'.")
    
    return features

# UNTUK BAR CHART FITUR PER CLUSTER
def get_cluster_feature_chart_data(results: Dict[str, Any], features: List[str]) -> pd.DataFrame:
    # VALIDASI JIKA SILHOUETTE TERBAIK TIDAK ADA
    if "best_silhouette" not in results or not results["best_silhouette"]:
        raise BarDataMissing("Silhouette terbaik tidak ditemukan.")
    
    labels = results["best_silhouette"].get("labels")
    all_features = results.get("features")
    if all_features is None:
        raise BarDataMissing("Fitur tidak ditemukan")

    feature = all_features[features].copy()
    feature["cluster"] = labels

    agg = (
        feature.groupby("cluster", as_index=False)[features]
        .mean(numeric_only=True)
        .sort_values("cluster")
    )

    chart_data = agg.melt(id_vars="cluster", var_name="Fitur", value_name="Mean")
    chart_data["Cluster"] = chart_data["cluster"].apply(lambda c: f"C{int(c)}")
    chart_data.drop(columns=["cluster"], inplace=True)
    return chart_data
