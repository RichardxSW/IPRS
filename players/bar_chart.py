import pandas as pd
from typing import Dict, List, Any

class BarDataMissing(Exception):
    pass

# --- HELPER DI LUAR ---
def format_cluster_label(c):
    """Ubah label cluster: 0 → C0, '1' → C1, 'C2' tetap C2."""
    s = str(c)
    if s.startswith("C"):
        return s
    try:
        return f"C{int(s)}"
    except ValueError:
        return s

#MEMBACA FITUR PER POSISI YANG AKAN DIPAKAI
def get_features_for_group(group_name: str, features_by_position: Dict[str, List[str]]) -> List[str]:
    features = features_by_position.get(group_name, [])

    if not features:
        raise BarDataMissing(f"Tidak ada fitur untuk posisi '{group_name}'.")
    
    return features

# UNTUK BAR CHART FITUR PER CLUSTER
# def get_cluster_feature_chart_data(results: Dict[str, Any], features: List[str]) -> pd.DataFrame:
#     # VALIDASI JIKA SILHOUETTE TERBAIK TIDAK ADA
#     if "best_silhouette" not in results or not results["best_silhouette"]:
#         raise BarDataMissing("Silhouette terbaik tidak ditemukan.")
    
#     labels = results["best_silhouette"].get("labels")
#     all_features = results.get("features")

#     if all_features is None:
#         raise BarDataMissing("Fitur tidak ditemukan")

#     feature = all_features[features].copy()
#     feature["cluster"] = labels

#     aggregate = (
#         feature.groupby("cluster", as_index=False)[features]
#         .mean(numeric_only=True)
#         .sort_values("cluster")
#     )

#     chart_data = aggregate.melt(id_vars="cluster", var_name="Fitur", value_name="Mean")
#     chart_data["Cluster"] = chart_data["cluster"].apply(lambda c: f"C{int(c)}")
#     chart_data.drop(columns=["cluster"], inplace=True)
#     return chart_data

def get_cluster_feature_chart_data(results: Dict[str, Any], features: List[str]) -> pd.DataFrame:
    best_sil = results.get("best_silhouette")
    if not best_sil:
        raise BarDataMissing("Silhouette terbaik tidak ditemukan.")

    labels = best_sil.get("labels")
    all_features = results.get("features")

    if all_features is None:
        raise BarDataMissing("Fitur tidak ditemukan.")

    if labels is None:
        raise BarDataMissing("Label cluster tidak ditemukan.")

    if len(labels) != len(all_features):
        raise BarDataMissing(
            f"Panjang labels ({len(labels)}) dan data fitur ({len(all_features)}) tidak sama."
        )

    # Ambil kolom fitur mentah per pemain
    try:
        feature_df = all_features[features].copy()
    except KeyError as e:
        raise BarDataMissing(f"Beberapa fitur tidak ditemukan di dataframe fitur: {e}")

    # Tambahkan label cluster
    feature_df["cluster"] = labels

    # Ubah ke format long (Value per pemain)
    long_df = feature_df.melt(
        id_vars="cluster",
        value_vars=features,
        var_name="Fitur",
        value_name="Value",
    )

    # Gunakan helper yg sudah dipisah
    long_df["Cluster"] = long_df["cluster"].apply(format_cluster_label)
    long_df.drop(columns=["cluster"], inplace=True)

    long_df = long_df.dropna(subset=["Value"])

    if long_df.empty:
        raise BarDataMissing("Semua nilai fitur kosong / NaN setelah diproses.")

    return long_df
