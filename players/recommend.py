import math
import re
import altair as alt
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

def _norm_scale_per_feature(df_raw: pd.DataFrame, features: list[str], q=0.98):
    """
    Skala per fitur supaya bar kiri/kanan fair:
    - Pakai quantile tinggi (default 98%) biar outlier gak ‘makan’ sumbu.
    - Fallback ke max kalau quantile = 0.
    """
    scale = {}
    for f in features:
        s = pd.to_numeric(df_raw[f], errors="coerce")
        vmax = np.nanquantile(s, q)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = np.nanmax(s)
        scale[f] = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    return scale

def fm_style_compare_chart(
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
    A = A.iloc[0]; B = B.iloc[0]

    scale = _norm_scale_per_feature(features_df, features_to_compare, q=0.98)

    rows = []
    for f in features_to_compare:
        va = float(A[f]); vb = float(B[f])
        s  = scale[f]
        # normalisasi ke [0..1] lalu ke domain [-1..1] dgn titik tengah 0
        na = min(max(va / s, 0.0), 1.0)
        nb = min(max(vb / s, 0.0), 1.0)
        d  = nb - na                    # Δ (kanan - kiri), dipakai utk arah bar warna
        left_fill  = -abs(d) if d < 0 else 0.0   # isi warna ke kiri (negatif)
        right_fill =  abs(d) if d > 0 else 0.0   # isi warna ke kanan (positif)

        rows.append({
            "Fitur": f,
            "val_left": va,
            "val_right": vb,
            "diff": d,
            "left_fill": left_fill,
            "right_fill": right_fill
        })

    base = pd.DataFrame(rows)
    dom = [-1, 1]

    # Latar putih penuh
    bg = alt.Chart(base).mark_bar(color="#eeeeee").encode(
        y=alt.Y("Fitur:N", sort=features_to_compare, title=None),
        x=alt.X("bg_min:Q", title=None, scale=alt.Scale(domain=dom)),
        x2="bg_max:Q",
    ).transform_calculate(bg_min="-1", bg_max="1")

    # Garis tengah
    mid = alt.Chart(base).mark_rule(color="#222", strokeWidth=1).encode(
        x=alt.X("zero:Q", scale=alt.Scale(domain=dom), title=None)
    ).transform_calculate(zero="0")

    # Isi warna: kiri (anchor menang) dan kanan (target menang)
    left_bar = alt.Chart(base).mark_bar().encode(
        y="Fitur:N",
        x=alt.X("zero:Q", scale=alt.Scale(domain=dom), title=None),
        x2="left_fill:Q",
        color=alt.value("#f59e0b")   # oranye utk condong ke kiri
    ).transform_calculate(zero="0")

    right_bar = alt.Chart(base).mark_bar().encode(
        y="Fitur:N",
        x=alt.X("zero:Q", scale=alt.Scale(domain=dom), title=None),
        x2="right_fill:Q",
        color=alt.value("#3b82f6")   # biru utk condong ke kanan
    ).transform_calculate(zero="0")

    # Angka asli di LUAR bar
    left_text = alt.Chart(base).mark_text(align="right", dx=-8).encode(
        y="Fitur:N",
        x=alt.value(0),  # dummy, kita set via transform_calculate
        text=alt.Text("val_left:Q", format=".2f"),
        color=alt.value("#1f2937"),
    ).transform_calculate(xpos="-1.05")

    right_text = alt.Chart(base).mark_text(align="left", dx=8).encode(
        y="Fitur:N",
        x=alt.value(0),
        text=alt.Text("val_right:Q", format=".2f"),
        color=alt.value("#1f2937"),
    ).transform_calculate(xpos="1.05")

    # Posisikan teks pakai x='xpos'
    left_text = left_text.encode(x=alt.X("xpos:Q", scale=alt.Scale(domain=dom)))
    right_text = right_text.encode(x=alt.X("xpos:Q", scale=alt.Scale(domain=dom)))

    # Tooltip opsional
    tt = alt.Chart(base).mark_bar(opacity=0).encode(
        y="Fitur:N",
        x=alt.X("bg_min:Q", scale=alt.Scale(domain=dom), title=None),
        x2="bg_max:Q",
        tooltip=[
            alt.Tooltip("Fitur:N"),
            alt.Tooltip("val_left:Q",  title=anchor_player, format=".3f"),
            alt.Tooltip("val_right:Q", title=target_player, format=".3f"),
            alt.Tooltip("diff:Q",      title="Δ (kanan−kiri)", format="+.3f"),
        ],
    ).transform_calculate(bg_min="-1", bg_max="1")

    chart = (bg + left_bar + right_bar + mid + left_text + right_text + tt).properties(
        height=28 * len(features_to_compare),
        width=720
    )
    return chart

def _feature_scale(df: pd.DataFrame, features: list[str], q=0.98):
    out = {}
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        vmax = np.nanquantile(s, q)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = np.nanmax(s)
        out[f] = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    return out

def _feature_scale_per_feature(df: pd.DataFrame, features: list[str], q=0.98):
    scale = {}
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        vmin, vmax = np.nanmin(s), np.nanquantile(s, q)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = np.nanmax(s)
        # rentang per fitur
        scale[f] = max(vmax - vmin, 1e-6)
    return scale

def fm_compare_hconcat(
    features_df: pd.DataFrame,   # dataframe fitur (punya kolom 'player_name' + fitur)
    anchor_player: str,
    target_player: str,
    features_to_compare: list[str],
):
    cols = ["player_name", *features_to_compare]
    A = features_df.loc[features_df["player_name"] == anchor_player, cols]
    B = features_df.loc[features_df["player_name"] == target_player, cols]
    if A.empty or B.empty:
        raise ValueError("Data pemain tidak ditemukan.")
    A = A.iloc[0]; B = B.iloc[0]

    scale = _feature_scale_per_feature(features_df, features_to_compare, q=0.98)

    # mins = features_df[features_to_compare].min()
    # maxs = features_df[features_to_compare].max()
    # rng = (maxs - mins).replace(0, np.nan).fillna(1.0)

    # q05 = features_df[features_to_compare].quantile(0.05)
    # q95 = features_df[features_to_compare].quantile(0.95)
    # rng = (q95 - q05).replace(0, np.nan).fillna(1.0)

    rows = []
    for f in features_to_compare:
        va = float(A[f])
        vb = float(B[f])
        d_raw  = vb - va                 # Δ RAW untuk ditampilkan di kolom 3

        s  = scale[f]
        # na = min(max(va / s, 0.0), 1.0)
        # nb = min(max(vb / s, 0.0), 1.0)
        vmax = max(abs(va), abs(vb), 1e-6)  # biar ga nol
        d_norm = np.clip(d_raw / vmax, -1.0, 1.0)      # untuk arah bar ([-1..1])

        rows.append({
            "Fitur": f,
            "val_left": va,
            "val_right": vb,
            "diff_raw": d_raw,
            "left_fill": (-abs(d_norm)) if d_norm < 0 else 0.0,
            "right_fill": (abs(d_norm)) if d_norm > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    dom = [-1, 1]
    bar_h = 28 * len(features_to_compare)


    # Axis helper: semua axis disembunyikan untuk kolom 2 & 3
    ax_none_x = alt.Axis(title=None, labels=False, ticks=False, domain=False)
    ax_none_y = alt.Axis(title=None, labels=False, ticks=False, domain=False)

    # Kolom 1: nama statistik
    names = (
        alt.Chart(df)
        .mark_text(align="right", dx=-6, fontWeight="bold")
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            text="Fitur:N",
            color=alt.value("#9ca3af"),
        )
        .properties(width=100, height=bar_h)
    )

    # Kolom 2: bar selisih + angka nilai RAW di luar bar
    bg = (
        alt.Chart(df)
        .mark_bar(color="#eeeeee", cornerRadius=6)
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
        .mark_bar(color="#f59e0b", cornerRadius=6)
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("zero:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            x2="left_fill:Q",
        )
        .transform_calculate(zero="0")
    )
    right_bar = (
        alt.Chart(df)
        .mark_bar(color="#3b82f6", cornerRadius=6)
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("zero:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            x2="right_fill:Q",
        )
        .transform_calculate(zero="0")
    )
    # angka nilai RAW di LUAR bar
    left_txt = (
        alt.Chart(df)
        .mark_text(align="right", dx=-8, fontWeight="bold")
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
        .mark_text(align="left", dx=8, fontWeight="bold")
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            x=alt.X("xpos:Q", scale=alt.Scale(domain=dom), axis=ax_none_x),
            text=alt.Text("val_right:Q", format=".1f"),
            color=alt.value("#f9fafb"),
        )
        .transform_calculate(xpos="1.05")
    )

    bars = (bg + left_bar + right_bar + mid + left_txt + right_txt).properties(
        width=420, height=bar_h
    )

    # Kolom 3: Δ RAW (teks berwarna sesuai tanda)
    delta_color = alt.condition("datum.diff_raw > 0", alt.value("#3b82f6"), alt.value("#f59e0b"))
    delta = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontWeight="bold")
        .encode(
            y=alt.Y("Fitur:N", sort=features_to_compare, axis=ax_none_y),
            text=alt.Text("diff_raw:Q", format="+.1f"),
            color=delta_color,
        )
        .properties(width=80, height=bar_h)
    )

    chart = alt.hconcat(names, bars, delta).resolve_scale(y="shared")
    return chart