from __future__ import annotations
import re
from typing import List
import numpy as np
import pandas as pd
from django.db import transaction
from .models import League, Player
from django.db.models import Count
from django.core.exceptions import ValidationError
import io
from django.db.models import Case, When, Value, F, CharField
from django.db.models.functions import Concat

# CHOICES BUAT POSISI PEMAIN ACUAN
POSITION_CHOICES = [
    "Pilih posisi pemain acuan",
    "ST",
    "LW",
    "RW",
    "AM",
    "CM",
    "LM",
    "RM",
    "DM",
    "CB",
    "LB",
    "RB"
]

# KOLOM NUMERIK
NUM_COLS = [
    "Age","Appearance","Total Minute","Total Goal","Goal/game","Shot/game","SoT/game",
    "Assist","Assist/game","Success Dribble/game","Key Pass/game","Successful Pass/game",
    "Long Ball/game","Successful Crossing/game","Ball Recovered/game","Dribbled Past/game",
    "Clearance/game","Error leading to shot","Error leading to shot/game",
    "Total duel won/game","Aerial duel won/game"
]

# UNTUK TEMPLATE DATASET
TEMPLATE_DATA = {
    "Player": ["Marc Klok", "Beckham Putra Nugraha"],
    "Team": ["Persib Bandung", "Persib Bandung"],
    "Nationality": ["Indonesia","Indonesia"],
    "Naturalisasi": ["TRUE", "FALSE"],
    "Position": ["CM, DM", "RW"],
    "Age": [32, 24],
    "Appearance": [34, 29],
    "Total Minute": [3060, 1975],
    "Total Goal": [2, 6],
    "Goal/game": [1,2],
    "Shot/game": [1, 1.5],
    "SoT/game": [1, 1],
    "Assist": [3, 6],
    "Assist/game": [1, 1.5],
    "Success Dribble/game": [4, 8],
    "Key Pass/game": [1, 2],
    "Successful Pass/game": [20, 15],
    "Long Ball/game": [10, 2],
    "Successful Crossing/game": [2, 5],
    "Ball Recovered/game": [10, 5],
    "Dribbled Past/game": [5, 2],
    "Clearance/game": [5, 1],
    "Error leading to shot": [5, 2],
    "Error leading to shot/game": [5, 0.5],
    "Total duel won/game": [5, 5],
    "Aerial duel won/game": [5, 1],
}

# UNTUK RESET SESSION REKOMENDASI
def clear_recommend_state(st):
    st.session_state["recommend_state"] = None
    st.session_state["features"] = None
    st.session_state["compare_recommend"] = None
    return st

# UNTUK RESET SESSION CLUSTERING
def clear_cluster_state(st):
    st.session_state["cluster_result"] = None
    # st.session_state["selected_season"] = None
    return st

#VALIDASI FITUR PADA DATASET
def get_required(row, columns, key, string=False, boolean=False):
    if key.lower() not in columns:
        raise KeyError(f"Kolom {key} harus ada di dataset.")
                    
    val = row[columns[key.lower()]]

    if pd.isna(val):
        return None
    
    if not string and not boolean:
        try:
            val = float(val)
        except ValueError:
            raise ValidationError(f"Kolom {key} harus bernilai numerik. (contoh: 2 atau 2,5)")
        
    if boolean:        
        if str(val).lower() in ["true", "false"]:
            return str(val)
        else:
            raise ValidationError(f"Kolom {key} harus bernilai TRUE atau FALSE.")
    
    return str(val).strip() if string else val


# POST DATASET KE DATABASE
@transaction.atomic
def post_dataset(league_name: str, season: str, df: pd.DataFrame) -> int:
    column = {c.lower(): c for c in df.columns}

    # VALIDASI FORMAT MUSIM
    pattern = r"^\d{4}/\d{4}$"
    if not re.match(pattern, season):
        raise ValidationError("Format musim tidak valid. Gunakan format seperti 2024/2025.")

    # VALIDASI MUSIM
    start, end = map(int, season.split("/"))
    if end - start != 1:
        raise ValidationError("Tahun akhir harus satu tahun setelah tahun awal, misal 2024/2025.")

    # CEK APAKAH SUDAH ADA DATA DENGAN MUSIM TERSEBUT
    if League.objects.filter(league_name=league_name.strip(), season=season.strip()).exists():
        raise ValidationError(f"Data untuk {league_name} musim {season} sudah ada.")
    
    # CEK APAKAH ROW DATA SUDAH LENGKAP
    POSITION_REQUIRED = ["ST","LW","RW","CM","DM","CB","LB","RB"]
    MIN_PER_POSITION = 18
    position_counts = {p: 0 for p in POSITION_REQUIRED}
    for pos_str in df["Position"]:
        position_str = str(pos_str).upper().replace(" ", "")
        position = set(position_str.split(",")) if "," in position_str else {position_str}
        for p in POSITION_REQUIRED:
            if p in position:
                position_counts[p] += 1

    position_check = [p for p, count in position_counts.items() if count < MIN_PER_POSITION]
    if position_check:
        raise ValidationError(
            "Data belum lengkap."
        )

    # MENYIMPAN LIGA
    ds = League.objects.create(
        league_name=league_name.strip(),
        season=season.strip()
    )

    # VALIDASI VARIABEL HARUS ADA
    bulk = []
    for _, row in df.iterrows():
        player = get_required(row, column, "Player", string=True)
        team = get_required(row, column, "Team", string=True)
        nat = get_required(row, column, "Nationality", string=True)
        naturalisasi = get_required(row, column, "Naturalisasi", boolean=True)
        pos  = get_required(row, column, "Position", string=True)
        age  = get_required(row, column, "Age")
        app  = get_required(row, column, "Appearance")
        total_minute  = get_required(row, column, "Total Minute")
        total_goal  = get_required(row, column, "Total Goal")
        goal_pg = get_required(row, column, "Goal/Game")
        shot_pg = get_required(row, column, "Shot/Game")
        sot_pg = get_required(row, column, "Sot/Game")
        assist = get_required(row, column, "Assist")
        assist_pg = get_required(row, column, "Assist/game")
        dribble_pg = get_required(row, column, "Successful Dribble/Game")
        keypass_pg = get_required(row, column, "Key Pass/Game")
        pass_pg = get_required(row, column, "Successful Pass/Game")
        longball_pg = get_required(row, column, "Long Ball/Game")
        crossing_pg = get_required(row, column, "Successful Crossing/Game")
        ballrecovered_pg = get_required(row, column, "Ball Recovered/Game")
        dribbledpast_pg = get_required(row, column, "Dribbled Past/Game")
        clearance_pg = get_required(row, column, "Clearance/Game")
        error = get_required(row, column, "Error Leading to Shot")
        error_pg = get_required(row, column, "Error Leading to Shot/Game")
        totalduel_pg = get_required(row, column, "Total Duel Won/Game")
        aerialduel_pg = get_required(row, column, "Aerial Duel Won/Game")

        bulk.append(
            Player(
                league=ds,
                player=player,
                team=team, 
                nationality=nat,
                naturalisasi=True if naturalisasi in ["True", "TRUE", "true", True] else False,
                position=pos,
                age=age,
                appearance=app,
                total_minute=total_minute,
                total_goal=total_goal,
                goal_per_game=goal_pg,
                shot_per_game=shot_pg,
                sot_per_game=sot_pg,
                assist=assist,
                assist_per_game=assist_pg,
                successful_dribble_per_game=dribble_pg,
                key_pass_per_game=keypass_pg,
                successful_pass_per_game=pass_pg,
                long_ball_per_game=longball_pg,
                successful_crossing_per_game=crossing_pg,
                ball_recovered_per_game=ballrecovered_pg,
                dribbled_past_per_game=dribbledpast_pg,
                clearance_per_game=clearance_pg,
                error=error,
                error_per_game=error_pg,
                total_duel_per_game=totalduel_pg,
                aerial_duel_per_game=aerialduel_pg
            )
        )

    if bulk:
        Player.objects.bulk_create(bulk, batch_size=1000)

    return ds.id

# AMBIL DATA LIGA
def get_leagues() -> List[str]:
    return list(
        League.objects.values_list("league_name", flat=True).distinct().order_by("league_name")
    )

# AMBIL DAFTAR MUSIM
def get_seasons(selected_league) -> List[str]:
    return list(
        League.objects.filter(league_name=selected_league.strip()).values_list("season", flat=True).order_by("season")
    )

# AMBIL DATA KLUB
def get_clubs_by_season(selected_league, season: str) -> List[str]:
    club = list(
        Player.objects.filter(
            league__season=season,
            league__league_name=selected_league,
        )
        .order_by("team")
        .values_list("team", flat=True)
        .distinct()
    )
    return club

# AMBIL DATA PEMAIN
def get_players_by_season(selected_league, season: str, position: str) -> List[str]:
    players = list(
        Player.objects.filter(
            league__season=season,
            league__league_name=selected_league,
            position__icontains=position,
        ).annotate(
            player_name=Case(
                When(
                    naturalisasi=True,
                    then=Concat(F('player'), Value(' ( Naturalisasi )'))
                ),\
                default=F('player'),
                output_field=CharField()
            )
        ).order_by("player_name").values_list("player_name", flat=True)
    )
    return players

# AMBIL DATA PEMAIN DENGAN FILTER KLUB
def get_players_by_season_and_club(selected_league, season: str, position: str, club: str) -> List[str]:    
    qs = Player.objects.filter(
        league__league_name=selected_league,
        league__season=season,
        position__icontains=position,
    )

    if club and club.lower() != "semua":
        qs = qs.filter(team__iexact=club)

    players = list(qs.annotate(
        player_name=Case(
            When(
                naturalisasi=True,
                then=Concat(F('player'), Value(' ( Naturalisasi )'))
            ),\
            default=F('player'),
            output_field=CharField()
        )
    ).order_by("player_name").values_list("player_name", flat=True))
    return players

# DOWNLOAD TEMPLATE DATASET
def build_template_file(df) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Template Dataset")

    buf.seek(0)
    return buf.getvalue()

#BACA DATA PEMAIN ACUAN 
def get_player_detail(season: str, player_name: str) -> dict | None:
    fields = [
        "player_name", "team", "nationality", "position", "age",
        "appearance", "total_minute", "total_goal","assist",
        "goal_per_game", "shot_per_game", "sot_per_game",
        "assist_per_game", "successful_dribble_per_game", "key_pass_per_game",
        "successful_pass_per_game", "long_ball_per_game", "successful_crossing_per_game",
        "ball_recovered_per_game", "dribbled_past_per_game", "clearance_per_game",
        "error", "error_per_game", "total_duel_per_game", "aerial_duel_per_game",
    ]

    player_name = player_name.replace(" ( Naturalisasi )", "").strip()
    
    return (
        Player.objects
        .filter(league__season=season, player=player_name)
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
        .values(*fields)
        .first()
    )

#AMBIL DETAIL MUSIM YANG TERSIMPAN
def get_list_of_season():
    return list(
        League.objects
        .annotate(player_count=Count('players'))
        .values('id', 'league_name', 'season', 'player_count', 'uploaded_at')
        .order_by('-uploaded_at')
    )

#HAPUS MUSIM
def delete_dataset(dataset_id: int) -> bool:
    deleted, _ = League.objects.filter(id=dataset_id).delete()
    return deleted > 0


