from __future__ import annotations
import re
from typing import List
import numpy as np
import pandas as pd
from django.db import transaction
from .models import Player, Season
from django.db.models import Count
from django.core.exceptions import ValidationError
import io

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

TEMPLATE_DATA = {
        "Player": ["Marc Klok", np.nan],
        "Team": ["Persib Bandung", np.nan],
        "Nationality": ["Indonesia",np.nan],
        "Position": ["DM", np.nan],
        "Age": [25, np.nan],
        "Appearance": [34, np.nan],
        "Total Minute": [3060, np.nan],
        "Total Goal": [10, np.nan],
        "Goal/game": [1,np.nan],
        "Shot/game": [1, np.nan],
        "SoT/game": [1, np.nan],
        "Assist": [5, np.nan],
        "Assist/game": [1, np.nan],
        "Success Dribble/game": [8, np.nan],
        "Key Pass/game": [5, np.nan],
        "Successful Pass/game": [20, np.nan],
        "Long Ball/game": [10, np.nan],
        "Successful Crossing/game": [10, np.nan],
        "Ball Recovered/game": [10, np.nan],
        "Dribbled Past/game": [5, np.nan],
        "Clearance/game": [5,np.nan],
        "Error leading to shot": [5, np.nan],
        "Error leading to shot/game": [5, np.nan],
        "Total duel won/game": [5, np.nan],
        "Aerial duel won/game": [5, np.nan],
    }

def clear_recommend_state(st):
    st.session_state["recommend_state"] = None
    st.session_state["features"] = None
    st.session_state["compare_recommend"] = None
    return st

def clear_cluster_state(st):
    st.session_state["cluster_result"] = None
    st.session_state["selected_season"] = None
    return st

#VALIDASI FITUR PADA DATASET
def get_required(row, columns, key, string=False):
    if key not in columns:
        raise KeyError(f"Kolom {key} harus ada di dataset.")
                    
    val = row[columns[key]]

    if pd.isna(val):
        return None
    
    return str(val).strip() if string else val


# POST DATASET KE DATABASE
@transaction.atomic
def post_dataset(league_name: str, season: str, df: pd.DataFrame) -> int:
    column = {c.lower(): c for c in df.columns}

    # validasi format musim
    pattern = r"^\d{4}/\d{4}$"
    if not re.match(pattern, season):
        raise ValidationError("Format musim tidak valid. Gunakan format seperti 2024/2025.")

    start, end = map(int, season.split("/"))
    if end - start != 1:
        raise ValidationError("Tahun akhir harus satu tahun setelah tahun awal, misal 2024/2025.")

    # --- CEK APAKAH SUDAH ADA DATASET DENGAN MUSIM TERSEBUT ---
    if Season.objects.filter(season=season.strip()).exists():
        raise ValidationError(f"Data untuk {league_name} musim {season} sudah ada.")

    ds, created = Season.objects.get_or_create(
        league_name=league_name.strip(),
        season=season.strip()
    )
    if not created:
        ds.players.all().delete()

    bulk = []
    for _, row in df.iterrows():
        player = get_required(row, column, "player", string=True)
        team = get_required(row, column, "team", string=True)
        nat = get_required(row, column, "nationality", string=True)
        pos  = get_required(row, column, "position", string=True)
        age  = get_required(row, column, "age")
        app  = get_required(row, column, "appearance")
        total_minute  = get_required(row, column, "total minute")
        total_goal  = get_required(row, column, "total goal")
        goal_pg = get_required(row, column, "goal/game")
        shot_pg = get_required(row, column, "shot/game")
        sot_pg = get_required(row, column, "sot/game")
        assist = get_required(row, column, "assist")
        assist_pg = get_required(row, column, "assist/game")
        dribble_pg = get_required(row, column, "successful dribble/game")
        keypass_pg = get_required(row, column, "key pass/game")
        pass_pg = get_required(row, column, "successful pass/game")
        longball_pg = get_required(row, column, "long ball/game")
        crossing_pg = get_required(row, column, "successful crossing/game")
        ballrecovered_pg = get_required(row, column, "ball recovered/game")
        dribbledpast_pg = get_required(row, column, "dribbled past/game")
        clearance_pg = get_required(row, column, "clearance/game")
        error = get_required(row, column, "error leading to shot")
        error_pg = get_required(row, column, "error leading to shot/game")
        totalduel_pg = get_required(row, column, "total duel won/game")
        aerialduel_pg = get_required(row, column, "aerial duel won/game")

        bulk.append(
            Player(
                season=ds,
                player=player,
                team=team, 
                nationality=nat, 
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

# BACA DAFTAR MUSIM
def get_seasons() -> List[str]:
    return list(
        Season.objects.values_list("season", flat=True).distinct().order_by("season")
    )

# AMBIL DATA KLUB
def get_clubs_by_season(season: str) -> List[str]:
    club = list(
        Player.objects.filter(
            season__season=season
        )
        .order_by("team")
        .values_list("team", flat=True)
        .distinct()
    )
    print(len(club))
    return club

# AMBIL DATA PEMAIN
def get_players_by_season(season: str, position: str) -> List[str]:
    players = list(
        Player.objects.filter(
            season__season=season, 
            position__icontains=position,
        ).order_by("player").values_list("player", flat=True)
    )
    print(len(players))
    return players

# AMBIL DATA PEMAIN DENGAN FILTER KLUB
def get_players_by_season_and_club(season: str, position: str, club: str) -> List[str]:    
    qs = Player.objects.filter(
        season__season=season,
        position__icontains=position,
    )
    if club and club.lower() != "semua":
        qs = qs.filter(team__iexact=club)

    players = list(qs.order_by("player").values_list("player", flat=True))
    print(len(players))
    return players

# DOWNLOAD TEMPLATE DATASET
def build_template_file() -> bytes:
    template = pd.DataFrame(TEMPLATE_DATA)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        template.to_excel(writer, index=False, sheet_name="Template Dataset")

    buf.seek(0)
    return buf.getvalue()

#BACA DATA PEMAIN ACUAN 
def get_player_detail(season: str, player_name: str) -> dict | None:
    fields = [
        "player", "team", "nationality", "position", "age",
        "appearance", "total_minute", "total_goal","assist",
        "goal_per_game", "shot_per_game", "sot_per_game",
        "assist_per_game", "successful_dribble_per_game", "key_pass_per_game",
        "successful_pass_per_game", "long_ball_per_game", "successful_crossing_per_game",
        "ball_recovered_per_game", "dribbled_past_per_game", "clearance_per_game",
        "error", "error_per_game", "total_duel_per_game", "aerial_duel_per_game",
    ]
    return (
        Player.objects
        .filter(season__season=season, player=player_name)
        .values(*fields)
        .first()
    )

#BACA DETAIL MUSIM YANG TERSIMPAN
def get_list_of_season():
    return list(
        Season.objects
        .annotate(player_count=Count('players'))
        .values('id', 'league_name', 'season', 'player_count', 'uploaded_at')
        .order_by('-uploaded_at')
    )

#HAPUS MUSIM
def delete_dataset(dataset_id: int) -> bool:
    deleted, _ = Season.objects.filter(id=dataset_id).delete()
    return deleted > 0


