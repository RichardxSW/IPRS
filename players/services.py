from __future__ import annotations
from typing import List
import pandas as pd
from django.db import transaction
from .models import Dataset, Player
from django.db.models import Count
from django.core.exceptions import ValidationError
import io


#VALIDASI DATASET
def get_required(row, cols, key, as_str=False):
    """Ambil nilai dari kolom. Raise Exception jika kolom tidak ditemukan."""
    if key not in cols:
        raise KeyError(f"Kolom {key} harus ada di dataset.")
                    #    f"Kolom yang tersedia: {list(cols.keys())}")
    val = row[cols[key]]
    if pd.isna(val):
        return None
    return str(val).strip() if as_str else val


# POST DATASET KE DATABASE
@transaction.atomic
def insert_dataset_and_players(league_name: str, season: str, df: pd.DataFrame) -> int:
    """
    Simpan/replace dataset + pemain. Return dataset_id.
    """
    # Dataset
    cols = {c.lower(): c for c in df.columns}

    # --- CEK APAKAH SUDAH ADA DATASET DENGAN LIGA & MUSIM TERSEBUT ---
    if Dataset.objects.filter(league_name=league_name.strip(), season=season.strip()).exists():
        raise ValidationError(f"Data untuk {league_name} musim {season} sudah ada.")

    ds, created = Dataset.objects.get_or_create(
        league_name=league_name.strip(),
        season=season.strip()
    )
    if not created:
        ds.players.all().delete()

    # Player
    bulk = []
    for _, row in df.iterrows():
        player = get_required(row, cols, "player", as_str=True)
        team = get_required(row, cols, "team", as_str=True)
        nat = get_required(row, cols, "nationality", as_str=True)
        pos  = get_required(row, cols, "position", as_str=True)
        age  = get_required(row, cols, "age")
        app  = get_required(row, cols, "appearance")
        total_minute  = get_required(row, cols, "total minute")
        total_goal  = get_required(row, cols, "total goal")
        goal_pg = get_required(row, cols, "goal/game")
        shot_pg = get_required(row, cols, "shot/game")
        sot_pg = get_required(row, cols, "sot/game")
        assist = get_required(row, cols, "assist")
        assist_pg = get_required(row, cols, "assist/game")
        dribble_pg = get_required(row, cols, "successful dribble/game")
        keypass_pg = get_required(row, cols, "key pass/game")
        pass_pg = get_required(row, cols, "successful pass/game")
        longball_pg = get_required(row, cols, "long ball/game")
        crossing_pg = get_required(row, cols, "successful crossing/game")
        ballrecovered_pg = get_required(row, cols, "ball recovered/game")
        dribbledpast_pg = get_required(row, cols, "dribbled past/game")
        clearance_pg = get_required(row, cols, "clearance/game")
        error = get_required(row, cols, "error leading to shot")
        error_pg = get_required(row, cols, "error leading to shot/game")
        totalduel_pg = get_required(row, cols, "total duel won/game")
        aerialduel_pg = get_required(row, cols, "aerial duel won/game")


        bulk.append(
            Player(
                dataset=ds,
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

# BACA DATA MUSIM
def get_seasons() -> List[str]:
    return list(
        Dataset.objects.values_list("season", flat=True).distinct().order_by("season")
    )

# BACA DATA PEMAIN
def get_players_by_season(season: str, position: str) -> List[str]:
    players = list(
        Player.objects.filter(
            dataset__season=season, 
            position__icontains=position,
        ).order_by("player").values_list("player", flat=True)
    )
    print(len(players))
    return players

# DOWNLOAD TEMPLATE DATASET
def make_template_excel_bytes() -> bytes:
    template = pd.DataFrame({
        "Player": ["Marc Klok", ""],
        "Team": ["Persib Bandung", ""],
        "Nationality": ["Indonesia", ""],
        "Position": ["DM", ""],
        "Age": [25, ""],
        "Appearance": [34, ""],
        "Total Minute": [3060, ""],
        "Total Goal": [10, ""],
        "Goal/game": [1, ""],
        "Shot/game": [1, ""],
        "SoT/game": [1, ""],
        "Assist": [5, ""],
        "Assist/game": [1, ""],
        "Success Dribble/game": [8, ""],
        "Key Pass/game": [5, ""],
        "Successful Pass/game": [20, ""],
        "Long Ball/game": [10, ""],
        "Successful Crossing/game": [10, ""],
        "Ball Recovered/game": [10, ""],
        "Dribbled Past/game": [5, ""],
        "Clearance/game": [5,""],
        "Error leading to shot": [5, ""],
        "Error leading to shot/game": [5, ""],
        "Total duel won/game": [5, ""],
        "Aerial duel won/game": [5, ""],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        template.to_excel(writer, index=False, sheet_name="Template Dataset")

    # Kembalikan byte dari buffer
    buf.seek(0)
    return buf.getvalue()


def get_player_detail(season: str, player_name: str) -> dict | None:
    """
    Ambil 1 baris detail pemain untuk musim tertentu. Return dict atau None.
    """
    fields = [
        "player","team","nationality","position","age",
        "appearance","total_minute","total_goal","assist",
        "goal_per_game","shot_per_game","sot_per_game",
        "assist_per_game","successful_dribble_per_game","key_pass_per_game",
        "successful_pass_per_game","long_ball_per_game","successful_crossing_per_game",
        "ball_recovered_per_game","dribbled_past_per_game","clearance_per_game",
        "error","error_per_game","total_duel_per_game","aerial_duel_per_game",
    ]
    return (
        Player.objects
        .filter(dataset__season=season, player=player_name)
        .values(*fields)
        .first()
    )

def get_list_of_dataset():
    """
    Kembalikan list dict: id, league_name, season, player_count, uploaded_at
    """
    return list(
        Dataset.objects
        .annotate(player_count=Count('players'))
        .values('id', 'league_name', 'season', 'player_count', 'uploaded_at')
        .order_by('-uploaded_at')
    )

def delete_dataset(dataset_id: int) -> bool:
    """
    Hapus 1 dataset (cascade akan hapus players-nya).
    Return True jika ada yang terhapus.
    """
    deleted, _ = Dataset.objects.filter(id=dataset_id).delete()
    return deleted > 0


