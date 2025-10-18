# players/services.py
from __future__ import annotations
from typing import List
import pandas as pd
from django.db import transaction
from .models import Dataset, Player

@transaction.atomic
def insert_dataset_and_players(league_name: str, season: str, df: pd.DataFrame) -> int:
    """
    Simpan/replace dataset + pemain. Return dataset_id.
    """
    cols = {c.lower(): c for c in df.columns}
    if "player_name" not in cols:
        raise ValueError("Kolom wajib 'player_name' tidak ditemukan pada file.")

    ds, created = Dataset.objects.get_or_create(
        league_name=league_name.strip(),
        season=season.strip()
    )
    if not created:
        ds.players.all().delete()

    bulk = []
    for _, row in df.iterrows():
        name = str(row[cols["player_name"]]).strip()
        team = str(row[cols["team"]]).strip() if "team" in cols else None
        pos  = str(row[cols["position"]]).strip() if "position" in cols else None
        bulk.append(Player(dataset=ds, player_name=name, team=team, position=pos))
    if bulk:
        Player.objects.bulk_create(bulk, batch_size=1000)
    return ds.id

def get_seasons() -> List[str]:
    return list(
        Dataset.objects.values_list("season", flat=True).distinct().order_by("season")
    )

def get_players_by_season(season: str) -> List[str]:
    return list(
        Player.objects.filter(dataset__season=season)
        .order_by("player_name")
        .values_list("player_name", flat=True)
    )

# utils/files.py
import io
from datetime import datetime

def make_template_excel_bytes() -> bytes:
    template = pd.DataFrame({
        "player_name": ["Contoh A", "Contoh B"],
        "team": ["Contoh FC", "Demo United"],
        "position": ["AM", "AM"],
        "goals_per90": [0.25, 0.15],
        "assists_per90": [0.18, 0.22],
        "key_passes_per90": [2.1, 1.6],
        "shot_creating_actions_per90": [3.5, 2.8],
        "passes_completed_pct": [82.0, 79.5],
        "duels_won": [28, 24],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        template.to_excel(writer, index=False, sheet_name="Template Dataset")

    # Kembalikan byte dari buffer
    buf.seek(0)
    return buf.getvalue()
