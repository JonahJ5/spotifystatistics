import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from spotify_app.config import MAX_FILES, MIN_STREAM_MS
from spotify_app.data_transform import add_time_fields


def is_safe_path(filename: str) -> bool:
    p = Path(filename)
    return (not p.is_absolute()) and (".." not in p.parts)


def _is_audio_streaming_history_json(name: str) -> bool:
    n = name.lower()

    if not n.endswith(".json"):
        return False

    if "streaming_history_audio" in n:
        return True

    if "streaming_history" in n and "video" not in n:
        return True

    return False


def load_spotify_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Parse Spotify streaming history JSON files from a ZIP into a DataFrame."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        infos = [i for i in zf.infolist() if is_safe_path(i.filename)]

        if len(infos) > MAX_FILES:
            raise ValueError(f"Too many files in zip ({len(infos)}).")

        targets = [i for i in infos if _is_audio_streaming_history_json(i.filename)]

        if not targets:
            raise ValueError(
                "Could not find any audio Streaming History JSON files in the ZIP.\n"
                "Look for files named like 'Streaming_History_Audio_*.json' inside your export."
            )

        rows = []

        for info in targets:
            with zf.open(info) as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue

            if isinstance(data, list):
                for ev in data:
                    if isinstance(ev, dict):
                        ev["_source_file"] = info.filename
                        rows.append(ev)

        if not rows:
            raise ValueError("No play events found in the ZIP.")

    df = pd.DataFrame(rows)

    rename_map = {
        "ts": "played_at",
        "ms_played": "ms_played",
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album",
    }

    df = df.rename(columns=rename_map)

    for col in ["played_at", "ms_played", "track", "artist", "album"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce", utc=True)
    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce")

    required = ["track", "artist", "album"]

    for c in required:
        df[c] = df[c].astype("string")

    df = df.dropna(subset=["played_at"])
    df = df[df["ms_played"].fillna(0) >= MIN_STREAM_MS]

    df = df.dropna(subset=required)
    df = df[
        (df["track"].str.strip() != "")
        & (df["artist"].str.strip() != "")
        & (df["album"].str.strip() != "")
    ].copy()

    df["minutes"] = df["ms_played"].fillna(0) / 60000.0
    df = add_time_fields(df)

    return df
