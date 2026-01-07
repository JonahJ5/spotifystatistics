import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Spotify Wrapped (Upload ZIP)", layout="wide")
st.title("Spotify Wrapped-style Dashboard")
st.caption("Upload the ZIP file Spotify gives you (my_spotify_data.zip).")


# -----------------------------
# Constants
# -----------------------------
MAX_ZIP_MB = 300
MAX_FILES = 500
DEFAULT_TOPN = 15


# -----------------------------
# Helpers
# -----------------------------
def is_safe_path(filename: str) -> bool:
    p = Path(filename)
    return (not p.is_absolute()) and (".." not in p.parts)


def _is_audio_streaming_history_json(name: str) -> bool:
    n = name.lower()
    if not n.endswith(".json"):
        return False
    # strongly prefer the explicit audio files
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

            # Streaming history files are usually a list[dict]
            if isinstance(data, list):
                for ev in data:
                    if isinstance(ev, dict):
                        ev["_source_file"] = info.filename
                        rows.append(ev)

        if not rows:
            raise ValueError("No play events found in the ZIP (audio streaming history files were present).")

    df = pd.DataFrame(rows)

    # No
