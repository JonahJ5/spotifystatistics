import io
import json
import zipfile
from typing import Dict

import pandas as pd

from spotify_app.config import MIN_STREAM_SECONDS
from spotify_app.data_transform import safe_group_sum


def build_exports(df: pd.DataFrame, topn: int) -> Dict[str, bytes]:
    exports: Dict[str, bytes] = {}

    summary = {
        "rows": int(len(df)),
        "start_utc": str(df["played_at"].min()) if len(df) else None,
        "end_utc": str(df["played_at"].max()) if len(df) else None,
        "total_minutes": float(df["minutes"].sum()),
        "track_plays": int(len(df)),
        "unique_artists": int(df["artist"].nunique()),
        "unique_tracks": int(df["track"].nunique()),
        "unique_albums": int(df["album"].nunique()),
        "minimum_stream_seconds": MIN_STREAM_SECONDS,
    }

    top_artists = safe_group_sum(df, ["artist"], "minutes", topn)
    top_tracks = safe_group_sum(df, ["track", "artist", "album"], "minutes", topn)
    top_albums = safe_group_sum(df, ["album", "artist"], "minutes", topn)

    summary["top_artists"] = top_artists.to_dict(orient="records")
    summary["top_tracks"] = top_tracks.to_dict(orient="records")
    summary["top_albums"] = top_albums.to_dict(orient="records")

    exports["wrapped_summary.json"] = json.dumps(summary, indent=2).encode("utf-8")
    exports["top_artists.csv"] = top_artists.to_csv(index=False).encode("utf-8")
    exports["top_tracks.csv"] = top_tracks.to_csv(index=False).encode("utf-8")
    exports["top_albums.csv"] = top_albums.to_csv(index=False).encode("utf-8")

    daily = df.groupby("day", as_index=False)["minutes"].sum().sort_values("day")
    exports["daily_minutes.csv"] = daily.to_csv(index=False).encode("utf-8")

    monthly = df.groupby("month", as_index=False)["minutes"].sum().sort_values("month")
    exports["monthly_minutes.csv"] = monthly.to_csv(index=False).encode("utf-8")

    return exports


def make_zip_bytes(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)

    buf.seek(0)
    return buf.read()
