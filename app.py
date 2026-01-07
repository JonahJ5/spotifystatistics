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
st.title("Spotify Statistics")
st.caption("request extended streaming history from spotify (https://www.spotify.com/us/account/privacy/")
st.caption("this may take a few days to receive")
st.caption("Upload the ZIP file Spotify gives you (my_spotify_data.zip).")



# -----------------------------
# Constants
# -----------------------------
MAX_ZIP_MB = 300
MAX_FILES = 500
DEFAULT_TOPN = 15
SESSION_GAP_MINUTES = 30  # fixed (no slider)


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


def _auto_hbar_height(n_rows: int, min_h: int = 450, per_row: int = 26, pad: int = 140, max_h: int = 2200) -> int:
    """Make horizontal bar charts tall enough so labels don't get cut off when Top N grows."""
    return min(max_h, max(min_h, pad + per_row * max(1, n_rows)))


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
            raise ValueError("No play events found in the ZIP (audio streaming history files were present).")

    df = pd.DataFrame(rows)

    # Normalize Spotify fields
    rename_map = {
        "ts": "played_at",
        "ms_played": "ms_played",
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album",
        "conn_country": "conn_country",
    }
    df = df.rename(columns=rename_map)

    # Ensure core columns exist AFTER rename
    for col in ["played_at", "ms_played", "track", "artist", "album"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Parse types
    df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce", utc=True)
    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce")

    # Keep only real music plays (no missing metadata, ms_played > 0)
    required = ["track", "artist", "album"]
    for c in required:
        df[c] = df[c].astype("string")

    df = df.dropna(subset=["played_at"])
    df = df[df["ms_played"].fillna(0) > 0]

    df = df.dropna(subset=required)
    df = df[
        (df["track"].str.strip() != "") &
        (df["artist"].str.strip() != "") &
        (df["album"].str.strip() != "")
    ].copy()

    # Derived metrics
    df["minutes"] = df["ms_played"].fillna(0) / 60000.0

    # Derived time parts (UTC)
    df["date"] = df["played_at"].dt.date
    df["day"] = df["played_at"].dt.floor("D")
    df["month"] = df["played_at"].dt.strftime("%Y-%m")
    df["year"] = df["played_at"].dt.year.astype("int64")
    df["hour"] = df["played_at"].dt.hour
    df["dow"] = df["played_at"].dt.day_name()

    return df


def safe_group_sum(df: pd.DataFrame, group_cols, value_col: str, topn: Optional[int] = None) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, as_index=False)[value_col].sum()
        .sort_values(value_col, ascending=False)
    )
    return out.head(topn) if topn else out


def safe_group_count(df: pd.DataFrame, group_cols, topn: Optional[int] = None, name: str = "count") -> pd.DataFrame:
    out = df.groupby(group_cols, as_index=False).size().rename(columns={"size": name})
    out = out.sort_values(name, ascending=False)
    return out.head(topn) if topn else out


def build_exports(df: pd.DataFrame, topn: int) -> Dict[str, bytes]:
    exports: Dict[str, bytes] = {}

    summary = {
        "rows": int(len(df)),
        "start_utc": str(df["played_at"].min()) if len(df) else None,
        "end_utc": str(df["played_at"].max()) if len(df) else None,
        "total_minutes": float(df["minutes"].sum()),
        "unique_artists": int(df["artist"].nunique()),
        "unique_tracks": int(df["track"].nunique()),
        "unique_albums": int(df["album"].nunique()),
    }

    top_artists = safe_group_sum(df, ["artist"], "minutes", topn)
    top_tracks = safe_group_sum(df, ["track", "artist"], "minutes", topn)
    top_albums = safe_group_sum(df, ["album"], "minutes", topn)

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


def compute_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    if df["played_at"].isna().all():
        return pd.DataFrame()

    d = df.sort_values("played_at").copy()
    gaps = d["played_at"].diff().dt.total_seconds().fillna(0) / 60.0
    d["new_session"] = (gaps > gap_minutes).astype(int)
    d["session_id"] = d["new_session"].cumsum()

    sessions = d.groupby("session_id", as_index=False).agg(
        session_start=("played_at", "min"),
        session_end=("played_at", "max"),
        plays=("played_at", "size"),
        minutes=("minutes", "sum"),
    )
    sessions["duration_minutes"] = (sessions["session_end"] - sessions["session_start"]).dt.total_seconds() / 60.0
    sessions["session_date"] = sessions["session_start"].dt.date
    return sessions.sort_values("session_start")


# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload Spotify ZIP", type=["zip"])

if not uploaded:
    st.info("Upload your Spotify ZIP to generate your dashboard.")
    st.stop()

size_mb = uploaded.size / (1024 * 1024)
if size_mb > MAX_ZIP_MB:
    st.error(f"ZIP too large ({size_mb:.1f} MB). Max is {MAX_ZIP_MB} MB.")
    st.stop()

with st.spinner("Reading ZIP and parsing streaming history…"):
    df_all = load_spotify_from_zip(uploaded.read())

if df_all.empty:
    st.error("After filtering, no valid music plays were found (track/artist/album missing or ms_played=0).")
    st.stop()

st.success(f"Loaded {len(df_all):,} music play events (audio only, metadata required).")


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    topn = st.slider("Top N", min_value=5, max_value=50, value=DEFAULT_TOPN, step=5)

    years = sorted(df_all["year"].dropna().unique().tolist())
    year_options = ["All years (Select all)"] + [str(y) for y in years]
    selected_year = st.selectbox("Year", options=year_options, index=0)

    show_preview = st.checkbox("Show preview table", value=False)

# Apply year filter
df = df_all.copy()
if selected_year != "All years (Select all)":
    df = df[df["year"] == int(selected_year)].copy()

if df.empty:
    st.warning("No rows for that selection.")
    st.stop()


# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total minutes", f"{df['minutes'].sum():,.0f}")
k2.metric("Unique artists", f"{df['artist'].nunique():,}")
k3.metric("Unique tracks", f"{df['track'].nunique():,}")
k4.metric("Unique albums", f"{df['album'].nunique():,}")

if show_preview:
    st.subheader("Preview (filtered)")
    st.dataframe(df.head(100), use_container_width=True)


# -----------------------------
# Tabs
# -----------------------------
tab_rank, tab_time, tab_trends, tab_sessions = st.tabs(
    ["Rankings", "Time Patterns", "Trends", "Sessions & Behavior"]
)

# ---- Rankings tab ----
with tab_rank:
    c1, c2 = st.columns(2)

    with c1:
        top_artists = safe_group_sum(df, ["artist"], "minutes", topn)
        plot_df = top_artists.sort_values("minutes", ascending=True)
        fig = px.bar(
            plot_df, x="minutes", y="artist", orientation="h",
            title="Top Artists (by minutes)",
            color="artist"
        )
        fig.update_layout(
            showlegend=False,
            height=_auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=260, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_tracks = safe_group_sum(df, ["track", "artist"], "minutes", topn)
        plot_df = top_tracks.sort_values("minutes", ascending=True)
        fig = px.bar(
            plot_df, x="minutes", y="track", orientation="h",
            title="Top Tracks (by minutes)",
            hover_data=["artist"],
            color="track"
        )
        fig.update_layout(
            showlegend=False,
            height=_auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=320, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        top_albums = safe_group_sum(df, ["album"], "minutes", topn)
        plot_df = top_albums.sort_values("minutes", ascending=True)
        fig = px.bar(
            plot_df, x="minutes", y="album", orientation="h",
            title="Top Albums (by minutes)",
            color="album"
        )
        fig.update_layout(
            showlegend=False,
            height=_auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=320, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        top_artists_plays = safe_group_count(df, ["artist"], topn, name="plays")
        plot_df = top_artists_plays.sort_values("plays", ascending=True)
        fig = px.bar(
            plot_df, x="plays", y="artist", orientation="h",
            title="Top Artists (by play count)",
            color="artist"
        )
        fig.update_layout(
            showlegend=False,
            height=_auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=260, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    plays_per_track = df.groupby(["track", "artist"], as_index=False).size().rename(columns={"size": "plays"})
    st.plotly_chart(
        px.histogram(plays_per_track, x="plays", nbins=40, title="Track repeat distribution (plays per track)"),
        use_container_width=True
    )

    # (4) Artist peak day ranking (each artist's single biggest day)
    st.subheader("Artist peak days (each artist’s biggest single day)")
    daily_artist = df.groupby(["artist", "day"], as_index=False)["minutes"].sum()

    idx = daily_artist.groupby("artist")["minutes"].idxmax()
    artist_peaks = daily_artist.loc[idx].copy()

    artist_peaks = artist_peaks.sort_values("minutes", ascending=False).head(topn).copy()
    artist_peaks["date"] = artist_peaks["day"].dt.date
    artist_peaks["label"] = artist_peaks["artist"] + " — " + artist_peaks["date"].astype(str)
    artist_peaks["peak_hours"] = artist_peaks["minutes"] / 60.0

    plot_df = artist_peaks.sort_values("minutes", ascending=True)
    fig = px.bar(
        plot_df,
        x="minutes",
        y="label",
        orientation="h",
        title="Top artist-days (each artist’s #1 day)",
        hover_data={"minutes": ":.0f", "peak_hours": ":.2f"},
        color="artist"
    )
    fig.update_layout(
        showlegend=False,
        height=_auto_hbar_height(len(plot_df)),
        yaxis=dict(automargin=True),
        margin=dict(l=340, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---- Time Patterns tab ----
with tab_time:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    heat = df.groupby(["dow", "hour"], as_index=False)["minutes"].sum()
    heat["dow"] = pd.Categorical(heat["dow"], categories=order, ordered=True)
    heat = heat.sort_values(["dow", "hour"])

    st.plotly_chart(
        px.density_heatmap(heat, x="hour", y="dow", z="minutes", title="Listening heatmap (minutes) — UTC"),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        by_dow = df.groupby("dow", as_index=False)["minutes"].sum()
        by_dow["dow"] = pd.Categorical(by_dow["dow"], categories=order, ordered=True)
        by_dow = by_dow.sort_values("dow")
        fig = px.bar(by_dow, x="dow", y="minutes", title="Minutes by day of week — UTC", color="dow")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_hour = df.groupby("hour", as_index=False)["minutes"].sum().sort_values("hour")
        fig = px.bar(by_hour, x="hour", y="minutes", title="Minutes by hour of day — UTC", color="hour")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # (3) Most-listened days: date-only + top artist + minutes
    daily_total = (
        df.groupby("day", as_index=False)["minutes"].sum()
          .rename(columns={"minutes": "total_minutes"})
    )
    daily_artist2 = (
        df.groupby(["day", "artist"], as_index=False)["minutes"].sum()
          .rename(columns={"minutes": "top_artist_minutes"})
    )

    idx2 = daily_artist2.groupby("day")["top_artist_minutes"].idxmax()
    daily_top_artist = daily_artist2.loc[idx2].rename(columns={"artist": "top_artist"}).copy()

    top_days = (
        daily_total.merge(daily_top_artist[["day", "top_artist", "top_artist_minutes"]], on="day", how="left")
        .sort_values("total_minutes", ascending=False)
        .head(25)
        .copy()
    )
    top_days["date"] = top_days["day"].dt.date
    top_days["total_minutes"] = top_days["total_minutes"].round(1)
    top_days["top_artist_minutes"] = top_days["top_artist_minutes"].round(1)

    st.subheader("Most-listened days (with top artist)")
    st.dataframe(
        top_days[["date", "total_minutes", "top_artist", "top_artist_minutes"]],
        use_container_width=True
    )


# ---- Trends tab ----
with tab_trends:
    daily = df.groupby("day", as_index=False)["minutes"].sum().sort_values("day")
    st.plotly_chart(px.line(daily, x="day", y="minutes", title="Daily minutes over time"), use_container_width=True)

    monthly = df.groupby("month", as_index=False)["minutes"].sum().sort_values("month")
    fig = px.bar(monthly, x="month", y="minutes", title="Minutes by month", color="month")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    daily3 = daily.copy()
    daily3["cumulative_minutes"] = daily3["minutes"].cumsum()
    st.plotly_chart(px.line(daily3, x="day", y="cumulative_minutes", title="Cumulative minutes over time"), use_container_width=True)

    top_artist_list = (
        df.groupby("artist", as_index=False)["minutes"].sum()
          .sort_values("minutes", ascending=False)
          .head(8)["artist"].tolist()
    )
    trend = (
        df[df["artist"].isin(top_artist_list)]
          .groupby(["month", "artist"], as_index=False)["minutes"].sum()
          .sort_values("month")
    )
    st.plotly_chart(px.area(trend, x="month", y="minutes", color="artist", title="Top artists over time (monthly)"),
                    use_container_width=True)

    diversity = df.groupby("day", as_index=False).agg(unique_artists=("artist", "nunique")).sort_values("day")
    st.plotly_chart(px.line(diversity, x="day", y="unique_artists", title="Artist diversity over time"), use_container_width=True)

    first_artist = (
        df.sort_values("played_at")
          .groupby("artist", as_index=False)
          .first()[["artist", "played_at"]]
    )
    first_artist["month"] = first_artist["played_at"].dt.strftime("%Y-%m")
    discovery = (
        first_artist.groupby("month", as_index=False)
          .size()
          .rename(columns={"size": "new_artists"})
          .sort_values("month")
    )
    fig = px.bar(discovery, x="month", y="new_artists", title="New artists discovered by month", color="month")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ---- Sessions tab ----
with tab_sessions:
    sessions = compute_sessions(df, gap_minutes=SESSION_GAP_MINUTES)
    if sessions.empty:
        st.info("No sessions computed.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.histogram(
                    sessions,
                    x="duration_minutes",
                    nbins=40,
                    title=f"Session duration distribution (gap>{SESSION_GAP_MINUTES} min starts new session)"
                ),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                px.histogram(sessions, x="minutes", nbins=40, title="Minutes per session distribution"),
                use_container_width=True
            )

        st.subheader("Longest sessions")
        longest = sessions.sort_values("minutes", ascending=False).head(20).copy()
        longest["minutes"] = longest["minutes"].round(1)
        longest["duration_minutes"] = longest["duration_minutes"].round(1)
        st.dataframe(
            longest[["session_date", "minutes", "duration_minutes", "plays"]],
            use_container_width=True
        )

        # (6) Hover shows date + start/end
        fig = px.scatter(
            sessions,
            x="plays",
            y="minutes",
            title="Plays vs minutes per session",
            hover_data={
                "session_date": True,
                "session_start": True,
                "session_end": True,
                "duration_minutes": ":.1f",
                "plays": True,
                "minutes": ":.1f",
                "session_id": False,
            },
        )
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Export section
# -----------------------------
st.divider()
st.subheader("Export")

exports = build_exports(df, topn=topn)

st.download_button(
    "Download Wrapped Summary (JSON)",
    data=exports["wrapped_summary.json"],
    file_name="wrapped_summary.json",
    mime="application/json",
)

csv_cols = st.columns(3)
csv_files = ["top_artists.csv", "top_tracks.csv", "top_albums.csv"]
csv_labels = ["Top Artists (CSV)", "Top Tracks (CSV)", "Top Albums (CSV)"]

for col, fname, label in zip(csv_cols, csv_files, csv_labels):
    with col:
        st.download_button(label, data=exports[fname], file_name=fname, mime="text/csv")

zip_bytes = make_zip_bytes(exports)
st.download_button(
    "Download ALL exports as ZIP",
    data=zip_bytes,
    file_name="wrapped_exports.zip",
    mime="application/zip",
)
