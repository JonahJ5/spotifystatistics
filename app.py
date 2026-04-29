import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional
import random
import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Spotify Statistics", layout="wide")

st.title("Spotify Statistics")
st.caption("Created by Jonah Jutzi")
st.markdown("[View the GitHub repository](https://github.com/JonahJ5/spotifystatistics)")

st.caption("Request Extended Streaming History from Spotify: https://www.spotify.com/us/account/privacy/")
st.caption("This may take a few days to receive.")
st.caption("Upload the ZIP file Spotify gives you, usually named `my_spotify_data.zip`.")


ASSETS_DIR = Path(__file__).parent / "assets"
HELP_IMG = ASSETS_DIR / "spotify_directions.png"

with st.expander("⚠️ Important: Select *Extended streaming history*", expanded=True):
    if HELP_IMG.exists():
        st.image(str(HELP_IMG), caption="Select Extended streaming history (lifetime).", use_container_width=True)
    else:
        st.warning(f"Screenshot not found at: {HELP_IMG}")

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

# Derived time fields
df = add_time_fields(df)

return df

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

def format_hour_label(hour: int) -> str:
    """Convert 0-23 hour to user-friendly time labels."""
    if hour == 0:
        return "12 AM"
    if hour < 12:
        return f"{hour} AM"
    if hour == 12:
        return "12 PM"
    return f"{hour - 12} PM"


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable time fields for clearer charts."""
    d = df.copy()

    d["date"] = d["played_at"].dt.date
    d["day"] = d["played_at"].dt.floor("D")
    d["week"] = d["played_at"].dt.to_period("W").dt.start_time
    d["month"] = d["played_at"].dt.strftime("%Y-%m")
    d["year"] = d["played_at"].dt.year.astype("int64")

    d["hour"] = d["played_at"].dt.hour
    d["hour_label"] = d["hour"].apply(format_hour_label)

    d["day_of_week"] = d["played_at"].dt.day_name()
    d["dow"] = d["day_of_week"]  # keep old name so existing code still works

    return d


def make_example_data(n_rows: int = 2500) -> pd.DataFrame:
    """Create fake Spotify-style listening data so users can preview the dashboard."""
    rng = random.Random(42)

    artists = [
        "Arctic Monkeys", "SZA", "Kendrick Lamar", "Tame Impala", "Fleetwood Mac",
        "Drake", "Frank Ocean", "Taylor Swift", "The Weeknd", "Mac Miller",
        "Paramore", "Tyler, The Creator"
    ]

    tracks_by_artist = {
        "Arctic Monkeys": ["Do I Wanna Know?", "505", "R U Mine?", "Fluorescent Adolescent"],
        "SZA": ["Kill Bill", "Good Days", "Snooze", "Broken Clocks"],
        "Kendrick Lamar": ["HUMBLE.", "Money Trees", "DNA.", "Alright"],
        "Tame Impala": ["The Less I Know The Better", "Let It Happen", "Eventually", "Borderline"],
        "Fleetwood Mac": ["Dreams", "The Chain", "Go Your Own Way", "Landslide"],
        "Drake": ["Passionfruit", "God's Plan", "One Dance", "Headlines"],
        "Frank Ocean": ["Pink + White", "Nights", "Thinkin Bout You", "Lost"],
        "Taylor Swift": ["Cruel Summer", "Style", "Anti-Hero", "Blank Space"],
        "The Weeknd": ["Blinding Lights", "Save Your Tears", "Starboy", "Out of Time"],
        "Mac Miller": ["Good News", "Self Care", "Dang!", "Weekend"],
        "Paramore": ["Still Into You", "Hard Times", "Misery Business", "Ain't It Fun"],
        "Tyler, The Creator": ["See You Again", "EARFQUAKE", "WUSYANAME", "Sweet"]
    }

    artist_weights = [11, 10, 10, 9, 8, 8, 8, 8, 8, 7, 6, 5]
    hour_weights = [
        2, 1, 1, 1, 1, 2, 4, 6, 7, 6, 5, 5,
        6, 6, 7, 8, 9, 11, 13, 14, 12, 9, 6, 4
    ]

    today = pd.Timestamp.now(tz="UTC").floor("D")
    start = today - pd.Timedelta(days=365)

    rows = []
    for _ in range(n_rows):
        artist = rng.choices(artists, weights=artist_weights, k=1)[0]
        track = rng.choice(tracks_by_artist[artist])
        album = f"{artist} Essentials"

        day_offset = rng.randrange(365)
        hour = rng.choices(list(range(24)), weights=hour_weights, k=1)[0]
        minute = rng.randrange(60)

        played_at = start + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)
        ms_played = rng.randint(45_000, 240_000)

        rows.append({
            "played_at": played_at,
            "ms_played": ms_played,
            "track": track,
            "artist": artist,
            "album": album,
            "_source_file": "example_data"
        })

    example = pd.DataFrame(rows).sort_values("played_at").reset_index(drop=True)
    example["minutes"] = example["ms_played"] / 60000.0
    example = add_time_fields(example)

    return example


def period_settings(granularity: str):
    """Return the dataframe column and display label for a selected time grouping."""
    if granularity == "Day":
        return "day", "Day"
    if granularity == "Week":
        return "week", "Week"
    if granularity == "Month":
        return "month", "Month"
    return "year", "Year"
# -----------------------------
# Upload
# -----------------------------
# -----------------------------
# Upload / Example data
# -----------------------------
st.subheader("Get started")

use_example_data = st.toggle(
    "Use example data to preview the dashboard",
    value=True,
    help="Turn this off when you are ready to upload your own Spotify Extended Streaming History ZIP."
)

uploaded = st.file_uploader("Upload Spotify ZIP", type=["zip"])

if use_example_data:
    df_all = make_example_data()
    st.info(
        "You are viewing example data. Upload your own Spotify ZIP and turn off example data "
        "to generate your personal dashboard."
    )

else:
    if not uploaded:
        st.info("Upload your Spotify ZIP, or turn on example data to preview the dashboard.")
        st.stop()

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_ZIP_MB:
        st.error(f"ZIP too large ({size_mb:.1f} MB). Max is {MAX_ZIP_MB} MB.")
        st.stop()

    with st.spinner("Reading ZIP and parsing streaming history…"):
        df_all = load_spotify_from_zip(uploaded.read())

    if df_all.empty:
        st.error("After filtering, no valid music plays were found: track/artist/album missing or ms_played=0.")
        st.stop()

    st.success(f"Loaded {len(df_all):,} music play events.")

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
# ---- Time Patterns tab ----
with tab_time:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hour_order = [format_hour_label(h) for h in range(24)]

    c1, c2 = st.columns(2)

    with c1:
        by_dow = df.groupby("day_of_week", as_index=False)["minutes"].sum()
        by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], categories=order, ordered=True)
        by_dow = by_dow.sort_values("day_of_week")

        fig = px.bar(
            by_dow,
            x="day_of_week",
            y="minutes",
            title="Minutes by Day of the Week — UTC",
            color="day_of_week",
            labels={
                "day_of_week": "Day of the Week",
                "minutes": "Minutes"
            },
            custom_data=["day_of_week", "minutes"]
        )
        fig.update_traces(
            hovertemplate="Day of the Week: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_hour = df.groupby(["hour", "hour_label"], as_index=False)["minutes"].sum().sort_values("hour")
        by_hour["hour_label"] = pd.Categorical(by_hour["hour_label"], categories=hour_order, ordered=True)
        by_hour = by_hour.sort_values("hour_label")

        fig = px.bar(
            by_hour,
            x="hour_label",
            y="minutes",
            title="Minutes by Hour of Day — UTC",
            color="hour",
            labels={
                "hour_label": "Hour of Day",
                "minutes": "Minutes"
            },
            custom_data=["hour_label", "minutes"]
        )
        fig.update_traces(
            hovertemplate="Hour of Day: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most-listened days")

    daily_total = (
        df.groupby("day", as_index=False)
          .agg(
              total_minutes=("minutes", "sum"),
              number_of_artists=("artist", "nunique"),
              plays=("track", "size")
          )
          .sort_values("total_minutes", ascending=False)
          .head(25)
          .copy()
    )

    daily_total["date"] = daily_total["day"].dt.date
    daily_total["total_minutes"] = daily_total["total_minutes"].round(1)

    st.dataframe(
        daily_total[["date", "total_minutes", "number_of_artists", "plays"]],
        use_container_width=True,
        hide_index=True
    )

    day_options = daily_total["date"].astype(str).tolist()

    selected_day = st.selectbox(
        "Select one of your most-listened days to see more detail",
        options=day_options
    )

    selected_day_date = pd.to_datetime(selected_day).date()

    day_detail = df[df["date"] == selected_day_date].copy()

    with st.expander(f"Details for {selected_day}", expanded=True):
        day_k1, day_k2, day_k3 = st.columns(3)
        day_k1.metric("Total minutes", f"{day_detail['minutes'].sum():,.1f}")
        day_k2.metric("Unique artists", f"{day_detail['artist'].nunique():,}")
        day_k3.metric("Songs played", f"{len(day_detail):,}")

        top_day_artists = (
            day_detail.groupby("artist", as_index=False)["minutes"].sum()
            .sort_values("minutes", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_day_artists.sort_values("minutes", ascending=True),
            x="minutes",
            y="artist",
            orientation="h",
            title=f"Top Artists on {selected_day}",
            labels={
                "artist": "Artist",
                "minutes": "Minutes"
            },
            custom_data=["artist", "minutes"]
        )
        fig.update_traces(
            hovertemplate="Artist: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
        )
        fig.update_layout(
            showlegend=False,
            height=_auto_hbar_height(len(top_day_artists)),
            yaxis=dict(automargin=True),
            margin=dict(l=260, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        top_day_tracks = (
            day_detail.groupby(["track", "artist"], as_index=False)["minutes"].sum()
            .sort_values("minutes", ascending=False)
            .head(10)
        )
        top_day_tracks["minutes"] = top_day_tracks["minutes"].round(1)

        st.dataframe(
            top_day_tracks[["track", "artist", "minutes"]],
            use_container_width=True,
            hide_index=True
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
# ---- Trends tab ----
with tab_trends:
    st.subheader("Listening over time")

    trend_granularity = st.selectbox(
        "Choose time grouping for listening trend",
        options=["Day", "Week", "Year"],
        index=0
    )

    period_col, period_label = period_settings(trend_granularity)

    listening_trend = (
        df.groupby(period_col, as_index=False)["minutes"].sum()
          .sort_values(period_col)
    )
    listening_trend["hours"] = listening_trend["minutes"] / 60.0

    fig = px.line(
        listening_trend,
        x=period_col,
        y="hours",
        title=f"Hours Over Time by {period_label}",
        labels={
            period_col: period_label,
            "hours": "Hours"
        },
        custom_data=[period_col, "hours"]
    )
    fig.update_traces(
        hovertemplate=f"{period_label}: %{{customdata[0]}}<br>Hours: %{{customdata[1]:,.2f}}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

    cumulative = listening_trend.copy()
    cumulative["cumulative_hours"] = cumulative["hours"].cumsum()

    fig = px.line(
        cumulative,
        x=period_col,
        y="cumulative_hours",
        title=f"Cumulative Hours Over Time by {period_label}",
        labels={
            period_col: period_label,
            "cumulative_hours": "Cumulative Hours"
        },
        custom_data=[period_col, "cumulative_hours"]
    )
    fig.update_traces(
        hovertemplate=f"{period_label}: %{{customdata[0]}}<br>Cumulative Hours: %{{customdata[1]:,.2f}}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Artist trends")

    artist_granularity = st.selectbox(
        "Choose time grouping for artist trend charts",
        options=["Month", "Year"],
        index=0
    )

    artist_period_col, artist_period_label = period_settings(artist_granularity)

    top_artist_list = (
        df.groupby("artist", as_index=False)["minutes"].sum()
          .sort_values("minutes", ascending=False)
          .head(8)["artist"].tolist()
    )

    trend = (
        df[df["artist"].isin(top_artist_list)]
          .groupby([artist_period_col, "artist"], as_index=False)["minutes"].sum()
          .sort_values(artist_period_col)
    )
    trend["hours"] = trend["minutes"] / 60.0

    fig = px.area(
        trend,
        x=artist_period_col,
        y="hours",
        color="artist",
        title=f"Top Artists Over Time by {artist_period_label}",
        labels={
            artist_period_col: artist_period_label,
            "hours": "Hours",
            "artist": "Artist"
        },
        custom_data=[artist_period_col, "artist", "hours"]
    )
    fig.update_traces(
        hovertemplate=f"{artist_period_label}: %{{customdata[0]}}<br>Artist: %{{customdata[1]}}<br>Hours: %{{customdata[2]:,.2f}}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

    diversity = (
        df.groupby(artist_period_col, as_index=False)
          .agg(unique_artists=("artist", "nunique"))
          .sort_values(artist_period_col)
    )

    fig = px.line(
        diversity,
        x=artist_period_col,
        y="unique_artists",
        title=f"Artist Diversity Over Time by {artist_period_label}",
        labels={
            artist_period_col: artist_period_label,
            "unique_artists": "Unique Artists"
        },
        custom_data=[artist_period_col, "unique_artists"]
    )
    fig.update_traces(
        hovertemplate=f"{artist_period_label}: %{{customdata[0]}}<br>Unique Artists: %{{customdata[1]:,}}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

    first_artist = (
        df.sort_values("played_at")
          .groupby("artist", as_index=False)
          .first()[["artist", "played_at"]]
    )

    if artist_granularity == "Month":
        first_artist["period"] = first_artist["played_at"].dt.strftime("%Y-%m")
        discovery_period_label = "Month"
    else:
        first_artist["period"] = first_artist["played_at"].dt.year
        discovery_period_label = "Year"

    discovery = (
        first_artist.groupby("period", as_index=False)
          .size()
          .rename(columns={"size": "new_artists"})
          .sort_values("period")
    )

    fig = px.bar(
        discovery,
        x="period",
        y="new_artists",
        title=f"New Artists Discovered by {discovery_period_label}",
        labels={
            "period": discovery_period_label,
            "new_artists": "New Artists"
        },
        custom_data=["period", "new_artists"]
    )
    fig.update_traces(
        hovertemplate=f"{discovery_period_label}: %{{customdata[0]}}<br>New Artists: %{{customdata[1]:,}}<extra></extra>"
    )
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
