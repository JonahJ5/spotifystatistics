import pandas as pd
import plotly.express as px
import streamlit as st

from spotify_app.config import (
    DEFAULT_TOPN,
    HELP_IMG,
    MAX_ZIP_MB,
    MIN_STREAM_SECONDS,
    SESSION_GAP_MINUTES,
    TIMEZONE_OPTIONS,
)
from spotify_app.data_loader import load_spotify_from_zip
from spotify_app.data_transform import (
    add_time_fields,
    auto_hbar_height,
    compute_sessions,
    format_hour_label,
    period_settings,
    safe_group_count,
    safe_group_sum,
)
from spotify_app.example_data import make_example_data
from spotify_app.exports import build_exports, make_zip_bytes
from spotify_app.pdf_report import build_shareable_pdf


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Spotify Statistics", layout="wide")

st.title("Spotify Statistics")
st.caption("Created by Jonah Jutzi")

st.markdown(
    """
    Explore your Spotify listening history through interactive charts, rankings, trends, and session-based insights.

    This app uses your **Spotify Extended Streaming History** export to show your top artists, tracks, albums,
    listening patterns, repeated songs, and listening sessions. If you have not uploaded your data yet, the dashboard
    will show example data so you can preview how everything works.

    **Need your data?** Request your Spotify Extended Streaming History here:  
    [Spotify Privacy / Account Data Request](https://www.spotify.com/us/account/privacy/)

    [View the GitHub repository](https://github.com/JonahJ5/spotifystatistics)
    """
)


# -----------------------------
# Upload / Example data
# -----------------------------
st.subheader("Get started")

uploaded = st.file_uploader(
    "Upload Spotify ZIP",
    type=["zip"],
    help="Upload your Spotify Extended Streaming History ZIP to replace the example dashboard with your personal data.",
)

if uploaded:
    size_mb = uploaded.size / (1024 * 1024)

    if size_mb > MAX_ZIP_MB:
        st.error(f"ZIP too large ({size_mb:.1f} MB). Max is {MAX_ZIP_MB} MB.")
        st.stop()

    with st.spinner("Reading ZIP and parsing streaming history…"):
        df_all = load_spotify_from_zip(uploaded.read())

    if df_all.empty:
        st.error(
            f"After filtering, no valid music plays were found. "
            f"Tracks must have valid metadata and at least {MIN_STREAM_SECONDS} seconds played."
        )
        st.stop()

    st.success(f"Loaded {len(df_all):,} music play events from your Spotify data.")

else:
    st.info(
        "To use your own data, request Extended Streaming History from Spotify, wait for the export, "
        "then upload the ZIP file Spotify provides. After uploading, select the timezone that best matches "
        "where you usually listen so day-of-week and hour-of-day charts are accurate."
    )

    st.markdown(
        "[Request your Spotify Extended Streaming History here](https://www.spotify.com/us/account/privacy/)"
    )

    with st.expander("How to request the correct Spotify data", expanded=False):
        st.markdown(
            """
            When requesting your Spotify data, make sure to select **Extended streaming history**.
            This is different from the smaller account-data export.
            """
        )

        if HELP_IMG.exists():
            st.image(
                str(HELP_IMG),
                caption="Select Extended streaming history.",
                width=550,
            )
        else:
            st.warning(f"Screenshot not found at: {HELP_IMG}")

    df_all = make_example_data()

    st.info(
        "Showing example data so you can preview the dashboard. "
        "Upload your Spotify ZIP above to replace this with your personal listening history."
    )


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    topn = st.slider("Top N", min_value=5, max_value=50, value=DEFAULT_TOPN, step=5)

    selected_timezone_label = st.selectbox(
        "Timezone",
        options=list(TIMEZONE_OPTIONS.keys()),
        index=0,
        help="Used for day-of-week, hour-of-day, most-listened days, and trend date grouping.",
    )

    selected_timezone = TIMEZONE_OPTIONS[selected_timezone_label]

    df_all = add_time_fields(df_all, timezone=selected_timezone)

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
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total minutes", f"{df['minutes'].sum():,.0f}")
k2.metric("Track plays", f"{len(df):,}")
k3.metric("Unique artists", f"{df['artist'].nunique():,}")
k4.metric("Unique tracks", f"{df['track'].nunique():,}")
k5.metric("Unique albums", f"{df['album'].nunique():,}")

if show_preview:
    st.subheader("Preview (filtered)")
    st.dataframe(df.head(100), use_container_width=True)


# -----------------------------
# Tabs
# -----------------------------
tab_rank, tab_time, tab_trends, tab_sessions, tab_info = st.tabs(
    ["Rankings", "Time Patterns", "Trends", "Sessions & Behavior", "Info"]
)


# -----------------------------
# Rankings tab
# -----------------------------
with tab_rank:
    c1, c2 = st.columns(2)

    with c1:
        top_artists = safe_group_sum(df, ["artist"], "minutes", topn)
        plot_df = top_artists.sort_values("minutes", ascending=True)

        fig = px.bar(
            plot_df,
            x="minutes",
            y="artist",
            orientation="h",
            title="Top Artists by Minutes",
            color="artist",
            labels={"artist": "Artist", "minutes": "Minutes"},
            custom_data=["artist", "minutes"],
        )

        fig.update_traces(
            hovertemplate="Artist: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
        )

        fig.update_layout(
            showlegend=False,
            height=auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=260, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_tracks = safe_group_sum(df, ["track", "artist", "album"], "minutes", topn)
        plot_df = top_tracks.sort_values("minutes", ascending=True)

        fig = px.bar(
            plot_df,
            x="minutes",
            y="track",
            orientation="h",
            title="Top Tracks by Minutes",
            color="track",
            labels={"track": "Track", "artist": "Artist", "album": "Album", "minutes": "Minutes"},
            custom_data=["track", "artist", "album", "minutes"],
        )

        fig.update_traces(
            hovertemplate=(
                "Track: %{customdata[0]}<br>"
                "Artist: %{customdata[1]}<br>"
                "Album: %{customdata[2]}<br>"
                "Minutes: %{customdata[3]:,.1f}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            showlegend=False,
            height=auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=320, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        top_albums = safe_group_sum(df, ["album", "artist"], "minutes", topn)
        plot_df = top_albums.sort_values("minutes", ascending=True)

        fig = px.bar(
            plot_df,
            x="minutes",
            y="album",
            orientation="h",
            title="Top Albums by Minutes",
            color="album",
            labels={"album": "Album", "artist": "Artist", "minutes": "Minutes"},
            custom_data=["album", "artist", "minutes"],
        )

        fig.update_traces(
            hovertemplate=(
                "Album: %{customdata[0]}<br>"
                "Artist: %{customdata[1]}<br>"
                "Minutes: %{customdata[2]:,.1f}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            showlegend=False,
            height=auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=320, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    with c4:
        top_artists_plays = safe_group_count(df, ["artist"], topn, name="plays")
        plot_df = top_artists_plays.sort_values("plays", ascending=True)

        fig = px.bar(
            plot_df,
            x="plays",
            y="artist",
            orientation="h",
            title="Top Artists by Play Count",
            color="artist",
            labels={"artist": "Artist", "plays": "Plays"},
            custom_data=["artist", "plays"],
        )

        fig.update_traces(
            hovertemplate="Artist: %{customdata[0]}<br>Plays: %{customdata[1]:,}<extra></extra>"
        )

        fig.update_layout(
            showlegend=False,
            height=auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=260, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    plays_per_track = (
        df.groupby(["track", "artist"], as_index=False)
        .size()
        .rename(columns={"size": "plays"})
    )

    repeat_dist = (
        plays_per_track.groupby("plays", as_index=False)
        .size()
        .rename(columns={"size": "number_of_tracks"})
        .sort_values("plays")
    )

    fig = px.bar(
        repeat_dist,
        x="plays",
        y="number_of_tracks",
        title="Track Repeat Distribution",
        labels={"plays": "Plays per Track", "number_of_tracks": "Number of Tracks"},
        custom_data=["plays", "number_of_tracks"],
    )

    fig.update_traces(
        hovertemplate=(
            "Plays per Track: %{customdata[0]}<br>"
            "Number of Tracks: %{customdata[1]:,}"
            "<extra></extra>"
        )
    )

    fig.update_layout(xaxis_title="Plays per Track", yaxis_title="Number of Tracks")

    st.plotly_chart(fig, use_container_width=True)

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
        title="Top Artist Days",
        color="artist",
        labels={"label": "Artist and Date", "minutes": "Minutes", "peak_hours": "Hours"},
        custom_data=["artist", "date", "minutes", "peak_hours"],
    )

    fig.update_traces(
        hovertemplate=(
            "Artist: %{customdata[0]}<br>"
            "Date: %{customdata[1]}<br>"
            "Minutes: %{customdata[2]:,.1f}<br>"
            "Hours: %{customdata[3]:,.2f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        showlegend=False,
        height=auto_hbar_height(len(plot_df)),
        yaxis=dict(automargin=True),
        margin=dict(l=340, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Time Patterns tab
# -----------------------------
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
            title=f"Minutes by Day of the Week — {selected_timezone_label}",
            color="day_of_week",
            labels={"day_of_week": "Day of the Week", "minutes": "Minutes"},
            custom_data=["day_of_week", "minutes"],
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
            title=f"Minutes by Hour of Day — {selected_timezone_label}",
            color="hour",
            labels={"hour_label": "Hour of Day", "minutes": "Minutes"},
            custom_data=["hour_label", "minutes"],
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
            plays=("track", "size"),
        )
        .sort_values("total_minutes", ascending=False)
        .head(25)
        .copy()
    )

    daily_total["date"] = daily_total["day"].dt.date
    daily_total["total_minutes"] = daily_total["total_minutes"].round(1)

    daily_display = daily_total[["date", "total_minutes", "number_of_artists", "plays"]].rename(
        columns={
            "date": "Date",
            "total_minutes": "Total Minutes",
            "number_of_artists": "Number of Artists",
            "plays": "Track Plays",
        }
    )

    st.dataframe(daily_display, use_container_width=True, hide_index=True)

    day_options = daily_total["date"].astype(str).tolist()

    selected_day = st.selectbox(
        "Select one of your most-listened days to see more detail",
        options=day_options,
    )

    selected_day_date = pd.to_datetime(selected_day).date()
    day_detail = df[df["date"] == selected_day_date].copy()

    with st.expander(f"Details for {selected_day}", expanded=True):
        day_k1, day_k2, day_k3 = st.columns(3)

        day_k1.metric("Total minutes", f"{day_detail['minutes'].sum():,.1f}")
        day_k2.metric("Unique artists", f"{day_detail['artist'].nunique():,}")
        day_k3.metric("Track plays", f"{len(day_detail):,}")

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
            labels={"artist": "Artist", "minutes": "Minutes"},
            custom_data=["artist", "minutes"],
        )

        fig.update_traces(
            hovertemplate="Artist: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
        )

        fig.update_layout(
            showlegend=False,
            height=auto_hbar_height(len(top_day_artists)),
            yaxis=dict(automargin=True),
            margin=dict(l=260, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        top_day_tracks = (
            day_detail.groupby(["track", "artist"], as_index=False)
            .size()
            .rename(columns={"size": "track_plays"})
            .sort_values("track_plays", ascending=False)
            .head(10)
        )

        top_day_tracks_display = top_day_tracks.rename(
            columns={"track": "Track", "artist": "Artist", "track_plays": "Track Plays"}
        )

        st.dataframe(
            top_day_tracks_display[["Track", "Artist", "Track Plays"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Most repeated songs")
    repeat_sequence = df.sort_values("played_at").copy()
    repeat_sequence["track_key"] = repeat_sequence["track"].astype(str) + " — " + repeat_sequence["artist"].astype(str)
    repeat_sequence["new_streak"] = repeat_sequence["track_key"] != repeat_sequence["track_key"].shift()
    repeat_sequence["streak_id"] = repeat_sequence["new_streak"].cumsum()

    streaks = (
        repeat_sequence.groupby("streak_id", as_index=False)
        .agg(
            track=("track", "first"),
            artist=("artist", "first"),
            repeat_count=("track", "size"),
            streak_start=("played_at_local", "min"),
            streak_end=("played_at_local", "max"),
        )
    )

    streaks = streaks[streaks["repeat_count"] > 1].copy()
    streaks = streaks.sort_values("repeat_count", ascending=False).head(topn)

    if streaks.empty:
        st.info("No consecutive repeated tracks found for the current selection.")
    else:
        streaks["label"] = streaks["track"] + " — " + streaks["artist"]
        streaks["streak_start_display"] = streaks["streak_start"].dt.strftime("%b %d, %Y %I:%M %p")
        streaks["streak_end_display"] = streaks["streak_end"].dt.strftime("%b %d, %Y %I:%M %p")

        plot_df = streaks.sort_values("repeat_count", ascending=True)

        fig = px.bar(
            plot_df,
            x="repeat_count",
            y="label",
            orientation="h",
            title="Most Consecutively Repeated Songs",
            labels={"repeat_count": "Consecutive Plays", "label": "Track"},
            custom_data=["track", "artist", "repeat_count", "streak_start_display", "streak_end_display"],
        )

        fig.update_traces(
            hovertemplate=(
                "Track: %{customdata[0]}<br>"
                "Artist: %{customdata[1]}<br>"
                "Consecutive Plays: %{customdata[2]:,}<br>"
                "Start: %{customdata[3]}<br>"
                "End: %{customdata[4]}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            showlegend=False,
            height=auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=340, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Trends tab
# -----------------------------
with tab_trends:
    st.subheader("Listening over time")

    trend_granularity = st.selectbox(
        "Choose time grouping for listening trend",
        options=["Day", "Week", "Year"],
        index=0,
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
        labels={period_col: period_label, "hours": "Hours"},
        custom_data=[period_col, "hours"],
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
        labels={period_col: period_label, "cumulative_hours": "Cumulative Hours"},
        custom_data=[period_col, "cumulative_hours"],
    )

    fig.update_traces(
        hovertemplate=f"{period_label}: %{{customdata[0]}}<br>Cumulative Hours: %{{customdata[1]:,.2f}}<extra></extra>"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Artist trends")

    artist_granularity = st.selectbox(
        "Choose time grouping for artist trend charts",
        options=["Month", "Year"],
        index=0,
    )

    artist_period_col, artist_period_label = period_settings(artist_granularity)

    top_artist_list = (
        df.groupby("artist", as_index=False)["minutes"].sum()
        .sort_values("minutes", ascending=False)
        .head(8)["artist"]
        .tolist()
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
        labels={artist_period_col: artist_period_label, "hours": "Hours", "artist": "Artist"},
        custom_data=[artist_period_col, "artist", "hours"],
    )

    fig.update_traces(
        hovertemplate=(
            f"{artist_period_label}: %{{customdata[0]}}<br>"
            "Artist: %{customdata[1]}<br>"
            "Hours: %{customdata[2]:,.2f}"
            "<extra></extra>"
        )
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
        labels={artist_period_col: artist_period_label, "unique_artists": "Unique Artists"},
        custom_data=[artist_period_col, "unique_artists"],
    )

    fig.update_traces(
        hovertemplate=(
            f"{artist_period_label}: %{{customdata[0]}}<br>"
            "Unique Artists: %{customdata[1]:,}"
            "<extra></extra>"
        )
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
        labels={"period": discovery_period_label, "new_artists": "New Artists"},
        custom_data=["period", "new_artists"],
    )

    fig.update_traces(
        hovertemplate=(
            f"{discovery_period_label}: %{{customdata[0]}}<br>"
            "New Artists: %{customdata[1]:,}"
            "<extra></extra>"
        )
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sessions tab
# -----------------------------
with tab_sessions:
    sessions = compute_sessions(df, gap_minutes=SESSION_GAP_MINUTES)

    if sessions.empty:
        st.info("No sessions computed.")

    else:
        c1, c2 = st.columns(2)

        with c1:
            duration_sessions = sessions[sessions["duration_minutes"] > 0].copy()

            if duration_sessions.empty:
                st.info("No multi-track sessions available for duration distribution.")
            else:
                fig = px.histogram(
                    duration_sessions,
                    x="duration_minutes",
                    nbins=50,
                    title="Multi-track Session Duration Distribution",
                    labels={"duration_minutes": "Session Duration in Minutes"},
                )

                fig.update_traces(
                    hovertemplate=(
                        "Session Duration: %{x:.1f} minutes<br>"
                        "Number of Sessions: %{y}"
                        "<extra></extra>"
                    )
                )

                fig.update_layout(
                    xaxis_title="Session Duration in Minutes",
                    yaxis_title="Number of Sessions",
                )

                st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                sessions,
                x="minutes",
                nbins=40,
                title="Minutes per Session Distribution",
                labels={"minutes": "Minutes per Session"},
            )

            fig.update_traces(
                hovertemplate="Minutes per Session: %{x:.1f}<br>Number of Sessions: %{y}<extra></extra>"
            )

            fig.update_layout(
                xaxis_title="Minutes per Session",
                yaxis_title="Number of Sessions",
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Longest sessions")

        longest = sessions.sort_values("minutes", ascending=False).head(20).copy()
        longest["minutes"] = longest["minutes"].round(1)
        longest["duration_minutes"] = longest["duration_minutes"].round(1)
        longest["session_start_local"] = (
            longest["session_start"]
            .dt.tz_convert(selected_timezone)
            .dt.tz_localize(None)
        )
        longest["session_date"] = longest["session_start_local"].dt.date

        longest_display = longest[["session_date", "minutes", "duration_minutes", "plays"]].rename(
            columns={
                "session_date": "Session Date",
                "minutes": "Minutes",
                "duration_minutes": "Duration Minutes",
                "plays": "Track Plays",
            }
        )

        st.dataframe(longest_display, use_container_width=True, hide_index=True)

        sessions_plot = sessions.copy()
        sessions_plot["session_start_local"] = (
            sessions_plot["session_start"]
            .dt.tz_convert(selected_timezone)
            .dt.tz_localize(None)
        )
        sessions_plot["session_end_local"] = (
            sessions_plot["session_end"]
            .dt.tz_convert(selected_timezone)
            .dt.tz_localize(None)
        )

        sessions_plot["session_date_display"] = sessions_plot["session_start_local"].dt.date.astype(str)
        sessions_plot["session_start_display"] = sessions_plot["session_start_local"].dt.strftime(
            f"%b %d, %Y %I:%M %p {selected_timezone_label}"
        )
        sessions_plot["session_end_display"] = sessions_plot["session_end_local"].dt.strftime(
            f"%b %d, %Y %I:%M %p {selected_timezone_label}"
        )

        fig = px.scatter(
            sessions_plot,
            x="plays",
            y="minutes",
            title="Plays vs. Minutes per Session",
            labels={"plays": "Plays", "minutes": "Minutes"},
            custom_data=[
                "session_date_display",
                "session_start_display",
                "session_end_display",
                "duration_minutes",
                "plays",
                "minutes",
            ],
        )

        fig.update_traces(
            hovertemplate=(
                "Date: %{customdata[0]}<br>"
                "Start: %{customdata[1]}<br>"
                "End: %{customdata[2]}<br>"
                "Duration: %{customdata[3]:,.1f} minutes<br>"
                "Plays: %{customdata[4]:,}<br>"
                "Minutes: %{customdata[5]:,.1f}"
                "<extra></extra>"
            )
        )

        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Info tab
# -----------------------------
with tab_info:
    st.subheader("About this app")

    st.markdown(
        f"""
        **Spotify Statistics** is an interactive dashboard for exploring your Spotify Extended Streaming History.

        The app analyzes your listening history and creates charts for:

        - Top artists, tracks, and albums
        - Track repeat behavior
        - Listening patterns by day of week and hour of day
        - Most-listened days
        - Consecutively repeated songs
        - Listening trends over time
        - Artist diversity and new artist discovery
        - Listening sessions

        ### Data used

        This app is designed for Spotify's **Extended Streaming History** export. It looks for audio streaming history JSON files
        inside the ZIP file Spotify provides.

        The app uses these fields from Spotify's data:

        - `ts`: timestamp of the stream
        - `ms_played`: milliseconds played
        - `master_metadata_track_name`: track name
        - `master_metadata_album_artist_name`: artist name
        - `master_metadata_album_album_name`: album name

        These fields are renamed inside the app as:

        - `played_at`
        - `ms_played`
        - `track`
        - `artist`
        - `album`

        The app then creates additional fields for analysis, including:

        - `minutes`
        - `date`
        - `day`
        - `week`
        - `month`
        - `year`
        - `hour`
        - `day_of_week`
        - `played_at_local`

        ### Stream filtering

        To better match Spotify's stream-counting logic, this app only includes tracks played for at least
        **{MIN_STREAM_SECONDS} seconds**.

        ### Timezone

        The timezone selector controls day-of-week, hour-of-day, most-listened days, and trend grouping.
        Select the timezone that best reflects where you usually listen.

        ### Privacy

        The app does **not** require your Spotify login.

        Your data is **not stored, saved, captured, or sent anywhere outside of the active Streamlit session**.
        The uploaded ZIP file is only used to generate the dashboard while the app session is active.

        ### How to request your data

        Request your Spotify Extended Streaming History here:  
        [Spotify Privacy / Account Data Request](https://www.spotify.com/us/account/privacy/)

        ### Example data

        If no file is uploaded, the app shows fake example data so users can preview the dashboard before uploading
        personal Spotify data.

        ### Source code

        [View this project on GitHub](https://github.com/JonahJ5/spotifystatistics)
        """
    )


# -----------------------------
# Export section
# -----------------------------
st.divider()
st.subheader("Share / Export")

st.markdown(
    """
    Download a shareable summary if you want to send your results to friends.
    Technical JSON and CSV exports are also available below.
    """
)

try:
    pdf_bytes = build_shareable_pdf(df, topn=min(topn, 10))

    st.download_button(
        "Download Shareable Summary (PDF)",
        data=pdf_bytes,
        file_name="spotify_statistics_summary.pdf",
        mime="application/pdf",
    )

except ImportError:
    st.warning(
        "PDF export requires the `reportlab` and `kaleido` packages. Add both to requirements.txt to enable this feature."
    )

except Exception as e:
    st.warning("PDF export is temporarily unavailable. The interactive dashboard and technical exports still work.")
    st.caption(f"PDF error: {e}")

with st.expander("Technical exports: JSON and CSV", expanded=False):
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
            st.download_button(
                label,
                data=exports[fname],
                file_name=fname,
                mime="text/csv",
            )

    zip_bytes = make_zip_bytes(exports)

    st.download_button(
        "Download ALL technical exports as ZIP",
        data=zip_bytes,
        file_name="wrapped_exports.zip",
        mime="application/zip",
    )
