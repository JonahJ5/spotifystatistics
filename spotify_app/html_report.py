import html
from typing import List

import pandas as pd
import plotly.express as px

from spotify_app.config import SESSION_GAP_MINUTES
from spotify_app.data_transform import (
    auto_hbar_height,
    compute_sessions,
    format_hour_label,
    safe_group_count,
    safe_group_sum,
)


def _fig_html(fig) -> str:
    """Convert a Plotly figure into an embeddable HTML block."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": True, "responsive": True},
    )


def _section(title: str, body: str) -> str:
    return f"""
    <section class="section">
        <h2>{html.escape(title)}</h2>
        {body}
    </section>
    """


def _metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
    </div>
    """


def _df_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df.empty:
        return "<p>No data available for this section.</p>"

    return df.head(max_rows).to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True,
    )


def build_dashboard_html_report(
    df: pd.DataFrame,
    topn: int,
    selected_timezone_label: str = "Selected timezone",
) -> bytes:
    """Build a shareable HTML report with dashboard-style charts."""
    date_source = "played_at_local" if "played_at_local" in df.columns else "played_at"
    start_date = pd.to_datetime(df[date_source]).min().strftime("%b %d, %Y")
    end_date = pd.to_datetime(df[date_source]).max().strftime("%b %d, %Y")

    total_minutes = df["minutes"].sum()
    total_hours = total_minutes / 60

    metrics_html = "".join(
        [
            _metric_card("Total Hours", f"{total_hours:,.1f}"),
            _metric_card("Total Minutes", f"{total_minutes:,.0f}"),
            _metric_card("Track Plays", f"{len(df):,}"),
            _metric_card("Unique Artists", f"{df['artist'].nunique():,}"),
            _metric_card("Unique Tracks", f"{df['track'].nunique():,}"),
            _metric_card("Unique Albums", f"{df['album'].nunique():,}"),
        ]
    )

    sections: List[str] = []

    # -----------------------------
    # Rankings
    # -----------------------------
    top_artists = safe_group_sum(df, ["artist"], "minutes", topn)
    plot_df = top_artists.sort_values("minutes", ascending=True)

    fig_top_artists = px.bar(
        plot_df,
        x="minutes",
        y="artist",
        orientation="h",
        title="Top Artists by Minutes",
        color="artist",
        labels={"artist": "Artist", "minutes": "Minutes"},
        custom_data=["artist", "minutes"],
    )
    fig_top_artists.update_traces(
        hovertemplate="Artist: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
    )
    fig_top_artists.update_layout(
        showlegend=False,
        height=auto_hbar_height(len(plot_df)),
        yaxis=dict(automargin=True),
        margin=dict(l=260, r=20, t=60, b=40),
    )

    top_tracks = safe_group_sum(df, ["track", "artist", "album"], "minutes", topn)
    plot_df = top_tracks.sort_values("minutes", ascending=True)

    fig_top_tracks = px.bar(
        plot_df,
        x="minutes",
        y="track",
        orientation="h",
        title="Top Tracks by Minutes",
        color="track",
        labels={"track": "Track", "artist": "Artist", "album": "Album", "minutes": "Minutes"},
        custom_data=["track", "artist", "album", "minutes"],
    )
    fig_top_tracks.update_traces(
        hovertemplate=(
            "Track: %{customdata[0]}<br>"
            "Artist: %{customdata[1]}<br>"
            "Album: %{customdata[2]}<br>"
            "Minutes: %{customdata[3]:,.1f}"
            "<extra></extra>"
        )
    )
    fig_top_tracks.update_layout(
        showlegend=False,
        height=auto_hbar_height(len(plot_df)),
        yaxis=dict(automargin=True),
        margin=dict(l=320, r=20, t=60, b=40),
    )

    top_albums = safe_group_sum(df, ["album", "artist"], "minutes", topn)
    plot_df = top_albums.sort_values("minutes", ascending=True)

    fig_top_albums = px.bar(
        plot_df,
        x="minutes",
        y="album",
        orientation="h",
        title="Top Albums by Minutes",
        color="album",
        labels={"album": "Album", "artist": "Artist", "minutes": "Minutes"},
        custom_data=["album", "artist", "minutes"],
    )
    fig_top_albums.update_traces(
        hovertemplate=(
            "Album: %{customdata[0]}<br>"
            "Artist: %{customdata[1]}<br>"
            "Minutes: %{customdata[2]:,.1f}"
            "<extra></extra>"
        )
    )
    fig_top_albums.update_layout(
        showlegend=False,
        height=auto_hbar_height(len(plot_df)),
        yaxis=dict(automargin=True),
        margin=dict(l=320, r=20, t=60, b=40),
    )

    top_artists_plays = safe_group_count(df, ["artist"], topn, name="plays")
    plot_df = top_artists_plays.sort_values("plays", ascending=True)

    fig_artist_plays = px.bar(
        plot_df,
        x="plays",
        y="artist",
        orientation="h",
        title="Top Artists by Play Count",
        color="artist",
        labels={"artist": "Artist", "plays": "Plays"},
        custom_data=["artist", "plays"],
    )
    fig_artist_plays.update_traces(
        hovertemplate="Artist: %{customdata[0]}<br>Plays: %{customdata[1]:,}<extra></extra>"
    )
    fig_artist_plays.update_layout(
        showlegend=False,
        height=auto_hbar_height(len(plot_df)),
        yaxis=dict(automargin=True),
        margin=dict(l=260, r=20, t=60, b=40),
    )

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

    fig_repeat_dist = px.bar(
        repeat_dist,
        x="plays",
        y="number_of_tracks",
        title="Track Repeat Distribution",
        labels={"plays": "Plays per Track", "number_of_tracks": "Number of Tracks"},
        custom_data=["plays", "number_of_tracks"],
    )
    fig_repeat_dist.update_traces(
        hovertemplate=(
            "Plays per Track: %{customdata[0]}<br>"
            "Number of Tracks: %{customdata[1]:,}"
            "<extra></extra>"
        )
    )

    daily_artist = df.groupby(["artist", "day"], as_index=False)["minutes"].sum()
    idx = daily_artist.groupby("artist")["minutes"].idxmax()
    artist_peaks = daily_artist.loc[idx].copy()

    artist_peaks = artist_peaks.sort_values("minutes", ascending=False).head(topn).copy()
    artist_peaks["date"] = artist_peaks["day"].dt.date
    artist_peaks["label"] = artist_peaks["artist"] + " — " + artist_peaks["date"].astype(str)
    artist_peaks["peak_hours"] = artist_peaks["minutes"] / 60.0

    plot_df = artist_peaks.sort_values("minutes", ascending=True)

    fig_artist_peaks = px.bar(
        plot_df,
        x="minutes",
        y="label",
        orientation="h",
        title="Top Artist Days",
        color="artist",
        labels={"label": "Artist and Date", "minutes": "Minutes", "peak_hours": "Hours"},
        custom_data=["artist", "date", "minutes", "peak_hours"],
    )
    fig_artist_peaks.update_traces(
        hovertemplate=(
            "Artist: %{customdata[0]}<br>"
            "Date: %{customdata[1]}<br>"
            "Minutes: %{customdata[2]:,.1f}<br>"
            "Hours: %{customdata[3]:,.2f}"
            "<extra></extra>"
        )
    )
    fig_artist_peaks.update_layout(
        showlegend=False,
        height=auto_hbar_height(len(plot_df)),
        yaxis=dict(automargin=True),
        margin=dict(l=340, r=20, t=60, b=40),
    )

    sections.append(
        _section(
            "Rankings",
            f"""
            <div class="chart-grid">
                <div>{_fig_html(fig_top_artists)}</div>
                <div>{_fig_html(fig_top_tracks)}</div>
                <div>{_fig_html(fig_top_albums)}</div>
                <div>{_fig_html(fig_artist_plays)}</div>
            </div>
            <div class="chart-full">{_fig_html(fig_repeat_dist)}</div>
            <div class="chart-full">{_fig_html(fig_artist_peaks)}</div>
            """,
        )
    )

    # -----------------------------
    # Time Patterns
    # -----------------------------
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hour_order = [format_hour_label(h) for h in range(24)]

    by_dow = df.groupby("day_of_week", as_index=False)["minutes"].sum()
    by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], categories=order, ordered=True)
    by_dow = by_dow.sort_values("day_of_week")

    fig_dow = px.bar(
        by_dow,
        x="day_of_week",
        y="minutes",
        title=f"Minutes by Day of the Week — {selected_timezone_label}",
        color="day_of_week",
        labels={"day_of_week": "Day of the Week", "minutes": "Minutes"},
        custom_data=["day_of_week", "minutes"],
    )
    fig_dow.update_traces(
        hovertemplate="Day of the Week: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
    )
    fig_dow.update_layout(showlegend=False)

    by_hour = df.groupby(["hour", "hour_label"], as_index=False)["minutes"].sum().sort_values("hour")
    by_hour["hour_label"] = pd.Categorical(by_hour["hour_label"], categories=hour_order, ordered=True)
    by_hour = by_hour.sort_values("hour_label")

    fig_hour = px.bar(
        by_hour,
        x="hour_label",
        y="minutes",
        title=f"Minutes by Hour of Day — {selected_timezone_label}",
        color="hour",
        labels={"hour_label": "Hour of Day", "minutes": "Minutes"},
        custom_data=["hour_label", "minutes"],
    )
    fig_hour.update_traces(
        hovertemplate="Hour of Day: %{customdata[0]}<br>Minutes: %{customdata[1]:,.1f}<extra></extra>"
    )
    fig_hour.update_layout(showlegend=False)

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
        repeated_html = "<p>No consecutive repeated tracks found for the current selection.</p>"
    else:
        streaks["label"] = streaks["track"] + " — " + streaks["artist"]
        streaks["streak_start_display"] = streaks["streak_start"].dt.strftime("%b %d, %Y %I:%M %p")
        streaks["streak_end_display"] = streaks["streak_end"].dt.strftime("%b %d, %Y %I:%M %p")

        plot_df = streaks.sort_values("repeat_count", ascending=True)

        fig_repeated = px.bar(
            plot_df,
            x="repeat_count",
            y="label",
            orientation="h",
            title="Most Consecutively Repeated Songs",
            color="artist",
            labels={"repeat_count": "Consecutive Plays", "label": "Track", "artist": "Artist"},
            custom_data=["track", "artist", "repeat_count", "streak_start_display", "streak_end_display"],
        )
        fig_repeated.update_traces(
            hovertemplate=(
                "Track: %{customdata[0]}<br>"
                "Artist: %{customdata[1]}<br>"
                "Consecutive Plays: %{customdata[2]:,}<br>"
                "Start: %{customdata[3]}<br>"
                "End: %{customdata[4]}"
                "<extra></extra>"
            )
        )
        fig_repeated.update_layout(
            showlegend=False,
            height=auto_hbar_height(len(plot_df)),
            yaxis=dict(automargin=True),
            margin=dict(l=340, r=20, t=60, b=40),
        )
        repeated_html = _fig_html(fig_repeated)

    sections.append(
        _section(
            "Time Patterns",
            f"""
            <div class="chart-grid">
                <div>{_fig_html(fig_dow)}</div>
                <div>{_fig_html(fig_hour)}</div>
            </div>
            <h3>Most-listened days</h3>
            {_df_table(daily_display, max_rows=25)}
            <div class="chart-full">{repeated_html}</div>
            """,
        )
    )

    # -----------------------------
    # Trends
    # -----------------------------
    listening_trend = (
        df.groupby("day", as_index=False)["minutes"].sum()
        .sort_values("day")
    )
    listening_trend["hours"] = listening_trend["minutes"] / 60.0

    fig_listening = px.line(
        listening_trend,
        x="day",
        y="hours",
        title="Hours Over Time by Day",
        labels={"day": "Day", "hours": "Hours"},
        custom_data=["day", "hours"],
    )
    fig_listening.update_traces(
        hovertemplate="Day: %{customdata[0]}<br>Hours: %{customdata[1]:,.2f}<extra></extra>"
    )

    cumulative = listening_trend.copy()
    cumulative["cumulative_hours"] = cumulative["hours"].cumsum()

    fig_cumulative = px.line(
        cumulative,
        x="day",
        y="cumulative_hours",
        title="Cumulative Hours Over Time by Day",
        labels={"day": "Day", "cumulative_hours": "Cumulative Hours"},
        custom_data=["day", "cumulative_hours"],
    )
    fig_cumulative.update_traces(
        hovertemplate="Day: %{customdata[0]}<br>Cumulative Hours: %{customdata[1]:,.2f}<extra></extra>"
    )

    artist_period_col = "month"
    artist_period_label = "Month"

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

    fig_artist_trend = px.area(
        trend,
        x=artist_period_col,
        y="hours",
        color="artist",
        title=f"Top Artists Over Time by {artist_period_label}",
        labels={artist_period_col: artist_period_label, "hours": "Hours", "artist": "Artist"},
        custom_data=[artist_period_col, "artist", "hours"],
    )
    fig_artist_trend.update_traces(
        hovertemplate=(
            f"{artist_period_label}: %{{customdata[0]}}<br>"
            "Artist: %{customdata[1]}<br>"
            "Hours: %{customdata[2]:,.2f}"
            "<extra></extra>"
        )
    )

    diversity = (
        df.groupby(artist_period_col, as_index=False)
        .agg(unique_artists=("artist", "nunique"))
        .sort_values(artist_period_col)
    )

    fig_diversity = px.line(
        diversity,
        x=artist_period_col,
        y="unique_artists",
        title=f"Artist Diversity Over Time by {artist_period_label}",
        labels={artist_period_col: artist_period_label, "unique_artists": "Unique Artists"},
        custom_data=[artist_period_col, "unique_artists"],
    )
    fig_diversity.update_traces(
        hovertemplate=(
            f"{artist_period_label}: %{{customdata[0]}}<br>"
            "Unique Artists: %{customdata[1]:,}"
            "<extra></extra>"
        )
    )

    first_artist = (
        df.sort_values("played_at")
        .groupby("artist", as_index=False)
        .first()[["artist", "played_at"]]
    )
    first_artist["period"] = first_artist["played_at"].dt.strftime("%Y-%m")

    discovery = (
        first_artist.groupby("period", as_index=False)
        .size()
        .rename(columns={"size": "new_artists"})
        .sort_values("period")
    )

    fig_discovery = px.bar(
        discovery,
        x="period",
        y="new_artists",
        title="New Artists Discovered by Month",
        color="period",
        labels={"period": "Month", "new_artists": "New Artists"},
        custom_data=["period", "new_artists"],
    )
    fig_discovery.update_traces(
        hovertemplate="Month: %{customdata[0]}<br>New Artists: %{customdata[1]:,}<extra></extra>"
    )
    fig_discovery.update_layout(showlegend=False)

    sections.append(
        _section(
            "Trends",
            f"""
            <div class="chart-grid">
                <div>{_fig_html(fig_listening)}</div>
                <div>{_fig_html(fig_cumulative)}</div>
                <div>{_fig_html(fig_artist_trend)}</div>
                <div>{_fig_html(fig_diversity)}</div>
            </div>
            <div class="chart-full">{_fig_html(fig_discovery)}</div>
            """,
        )
    )

    # -----------------------------
    # Sessions & Behavior
    # -----------------------------
    sessions = compute_sessions(df, gap_minutes=SESSION_GAP_MINUTES)

    if sessions.empty:
        sessions_body = "<p>No sessions computed.</p>"
    else:
        duration_sessions = sessions[sessions["duration_minutes"] > 0].copy()

        if duration_sessions.empty:
            duration_html = "<p>No multi-track sessions available for duration distribution.</p>"
        else:
            fig_duration = px.histogram(
                duration_sessions,
                x="duration_minutes",
                nbins=50,
                title="Multi-track Session Duration Distribution",
                labels={"duration_minutes": "Session Duration in Minutes"},
            )
            fig_duration.update_traces(
                hovertemplate=(
                    "Session Duration: %{x:.1f} minutes<br>"
                    "Number of Sessions: %{y}"
                    "<extra></extra>"
                )
            )
            duration_html = _fig_html(fig_duration)

        fig_minutes = px.histogram(
            sessions,
            x="minutes",
            nbins=40,
            title="Minutes per Session Distribution",
            labels={"minutes": "Minutes per Session"},
        )
        fig_minutes.update_traces(
            hovertemplate="Minutes per Session: %{x:.1f}<br>Number of Sessions: %{y}<extra></extra>"
        )

        fig_scatter = px.scatter(
            sessions,
            x="plays",
            y="minutes",
            title="Plays vs. Minutes per Session",
            labels={"plays": "Plays", "minutes": "Minutes"},
            custom_data=["session_date", "duration_minutes", "plays", "minutes"],
        )
        fig_scatter.update_traces(
            hovertemplate=(
                "Date: %{customdata[0]}<br>"
                "Duration: %{customdata[1]:,.1f} minutes<br>"
                "Plays: %{customdata[2]:,}<br>"
                "Minutes: %{customdata[3]:,.1f}"
                "<extra></extra>"
            )
        )

        longest = sessions.sort_values("minutes", ascending=False).head(20).copy()
        longest["minutes"] = longest["minutes"].round(1)
        longest["duration_minutes"] = longest["duration_minutes"].round(1)
        longest["session_date"] = longest["session_start"].dt.date

        longest_display = longest[["session_date", "minutes", "duration_minutes", "plays"]].rename(
            columns={
                "session_date": "Session Date",
                "minutes": "Minutes",
                "duration_minutes": "Duration Minutes",
                "plays": "Track Plays",
            }
        )

        sessions_body = f"""
        <div class="chart-grid">
            <div>{duration_html}</div>
            <div>{_fig_html(fig_minutes)}</div>
        </div>
        <div class="chart-full">{_fig_html(fig_scatter)}</div>
        <h3>Longest sessions</h3>
        {_df_table(longest_display, max_rows=20)}
        """

    sections.append(_section("Sessions & Behavior", sessions_body))

    body = "\n".join(sections)

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Spotify Statistics Report</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            body {{
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: #f6f7f8;
                color: #191414;
            }}

            .page {{
                max-width: 1280px;
                margin: 0 auto;
                padding: 32px;
            }}

            .hero {{
                background: linear-gradient(135deg, #191414, #1DB954);
                color: white;
                border-radius: 24px;
                padding: 32px;
                margin-bottom: 24px;
            }}

            .hero h1 {{
                margin: 0 0 8px 0;
                font-size: 40px;
            }}

            .hero p {{
                margin: 6px 0;
                font-size: 16px;
            }}

            .metrics {{
                display: grid;
                grid-template-columns: repeat(6, minmax(140px, 1fr));
                gap: 14px;
                margin-bottom: 24px;
            }}

            .metric-card {{
                background: white;
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
            }}

            .metric-label {{
                color: #666;
                font-size: 13px;
                margin-bottom: 8px;
            }}

            .metric-value {{
                font-size: 24px;
                font-weight: 700;
            }}

            .section {{
                background: white;
                border-radius: 24px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
            }}

            .section h2 {{
                margin-top: 0;
                font-size: 28px;
            }}

            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 18px;
            }}

            .chart-full {{
                margin-top: 18px;
            }}

            .data-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 12px;
                margin-bottom: 18px;
                font-size: 14px;
            }}

            .data-table th {{
                background: #1DB954;
                color: white;
                text-align: left;
                padding: 10px;
            }}

            .data-table td {{
                border-bottom: 1px solid #e5e5e5;
                padding: 9px 10px;
            }}

            .footer {{
                color: #666;
                font-size: 13px;
                text-align: center;
                padding: 24px;
            }}

            @media print {{
                body {{
                    background: white;
                }}

                .section {{
                    box-shadow: none;
                    page-break-inside: avoid;
                }}

                .hero {{
                    box-shadow: none;
                }}
            }}

            @media (max-width: 900px) {{
                .metrics {{
                    grid-template-columns: repeat(2, minmax(140px, 1fr));
                }}

                .chart-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>

    <body>
        <div class="page">
            <div class="hero">
                <h1>Spotify Statistics Report</h1>
                <p>Created with Spotify Statistics by Jonah Jutzi</p>
                <p>Listening history range: {html.escape(start_date)} to {html.escape(end_date)}</p>
            </div>

            <div class="metrics">
                {metrics_html}
            </div>

            {body}

            <div class="footer">
                Spotify Statistics · Interactive HTML export
            </div>
        </div>
    </body>
    </html>
    """

    return full_html.encode("utf-8")
